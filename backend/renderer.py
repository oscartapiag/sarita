"""
Oscilloscope visualization renderer using Manim.

Creates animated VERTICAL waveform visualizations for each audio stem.
Uses Manim's native updater pattern for efficient rendering.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from manim import *

from backend.color_palettes import get_palette_for_key, DEFAULT_COLORS
from backend.harmonic_colors import get_frame_color


# Default vibrant colors for each stem (now imported from color_palettes)
STEM_COLORS = DEFAULT_COLORS

# Enable per-frame harmonic colors (can be disabled for performance)
USE_HARMONIC_COLORS = True

# Waveform display parameters  
NUM_WAVEFORM_POINTS = 150  # Points per waveform slice

# Preferred stem order: rhythm section together, then melodic, then other
STEM_ORDER = ["drums", "bass", "guitar", "piano", "vocals", "other"]

# Layout constraints
MAX_STEM_WIDTH = 6.0  # Maximum width per stem

# Lissajous curve parameters (iTunes-style flowing curves)
LISSAJOUS_ENABLED = True
LISSAJOUS_SPEED = 0.2          # How fast the phase animates (rotations per second)
LISSAJOUS_BASE_SIZE = 2.0      # Base curve size (in scene units)

# Glow effect parameters
GLOW_ENABLED = True
GLOW_LAYERS = 3                # Number of glow layers behind each line
GLOW_WIDTH_MULT = 3.0          # How much wider each glow layer is

# Dynamic Lissajous frequency ranges (derived from audio per-frame)
# freq_y: from dominant pitch (low notes → simple, high notes → complex)
FREQ_Y_MIN = 1.5   # Minimum freq_y (for lowest pitches)
FREQ_Y_MAX = 5.0   # Maximum freq_y (for highest pitches)

# freq_x: from spectral brightness (dark → simple, bright → complex)
FREQ_X_MIN = 1.0   # Minimum freq_x (for dark/muted sounds)
FREQ_X_MAX = 4.0   # Maximum freq_x (for bright/harsh sounds)

# Per-stem phase offsets (to spread stems apart visually)
STEM_PHASE_OFFSETS = {
    "drums":  0.0,
    "bass":   0.25,
    "vocals": 0.5,
    "guitar": 0.75,
    "piano":  0.33,
    "other":  0.66,
}


@dataclass
class StemData:
    """Data for a single stem visualization."""
    name: str
    color: str                    # Base/fallback color
    envelope: np.ndarray
    waveform: np.ndarray          # Full filtered waveform for visualization
    sample_rate: int              # Sample rate of the waveform
    total_frames: int             # Total frames in the video
    threshold: float              # Dynamic noise threshold (per-stem)
    analysis_fps: int             # FPS of the envelope data (for time-based sync)
    
    # Harmonic features for per-frame colors (optional)
    chroma: Optional[np.ndarray] = None              # Shape: (12, num_frames)
    spectral_brightness: Optional[np.ndarray] = None # Shape: (num_frames,)
    onsets: Optional[np.ndarray] = None              # Shape: (num_frames,)


class AMOscilloscope(Scene):
    """
    AM-style oscilloscope with Manim's native updater pattern.
    
    Uses add_updater() for efficient frame-by-frame animation.
    Lines get thicker when louder (key AM aesthetic).
    """
    
    def __init__(
        self,
        stems: list[StemData],
        duration: float,
        fps: int = 30,
        show_labels: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stems = stems
        self.audio_duration = duration
        self.target_fps = fps
        self.show_labels = show_labels
        self.total_frames = int(duration * fps)
        self.use_lissajous = kwargs.pop('use_lissajous', LISSAJOUS_ENABLED)
        
    def construct(self):
        """Build scene with fixed layout, per-frame opacity."""
        self.camera.background_color = BLACK
        
        num_stems = len(self.stems)
        if num_stems == 0:
            self.wait(self.audio_duration)
            return
        
        line_height = 5.0
        self.current_frame = 0
        
        # Full-screen layout: stems fill the entire width (14 units in Manim default)
        screen_width = 14.0
        width_per_stem = screen_width / num_stems
        left_x = -screen_width / 2 + width_per_stem / 2
        
        # Create lines and labels at fixed positions
        for i, stem in enumerate(self.stems):
            x_pos = left_x + i * width_per_stem
            
            # Create glow layers first (so they render behind the main line)
            glow_lines = []
            if GLOW_ENABLED:
                for glow_idx in range(GLOW_LAYERS):
                    glow = self._create_line(x_pos, line_height, stem.color)
                    glow.set_stroke(opacity=0)  # Start invisible
                    self.add(glow)
                    glow_lines.append(glow)
            
            # Create main line
            line = self._create_line(x_pos, line_height, stem.color)
            
            label = None
            if self.show_labels:
                label = Text(stem.name.capitalize(), font_size=18, color=stem.color)
                label.move_to([x_pos, -3.2, 0])
                self.add(label)
            
            # Each line gets its own updater (with glow layers)
            line.add_updater(self._make_fixed_updater(
                stem, x_pos, line_height, width_per_stem, label, glow_lines
            ))
            self.add(line)
        
        # Frame counter
        frame_tracker = ValueTracker(0)
        frame_tracker.add_updater(lambda m, dt: self._increment_frame())
        self.add(frame_tracker)
        
        self.wait(self.audio_duration)
        print(f"\r   └── Rendered {self.current_frame} frames")
    
    def _increment_frame(self):
        """Increment frame counter and show progress."""
        self.current_frame += 1
        if self.current_frame % 100 == 0:
            pct = 100 * self.current_frame / self.total_frames
            print(f"\r   └── Rendering: {pct:.1f}%", end="", flush=True)
    
    def _make_fixed_updater(self, stem: StemData, x_pos: float, height: float, 
                            max_displacement: float, label: Optional[Text],
                            glow_lines: list = None):
        """Create an updater for stem with dynamic audio-driven Lissajous curves and glow."""
        # Get phase offset for this stem (spreads them apart visually)
        phase_offset = STEM_PHASE_OFFSETS.get(stem.name, 0.0) * 2 * np.pi
        glow_lines = glow_lines or []
        
        def update_line(line: VMobject):
            frame = self.current_frame
            
            # Get amplitude from envelope
            playback_time = frame / self.target_fps
            env_idx = int(playback_time * stem.analysis_fps)
            env_idx = min(env_idx, len(stem.envelope) - 1)
            amplitude = float(stem.envelope[env_idx])
            
            # Calculate opacity based on threshold
            if amplitude > stem.threshold:
                opacity = min(1.0, 0.5 + amplitude * 0.5)
            else:
                opacity = 0
            
            # Dynamic frequency ratios from audio features (DISCRETE MODE)
            # Frequencies snap to integer ratios on beat/onset for punchy shape changes
            freq_x = 2  # Default
            freq_y = 3  # Default
            
            if stem.chroma is not None and stem.spectral_brightness is not None:
                feat_idx = min(frame, stem.chroma.shape[1] - 1)
                feat_idx = max(0, feat_idx)
                
                chroma = stem.chroma[:, feat_idx]
                brightness = stem.spectral_brightness[feat_idx]
                onset = stem.onsets[feat_idx] if stem.onsets is not None else 0.0
                
                # Only update shape on onset (transient detected) or first frame
                # This gives a punchy, beat-synced feel
                if onset > 0.3 or frame == 0:
                    # freq_y from dominant pitch: snap to integer
                    pitch_weights = chroma / (np.sum(chroma) + 1e-6)
                    weighted_pitch = np.sum(np.arange(12) * pitch_weights)
                    pitch_normalized = weighted_pitch / 11.0
                    freq_y_raw = FREQ_Y_MIN + pitch_normalized * (FREQ_Y_MAX - FREQ_Y_MIN)
                    freq_y = int(round(freq_y_raw))  # Snap to integer
                    freq_y = max(1, min(6, freq_y))  # Clamp to valid range
                    
                    # freq_x from spectral brightness: snap to integer
                    freq_x_raw = FREQ_X_MIN + brightness * (FREQ_X_MAX - FREQ_X_MIN)
                    freq_x = int(round(freq_x_raw))  # Snap to integer
                    freq_x = max(1, min(5, freq_x))  # Clamp to valid range
                    
                    # Store for next frames
                    self._last_freq_x = freq_x
                    self._last_freq_y = freq_y
                else:
                    # Hold previous shape until next onset
                    freq_x = getattr(self, '_last_freq_x', 2)
                    freq_y = getattr(self, '_last_freq_y', 3)
            
            # Draw Lissajous curve shape with dynamic frequencies
            if self.use_lissajous:
                self._update_lissajous_curve(
                    line, stem, x_pos, amplitude, playback_time,
                    freq_x, freq_y, phase_offset, max_displacement
                )
            else:
                # Fallback to original vertical waveform
                self._update_raw_waveform(line, stem, x_pos, 5.0, amplitude, 
                                          frame, max_displacement)
            
            # Per-frame harmonic color (if available)
            if USE_HARMONIC_COLORS and stem.chroma is not None:
                # Get frame index for features (clamped to valid range)
                feat_idx = min(frame, stem.chroma.shape[1] - 1)
                feat_idx = max(0, feat_idx)
                
                chroma = stem.chroma[:, feat_idx]
                brightness = stem.spectral_brightness[feat_idx] if stem.spectral_brightness is not None else 0.5
                onset = stem.onsets[feat_idx] if stem.onsets is not None else 0.0
                
                frame_color = get_frame_color(
                    stem_name=stem.name,
                    chroma=chroma,
                    spectral_brightness=brightness,
                    onset=onset,
                )
                line.set_stroke(color=frame_color, opacity=opacity, width=3)
                
                # Update glow layers (same color, wider stroke, lower opacity)
                for glow_idx, glow in enumerate(glow_lines):
                    # Copy the same points from main line
                    glow.set_points(line.get_points())
                    # Glow gets wider and fainter with each layer
                    glow_width = 3 + (glow_idx + 1) * GLOW_WIDTH_MULT
                    # Glow opacity scales with amplitude (louder = brighter glow)
                    glow_opacity = opacity * amplitude * 0.3 / (glow_idx + 1)
                    glow.set_stroke(color=frame_color, opacity=glow_opacity, width=glow_width)
            else:
                # Fallback to static color
                line.set_stroke(opacity=opacity, width=3)
                
                # Update glow layers for fallback
                for glow_idx, glow in enumerate(glow_lines):
                    glow.set_points(line.get_points())
                    glow_width = 3 + (glow_idx + 1) * GLOW_WIDTH_MULT
                    glow_opacity = opacity * amplitude * 0.3 / (glow_idx + 1)
                    glow.set_stroke(color=stem.color, opacity=glow_opacity, width=glow_width)
            
            # Update label opacity (position stays fixed)
            if label is not None:
                label.set_opacity(opacity)
        
        return update_line
    
    def _create_line(self, x_pos: float, height: float, color: str) -> VMobject:
        """Create initial vertical line."""
        y_vals = np.linspace(-height/2, height/2, 150)
        points = [np.array([x_pos, y, 0]) for y in y_vals]
        
        line = VMobject()
        line.set_points_as_corners(points)
        line.set_stroke(color=color, width=3, opacity=0.8)
        return line
    
    def _update_lissajous_curve(self, line: VMobject, stem: StemData, x_center: float,
                                 amplitude: float, playback_time: float,
                                 freq_x: int, freq_y: int, phase_offset: float,
                                 max_width: float):
        """
        Draw a Lissajous curve with audio waveform modulation.
        
        The curve shape flows in figure-8/rose patterns (Lissajous),
        but each point is displaced PERPENDICULAR to the curve
        based on the actual audio waveform — bringing back that
        visceral "feel each note" texture.
        """
        num_points = NUM_WAVEFORM_POINTS
        
        # Get audio waveform slice for this frame
        waveform = stem.waveform
        total_samples = len(waveform)
        frame = int(playback_time * self.target_fps)
        progress = frame / max(stem.total_frames, 1)
        
        center_sample = int(progress * total_samples)
        window_size = max(total_samples // max(stem.total_frames, 1) * 2, num_points * 4)
        
        start = max(0, center_sample - window_size // 2)
        end = min(total_samples, center_sample + window_size // 2)
        
        window = waveform[start:end] if end > start else np.zeros(num_points)
        
        if len(window) > 0:
            indices = np.linspace(0, len(window) - 1, num_points).astype(int)
            wave_slice = window[indices]
        else:
            wave_slice = np.zeros(num_points)
        
        # Parametric parameter runs from 0 to 2π
        t = np.linspace(0, 2 * np.pi, num_points)
        
        # Animating phase for flowing motion
        phase_anim = playback_time * LISSAJOUS_SPEED * 2 * np.pi + phase_offset
        
        # Base Lissajous curve (the "spine")
        # Curve size scales to fill allocated width, modulated by amplitude
        base_size = max_width * 0.9  # Use 90% of allocated width as base
        curve_size = base_size * (0.4 + 0.6 * amplitude)  # Range: 0.36 to 1.0 of base
        
        base_x = curve_size * np.sin(freq_x * t + phase_anim)
        base_y = curve_size * np.sin(freq_y * t)
        
        # Calculate curve tangent (derivative) for perpendicular displacement
        dx_dt = freq_x * curve_size * np.cos(freq_x * t + phase_anim)
        dy_dt = freq_y * curve_size * np.cos(freq_y * t)
        
        # Normal vector (perpendicular to tangent): rotate tangent by 90°
        # Normalize to unit length
        tangent_len = np.sqrt(dx_dt**2 + dy_dt**2) + 1e-6  # Avoid division by zero
        normal_x = -dy_dt / tangent_len
        normal_y = dx_dt / tangent_len
        
        # Displace along normal based on audio waveform
        # Amplitude scales the displacement intensity
        displacement_scale = 0.8 * (0.4 + 0.6 * amplitude)
        displacement = wave_slice * displacement_scale
        
        x_vals = x_center + base_x + normal_x * displacement
        y_vals = base_y + normal_y * displacement
        
        # Soft clamp X to stay within allocated width (allow slight overflow)
        half_width = max_width * 0.48
        x_vals = np.clip(x_vals, x_center - half_width, x_center + half_width)
        
        points = [np.array([x, y, 0]) for x, y in zip(x_vals, y_vals)]
        line.set_points_as_corners(points)
    
    def _update_raw_waveform(self, line: VMobject, stem: StemData, x_center: float, 
                             height: float, amplitude: float, frame: int,
                             max_displacement: float):
        """Update line with raw audio waveform (fallback when Lissajous disabled)."""
        num_points = NUM_WAVEFORM_POINTS
        y_vals = np.linspace(-height/2, height/2, num_points)
        
        # Get waveform slice for this frame
        waveform = stem.waveform
        total_samples = len(waveform)
        progress = frame / max(stem.total_frames, 1)
        
        # Calculate window position centered on current playback position
        center_sample = int(progress * total_samples)
        window_size = total_samples // max(stem.total_frames, 1) * 2  # 2 frames of audio
        window_size = max(window_size, num_points * 4)  # Minimum window size
        
        # Extract window with bounds checking
        start = max(0, center_sample - window_size // 2)
        end = min(total_samples, center_sample + window_size // 2)
        
        window = waveform[start:end] if end > start else np.zeros(num_points)
        
        # Resample to target points
        if len(window) > 0:
            indices = np.linspace(0, len(window) - 1, num_points).astype(int)
            wave_slice = window[indices]
        else:
            wave_slice = np.zeros(num_points)
        
        # Amplitude controls displacement size - louder = bigger horizontal waves
        # Minimum displacement of 0.3 so waveform shape is always visible
        amplitude_factor = 0.3 + amplitude * 0.7  # Range: 0.3 to 1.0
        displacement = max_displacement * 0.7 * amplitude_factor * wave_slice
        x_vals = x_center + displacement
        
        points = [np.array([x, y, 0]) for x, y in zip(x_vals, y_vals)]
        line.set_points_as_corners(points)


def render_oscilloscope(
    analysis_result,
    output_path: Path,
    quality: str = "medium",
    show_labels: bool = False,
    custom_colors: Optional[dict] = None,
    preview_duration: Optional[float] = None,
    use_lissajous: bool = True,
) -> Path:
    """Render the oscilloscope visualization to a video file."""
    # Use key-based colors if available, otherwise fall back to custom/defaults
    if hasattr(analysis_result, 'key_info') and analysis_result.key_info is not None:
        key_colors = get_palette_for_key(analysis_result.key_info)
        colors = {**key_colors, **(custom_colors or {})}
        key_name = f"{analysis_result.key_info.key} {analysis_result.key_info.mode}"
        print(f"   └── 🎵 Detected key: {key_name} (confidence: {analysis_result.key_info.confidence:.0%})")
    else:
        colors = {**STEM_COLORS, **(custom_colors or {})}
    
    # Get quality settings first (needed for total_frames calculation)
    quality_settings = {
        "low": {"pixel_height": 480, "pixel_width": 854, "frame_rate": 24},
        "medium": {"pixel_height": 720, "pixel_width": 1280, "frame_rate": 30},
        "high": {"pixel_height": 1080, "pixel_width": 1920, "frame_rate": 60},
    }
    settings = quality_settings.get(quality, quality_settings["medium"])
    
    duration = preview_duration if preview_duration else analysis_result.duration
    total_frames = int(duration * settings["frame_rate"])
    
    # Build stem data with waveform for visualization
    all_stems = []
    for stem_name, envelope in analysis_result.as_dict().items():
        if envelope is None:
            continue
        all_stems.append(StemData(
            name=stem_name,
            color=colors.get(stem_name, "#FFFFFF"),
            envelope=envelope.fps_envelope,
            waveform=envelope.waveform,
            sample_rate=envelope.sample_rate,
            total_frames=total_frames,
            threshold=envelope.noise_threshold,
            analysis_fps=analysis_result.analysis_fps,
            # Harmonic features for per-frame colors
            chroma=getattr(envelope, 'chroma', None),
            spectral_brightness=getattr(envelope, 'spectral_brightness', None),
            onsets=getattr(envelope, 'onsets', None),
        ))
    
    # Filter out completely empty stems (threshold >= 1.0 means always hidden)
    # Keep partially active stems for per-frame visibility
    active_stems = [s for s in all_stems if s.threshold < 1.0]
    
    # Sort by preferred order
    stem_order_map = {name: i for i, name in enumerate(STEM_ORDER)}
    active_stems.sort(key=lambda s: stem_order_map.get(s.name, 99))
    
    # Handle case where no stems have any content
    if len(active_stems) == 0:
        print("   └── No active stems detected, skipping render")
        return output_path

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config.pixel_height = settings["pixel_height"]
    config.pixel_width = settings["pixel_width"]
    config.frame_rate = settings["frame_rate"]
    config.output_file = output_path.stem
    config.media_dir = str(output_path.parent)
    config.write_to_movie = True
    config.disable_caching = True
    
    active_stem_names = [s.name for s in active_stems]
    print(f"   └── Rendering {len(active_stems)} active stems: {', '.join(active_stem_names)}")
    print(f"   └── Resolution: {settings['pixel_width']}x{settings['pixel_height']} @ {settings['frame_rate']}fps")
    print(f"   └── Duration: {duration:.1f}s ({int(duration * settings['frame_rate'])} frames)")
    
    # Debug: show threshold values for each stem
    for s in active_stems:
        sample_amp = float(s.envelope[len(s.envelope)//2])  # Sample from middle
        print(f"   └── {s.name}: threshold={s.threshold:.3f}, sample_amplitude={sample_amp:.3f}")
    
    print(f"   └── Mode: {'Lissajous curves' if use_lissajous else 'Classic waveforms'}")
    
    scene = AMOscilloscope(
        stems=active_stems,
        duration=duration,
        fps=settings["frame_rate"],
        show_labels=show_labels,
        use_lissajous=use_lissajous,
    )
    scene.render()
    
    video_file = output_path.parent / "videos" / f"{settings['pixel_height']}p{settings['frame_rate']}" / f"{output_path.stem}.mp4"
    
    if video_file.exists():
        return video_file
    
    for mp4 in Path(config.media_dir).rglob("*.mp4"):
        if output_path.stem in str(mp4):
            return mp4
    
    raise FileNotFoundError("Rendered video not found")


if __name__ == "__main__":
    print("Renderer module - use via sarita.py")
