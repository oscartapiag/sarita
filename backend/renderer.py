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


# Default vibrant colors for each stem (customizable)
STEM_COLORS = {
    "drums": "#FF6B35",    # Orange
    "bass": "#C41E3A",     # Deep Red  
    "vocals": "#FFD700",   # Gold
    "guitar": "#1E90FF",   # Electric Blue
    "piano": "#9B59B6",    # Purple
    "other": "#20B2AA",    # Teal
}

# Waveform display parameters  
NUM_WAVEFORM_POINTS = 150  # Points per waveform slice

# Preferred stem order: rhythm section together, then melodic, then other
STEM_ORDER = ["drums", "bass", "guitar", "piano", "vocals", "other"]

# Layout constraints
MAX_STEM_WIDTH = 6.0  # Maximum width per stem


@dataclass
class StemData:
    """Data for a single stem visualization."""
    name: str
    color: str
    envelope: np.ndarray
    waveform: np.ndarray      # Full filtered waveform for visualization
    sample_rate: int          # Sample rate of the waveform
    total_frames: int         # Total frames in the video
    threshold: float          # Dynamic noise threshold (per-stem)
    analysis_fps: int         # FPS of the envelope data (for time-based sync)


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
        
    def construct(self):
        """Build scene with fixed layout, per-frame opacity."""
        self.camera.background_color = BLACK
        
        num_stems = len(self.stems)
        if num_stems == 0:
            self.wait(self.audio_duration)
            return
        
        line_height = 5.0
        self.current_frame = 0
        
        # Fixed layout: fill available space (up to MAX_STEM_WIDTH each)
        width_per_stem = min(MAX_STEM_WIDTH, 12.0 / num_stems)
        group_width = width_per_stem * num_stems
        left_x = -group_width / 2 + width_per_stem / 2
        
        # Create lines and labels at fixed positions
        for i, stem in enumerate(self.stems):
            x_pos = left_x + i * width_per_stem
            
            line = self._create_line(x_pos, line_height, stem.color)
            
            label = None
            if self.show_labels:
                label = Text(stem.name.capitalize(), font_size=18, color=stem.color)
                label.move_to([x_pos, -3.2, 0])
                self.add(label)
            
            # Each line gets its own updater (fixed position, only opacity changes)
            line.add_updater(self._make_fixed_updater(stem, x_pos, line_height, width_per_stem, label))
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
                            max_displacement: float, label: Optional[Text]):
        """Create an updater for fixed position stem - only opacity and waveform change."""
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
            
            # Update the waveform at fixed position
            self._update_raw_waveform(line, stem, x_pos, height, amplitude, 
                                      frame, max_displacement)
            
            # Set opacity (thinner line for more detail)
            line.set_stroke(opacity=opacity, width=3)
            
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
    
    def _update_raw_waveform(self, line: VMobject, stem: StemData, x_center: float, 
                             height: float, amplitude: float, frame: int,
                             max_displacement: float):
        """Update line with raw audio waveform."""
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
) -> Path:
    """Render the oscilloscope visualization to a video file."""
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
    
    scene = AMOscilloscope(
        stems=active_stems,
        duration=duration,
        fps=settings["frame_rate"],
        show_labels=show_labels,
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
