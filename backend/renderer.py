"""
Oscilloscope visualization renderer using Manim.

Creates animated VERTICAL waveform visualizations for each audio stem.
Dynamic visibility: stems fade based on amplitude threshold.
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

# Visibility threshold (0-1)
DEFAULT_THRESHOLD = 0.1


@dataclass
class StemData:
    """Data for a single stem visualization."""
    name: str
    color: str
    envelope: np.ndarray
    waveform: np.ndarray
    threshold: float = DEFAULT_THRESHOLD


class DynamicOscilloscope(Scene):
    """
    Manim scene with VERTICAL oscilloscope lines.
    Dynamic visibility based on amplitude.
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
        """Build and animate the scene."""
        self.camera.background_color = BLACK
        
        num_stems = len(self.stems)
        
        # Calculate X positions for vertical lines (spread across screen)
        total_width = 12.0
        width_per_stem = total_width / num_stems
        left_x = -total_width / 2 + width_per_stem / 2
        
        # Line height (vertical extent)
        line_height = 5.0
        
        # Create lines for each stem
        lines = []
        labels = []
        x_positions = []
        
        for i, stem in enumerate(self.stems):
            x_pos = left_x + i * width_per_stem
            x_positions.append(x_pos)
            
            # Create vertical line
            line = self.create_vertical_line(x_pos, line_height, stem.color)
            lines.append(line)
            self.add(line)
            
            # Add label below
            if self.show_labels:
                label = Text(stem.name.capitalize(), font_size=18, color=stem.color)
                label.move_to([x_pos, -3.2, 0])
                labels.append(label)
                self.add(label)
        
        # Animate frame by frame
        frame_duration = 1.0 / self.target_fps
        
        for frame in range(self.total_frames):
            if frame % 50 == 0:
                print(f"\r   └── Rendering frame {frame}/{self.total_frames} ({100*frame/self.total_frames:.1f}%)", end="", flush=True)
            
            # Update each line
            for i, (stem, line, x_pos) in enumerate(zip(self.stems, lines, x_positions)):
                # Get envelope value using TIME-based lookup (handles FPS mismatch)
                current_time = frame / self.target_fps
                env_idx = int(current_time * len(stem.envelope) / self.audio_duration)
                env_idx = min(max(0, env_idx), len(stem.envelope) - 1)
                envelope_val = float(stem.envelope[env_idx])
                
                # Dynamic visibility: calculate opacity based on envelope
                if envelope_val > stem.threshold:
                    opacity = min(1.0, 0.3 + envelope_val * 0.7)
                else:
                    opacity = envelope_val * 3  # Fade out below threshold
                
                # Get waveform slice
                waveform_slice = self.get_waveform_slice(stem, frame)
                
                # Update vertical line with waveform
                amplitude = envelope_val * width_per_stem * 0.4  # Horizontal displacement
                self.update_vertical_line(line, waveform_slice, x_pos, line_height, amplitude)
                
                # Set opacity and width based on amplitude
                stroke_width = 2 + envelope_val * 4
                line.set_stroke(opacity=opacity, width=stroke_width)
                
                # Update label opacity
                if self.show_labels and i < len(labels):
                    labels[i].set_opacity(opacity)
            
            self.wait(frame_duration)
        
        print()
    
    def create_vertical_line(self, x_pos: float, height: float, color: str, 
                              num_points: int = 150) -> VMobject:
        """Create a vertical line at the given x position."""
        y_vals = np.linspace(-height/2, height/2, num_points)
        points = [np.array([x_pos, y, 0]) for y in y_vals]
        
        line = VMobject()
        line.set_points_as_corners(points)
        line.set_stroke(color=color, width=3, opacity=0.8)
        return line
    
    def update_vertical_line(self, line: VMobject, waveform: np.ndarray, 
                              x_center: float, height: float, amplitude: float,
                              num_points: int = 150):
        """Update vertical line with waveform displacement."""
        # Ensure correct number of points
        if len(waveform) != num_points:
            indices = np.linspace(0, len(waveform) - 1, num_points).astype(int)
            indices = np.clip(indices, 0, len(waveform) - 1)
            waveform = waveform[indices]
        
        y_vals = np.linspace(-height/2, height/2, num_points)
        # Horizontal displacement based on waveform
        x_vals = x_center + waveform * amplitude
        
        points = [np.array([x, y, 0]) for x, y in zip(x_vals, y_vals)]
        line.set_points_as_corners(points)
    
    def get_waveform_slice(self, stem: StemData, frame: int, num_points: int = 150) -> np.ndarray:
        """Get a slice of the waveform for the current frame."""
        waveform = stem.waveform
        progress = frame / max(1, self.total_frames)
        
        center_sample = int(progress * len(waveform))
        window_size = max(num_points * 4, len(waveform) // max(1, self.total_frames) * 4)
        
        start = max(0, center_sample - window_size // 2)
        end = min(len(waveform), center_sample + window_size // 2)
        
        if end <= start:
            return np.zeros(num_points)
        
        window = waveform[start:end]
        
        if len(window) > 0:
            indices = np.linspace(0, len(window) - 1, num_points).astype(int)
            indices = np.clip(indices, 0, len(window) - 1)
            return window[indices]
        
        return np.zeros(num_points)


def render_oscilloscope(
    analysis_result,
    output_path: Path,
    quality: str = "medium",
    show_labels: bool = False,
    custom_colors: Optional[dict] = None,
    preview_duration: Optional[float] = None,
) -> Path:
    """
    Render the oscilloscope visualization to a video file.
    """
    colors = {**STEM_COLORS, **(custom_colors or {})}
    
    # Build stem data
    stems = []
    for stem_name, envelope in analysis_result.as_dict().items():
        if envelope is None:
            continue
        stems.append(StemData(
            name=stem_name,
            color=colors.get(stem_name, "#FFFFFF"),
            envelope=envelope.fps_envelope,
            waveform=envelope.waveform,
        ))
    
    duration = preview_duration if preview_duration else analysis_result.duration
    
    quality_settings = {
        "low": {"pixel_height": 480, "pixel_width": 854, "frame_rate": 24},
        "medium": {"pixel_height": 720, "pixel_width": 1280, "frame_rate": 30},
        "high": {"pixel_height": 1080, "pixel_width": 1920, "frame_rate": 60},
    }
    settings = quality_settings.get(quality, quality_settings["medium"])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config.pixel_height = settings["pixel_height"]
    config.pixel_width = settings["pixel_width"]
    config.frame_rate = settings["frame_rate"]
    config.output_file = output_path.stem
    config.media_dir = str(output_path.parent)
    config.write_to_movie = True
    config.disable_caching = True
    
    print(f"   └── Rendering {len(stems)} stems at {settings['pixel_width']}x{settings['pixel_height']} @ {settings['frame_rate']}fps")
    print(f"   └── Duration: {duration:.1f}s ({int(duration * settings['frame_rate'])} frames)")
    
    scene = DynamicOscilloscope(
        stems=stems,
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
