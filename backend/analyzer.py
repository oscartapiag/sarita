"""
Audio analysis using Librosa.

Extracts amplitude envelopes and waveform data from audio stems for visualization.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StemEnvelope:
    """Amplitude envelope data for a single stem."""
    name: str
    envelope: np.ndarray      # Amplitude envelope (downsampled for visualization)
    waveform: np.ndarray      # Full waveform data (mono, normalized)
    sample_rate: int          # Original sample rate
    duration: float           # Duration in seconds
    fps_envelope: np.ndarray  # Envelope resampled to target FPS
    

@dataclass  
class AnalysisResult:
    """Container for all stem analysis results."""
    drums: StemEnvelope
    bass: StemEnvelope
    vocals: StemEnvelope
    guitar: StemEnvelope
    piano: StemEnvelope
    other: StemEnvelope
    duration: float
    fps: int
    
    def all(self) -> list[StemEnvelope]:
        """Return all envelopes as a list."""
        return [self.drums, self.bass, self.vocals, self.guitar, self.piano, self.other]
    
    def as_dict(self) -> dict[str, StemEnvelope]:
        """Return envelopes as a dictionary."""
        return {
            "drums": self.drums,
            "bass": self.bass,
            "vocals": self.vocals,
            "guitar": self.guitar,
            "piano": self.piano,
            "other": self.other,
        }


def analyze_stem(
    audio_path: Path,
    stem_name: str,
    target_fps: int = 60,
    hop_length: int = 512,
) -> StemEnvelope:
    """
    Analyze a single audio stem and extract its amplitude envelope.
    
    Uses absolute peak amplitude per frame for zero-latency sync.
    
    Args:
        audio_path: Path to the stem audio file
        stem_name: Name of the stem (for labeling)
        target_fps: Target frames per second for the envelope
        hop_length: Hop length for envelope extraction (unused, kept for API compat)
        
    Returns:
        StemEnvelope containing the analysis results
    """
    # Load audio (mono for analysis)
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(waveform) / sr
    
    # Normalize waveform to [-1, 1]
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val
    
    # Calculate zero-latency envelope using absolute peak amplitude per frame
    # This has no window latency unlike RMS
    num_frames = int(duration * target_fps)
    samples_per_frame = len(waveform) // num_frames
    
    fps_envelope = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * samples_per_frame
        end = min(start + samples_per_frame, len(waveform))
        if end > start:
            fps_envelope[i] = np.abs(waveform[start:end]).max()
    
    # Light smoothing to reduce spikiness (3-frame moving average)
    kernel = np.ones(3) / 3
    fps_envelope = np.convolve(fps_envelope, kernel, mode='same')
    
    # Normalize envelope to [0, 1]
    env_max = fps_envelope.max()
    if env_max > 0:
        fps_envelope = fps_envelope / env_max
    
    return StemEnvelope(
        name=stem_name,
        envelope=fps_envelope,  # Same as fps_envelope now
        waveform=waveform,
        sample_rate=sr,
        duration=duration,
        fps_envelope=fps_envelope,
    )


def analyze_stems(
    stem_paths: dict[str, Path],
    target_fps: int = 60,
) -> AnalysisResult:
    """
    Analyze all stems and extract amplitude envelopes.
    
    Args:
        stem_paths: Dictionary mapping stem names to file paths
        target_fps: Target frames per second for visualization
        
    Returns:
        AnalysisResult containing all stem envelopes
    """
    envelopes = {}
    duration = 0.0
    
    for stem_name, path in stem_paths.items():
        print(f"   └── Analyzing {stem_name}...")
        envelope = analyze_stem(path, stem_name, target_fps)
        envelopes[stem_name] = envelope
        duration = max(duration, envelope.duration)
    
    return AnalysisResult(
        drums=envelopes.get("drums"),
        bass=envelopes.get("bass"),
        vocals=envelopes.get("vocals"),
        guitar=envelopes.get("guitar"),
        piano=envelopes.get("piano"),
        other=envelopes.get("other"),
        duration=duration,
        fps=target_fps,
    )


def get_waveform_slice(
    envelope: StemEnvelope,
    frame_idx: int,
    total_frames: int,
    num_points: int = 200,
) -> np.ndarray:
    """
    Get a slice of the waveform for a specific frame.
    
    This is used to animate the oscilloscope - for each frame, we show
    a window of the waveform centered on the current playback position.
    
    Args:
        envelope: The stem envelope data
        frame_idx: Current frame index
        total_frames: Total number of frames in the video
        num_points: Number of points to return for the waveform slice
        
    Returns:
        Numpy array of waveform values for this frame
    """
    waveform = envelope.waveform
    progress = frame_idx / total_frames
    
    # Calculate window position
    center_sample = int(progress * len(waveform))
    window_size = len(waveform) // total_frames * 2  # Show 2 frames worth of audio
    window_size = max(window_size, num_points * 4)  # Ensure minimum window size
    
    # Extract window with padding
    start = max(0, center_sample - window_size // 2)
    end = min(len(waveform), center_sample + window_size // 2)
    
    window = waveform[start:end]
    
    # Resample to target number of points
    if len(window) > 0:
        indices = np.linspace(0, len(window) - 1, num_points).astype(int)
        return window[indices]
    else:
        return np.zeros(num_points)


# CLI test
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python -m backend.analyzer <stems_dir>")
        print("Example: python -m backend.analyzer temp/stems")
        sys.exit(1)
    
    stems_dir = Path(sys.argv[1])
    
    # Find stem files
    stem_paths = {}
    for stem_name in ["drums", "bass", "vocals", "guitar", "piano", "other"]:
        pattern = f"*_{stem_name}.wav"
        matches = list(stems_dir.glob(pattern))
        if matches:
            stem_paths[stem_name] = matches[0]
            print(f"Found {stem_name}: {matches[0].name}")
    
    if not stem_paths:
        print("No stem files found!")
        sys.exit(1)
    
    print()
    print("🎵 Analyzing stems...")
    result = analyze_stems(stem_paths, target_fps=60)
    
    print()
    print(f"📊 Analysis complete:")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   FPS: {result.fps}")
    print(f"   Frames: {int(result.duration * result.fps)}")
    print()
    
    for stem in result.all():
        if stem:
            print(f"   {stem.name}: envelope shape {stem.fps_envelope.shape}")
