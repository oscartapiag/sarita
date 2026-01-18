"""
Audio analysis using Librosa.

Extracts amplitude envelopes and waveform data from audio stems for visualization.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from scipy.signal import butter, filtfilt


# Per-stem bandpass filter settings (Hz)
# Tuned to each instrument's characteristic frequency range
STEM_FILTER_SETTINGS = {
    "drums":  {"low": 50,  "high": 10000},  # Kick thump through cymbal shimmer
    "bass":   {"low": 40,  "high": 5000},   # Low E fundamental through transients
    "vocals": {"low": 80,  "high": 6000},   # Remove rumble, cut above sibilance
    "guitar": {"low": 80,  "high": 6000},   # Body through presence
    "piano":  {"low": 27,  "high": 5000},   # Lowest A0 note through harmonics
    "other":  {"low": 40,  "high": 8000},   # Conservative range for misc content
}

# Key names for chroma-to-key mapping
KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


@dataclass
class KeyInfo:
    """Detected musical key information."""
    key: str           # Key name (e.g., "C", "F#", "Bb")
    mode: str          # "major" or "minor"
    confidence: float  # 0.0 to 1.0


def bandpass_filter(
    waveform: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the waveform.
    
    Args:
        waveform: Audio signal to filter
        sample_rate: Sample rate of the audio
        low_hz: High-pass cutoff (Hz)
        high_hz: Low-pass cutoff (Hz)
        order: Filter order (higher = sharper rolloff)
        
    Returns:
        Filtered waveform
    """
    nyquist = sample_rate / 2
    
    # Clamp frequencies to valid range
    low = max(low_hz / nyquist, 0.001)  # Avoid 0
    high = min(high_hz / nyquist, 0.999)  # Must be < 1
    
    if low >= high:
        return waveform  # Invalid range, return unfiltered
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, waveform)


def detect_key(audio_path: Path) -> KeyInfo:
    """
    Detect the musical key of an audio file using chroma analysis.
    
    Uses Krumhansl-Schmuckler key-finding algorithm with chroma features.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        KeyInfo with detected key, mode, and confidence
    """
    # Load audio (use lower sample rate for faster processing)
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)  # Analyze first 60s
    
    # Compute chroma features (CQT-based for better harmonic resolution)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Average chroma across time to get overall pitch class distribution
    chroma_avg = np.mean(chroma, axis=1)
    
    # Krumhansl-Schmuckler key profiles (correlation templates)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)
    chroma_norm = chroma_avg / np.linalg.norm(chroma_avg) if np.linalg.norm(chroma_avg) > 0 else chroma_avg
    
    # Correlate with all 24 keys (12 major + 12 minor)
    correlations = []
    for shift in range(12):
        # Rotate chroma to test each key
        rotated = np.roll(chroma_norm, -shift)
        major_corr = np.corrcoef(rotated, major_profile)[0, 1]
        minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
        correlations.append((major_corr, KEY_NAMES[shift], "major"))
        correlations.append((minor_corr, KEY_NAMES[shift], "minor"))
    
    # Find best match
    best = max(correlations, key=lambda x: x[0])
    confidence = (best[0] + 1) / 2  # Map correlation [-1, 1] to confidence [0, 1]
    
    return KeyInfo(
        key=best[1],
        mode=best[2],
        confidence=float(confidence)
    )


@dataclass
class StemEnvelope:
    """Amplitude envelope data for a single stem."""
    name: str
    envelope: np.ndarray      # Amplitude envelope (downsampled for visualization)
    waveform: np.ndarray      # Full waveform data (mono, normalized)
    sample_rate: int          # Original sample rate
    duration: float           # Duration in seconds
    fps_envelope: np.ndarray  # Envelope resampled to target FPS
    noise_threshold: float    # Dynamic visibility threshold (noise floor * 2)
    
    # Per-frame harmonic features for reactive colors
    chroma: np.ndarray              # Shape: (12, num_frames) - pitch class energy per frame
    spectral_brightness: np.ndarray # Shape: (num_frames,) - normalized spectral centroid
    onsets: np.ndarray              # Shape: (num_frames,) - 1.0 on transients, 0 otherwise
    

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
    analysis_fps: int  # FPS at which envelopes were calculated (for sync)
    key_info: Optional[KeyInfo] = None  # Detected musical key
    
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
    
    Uses RMS (Root Mean Square) for smooth, AM-style envelope that represents
    perceived loudness rather than peak amplitude.
    
    Args:
        audio_path: Path to the stem audio file
        stem_name: Name of the stem (for labeling)
        target_fps: Target frames per second for the envelope
        hop_length: Hop length for RMS calculation
        
    Returns:
        StemEnvelope containing the analysis results
    """
    # Load audio (mono for analysis)
    waveform, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(waveform) / sr
    
    # Apply bandpass filter for visualization (removes noise, smooths waveform)
    filter_settings = STEM_FILTER_SETTINGS.get(stem_name, {"low": 40, "high": 8000})
    filtered_waveform = bandpass_filter(
        waveform, sr, 
        filter_settings["low"], 
        filter_settings["high"]
    )
    
    # Normalize filtered waveform to [-1, 1] for consistent visualization
    max_val = np.max(np.abs(filtered_waveform))
    if max_val > 0:
        filtered_waveform = filtered_waveform / max_val
    
    # Calculate exact number of frames for video
    num_frames = int(duration * target_fps)
    
    # Use RMS for smooth envelope (measures energy, not peaks)
    # This prevents flicker from transients like plosives
    samples_per_frame = len(waveform) // num_frames
    
    amplitudes = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * samples_per_frame
        end = min(start + samples_per_frame, len(waveform))
        if end > start:
            chunk = waveform[start:end]
            # RMS = sqrt(mean(square(waveform)))
            amplitudes[i] = np.sqrt(np.mean(chunk ** 2))
    
    # Rolling average smoothing (5-frame window) for weightier animation
    kernel = np.ones(5) / 5
    amplitudes = np.convolve(amplitudes, kernel, mode='same')
    
    # Normalize to [0, 1]
    amp_max = amplitudes.max()
    if amp_max > 0:
        amplitudes = amplitudes / amp_max
    
    # Calculate initial noise threshold (15th percentile - aggressive)
    # This will be refined by cross-stem comparison later
    noise_floor = np.percentile(amplitudes, 15)
    noise_threshold = noise_floor
    
    # === HARMONIC FEATURES FOR REACTIVE COLORS ===
    
    # 1. Chroma (pitch class distribution per frame)
    # Use chroma_stft (faster than chroma_cqt, still good for visualization)
    chroma_hop = len(waveform) // num_frames
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, hop_length=chroma_hop)
    # Resample chroma to exactly match video frame count
    if chroma.shape[1] != num_frames:
        # Linear interpolation to target frame count
        x_old = np.linspace(0, 1, chroma.shape[1])
        x_new = np.linspace(0, 1, num_frames)
        chroma_resampled = np.zeros((12, num_frames))
        for i in range(12):
            chroma_resampled[i] = np.interp(x_new, x_old, chroma[i])
        chroma = chroma_resampled
    
    # 2. Spectral brightness (centroid normalized to 0-1)
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=chroma_hop)[0]
    # Resample to video FPS
    x_old = np.linspace(0, 1, len(centroid))
    x_new = np.linspace(0, 1, num_frames)
    spectral_brightness = np.interp(x_new, x_old, centroid)
    # Normalize to 0-1 (relative to max for this stem)
    if spectral_brightness.max() > 0:
        spectral_brightness = spectral_brightness / spectral_brightness.max()
    
    # 3. Onset detection (marks transients)
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sr, hop_length=chroma_hop)
    onsets = np.zeros(num_frames)
    # Map onset frames to video frames
    onset_to_video_ratio = num_frames / (len(waveform) / chroma_hop)
    for onset_frame in onset_frames:
        video_frame = int(onset_frame * onset_to_video_ratio)
        if 0 <= video_frame < num_frames:
            onsets[video_frame] = 1.0
            # Add short decay for visual smoothness
            for decay_frame in range(1, 4):
                if video_frame + decay_frame < num_frames:
                    onsets[video_frame + decay_frame] = max(
                        onsets[video_frame + decay_frame], 
                        1.0 - decay_frame * 0.3
                    )
    
    return StemEnvelope(
        name=stem_name,
        envelope=amplitudes,
        waveform=filtered_waveform,  # Store filtered waveform for visualization
        sample_rate=sr,
        duration=duration,
        fps_envelope=amplitudes,  # Same as envelope, 1:1 with video frames
        noise_threshold=noise_threshold,
        chroma=chroma,
        spectral_brightness=spectral_brightness,
        onsets=onsets,
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
    
    # Cross-stem comparison: compare each stem to the loudest one
    # Stems that are much quieter than the loudest are likely empty/noise
    all_maxes = {name: env.envelope.max() for name, env in envelopes.items()}
    global_max = max(all_maxes.values()) if all_maxes else 1.0
    
    for stem_name, env in envelopes.items():
        stem_max = all_maxes[stem_name]
        
        # If this stem's max is < 20% of the loudest stem, mark as empty
        if stem_max < global_max * 0.2:
            # Set threshold to 1.0 - will never show (all amplitudes are 0-1)
            env.noise_threshold = 1.0
        else:
            # Use 15th percentile + cross-stem scaling
            # Threshold is relative to both local noise floor and global context
            local_threshold = env.noise_threshold  # Already 15th percentile
            # Scale threshold: quieter stems get higher threshold
            relative_loudness = stem_max / global_max  # 0.2 to 1.0
            scaled_threshold = local_threshold / relative_loudness
            # Minimum threshold floor of 0.1
            env.noise_threshold = max(scaled_threshold, 0.1)
    
    return AnalysisResult(
        drums=envelopes.get("drums"),
        bass=envelopes.get("bass"),
        vocals=envelopes.get("vocals"),
        guitar=envelopes.get("guitar"),
        piano=envelopes.get("piano"),
        other=envelopes.get("other"),
        duration=duration,
        fps=target_fps,
        analysis_fps=target_fps,  # Track for sync in renderer
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
