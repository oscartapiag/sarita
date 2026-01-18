"""
Harmonic color engine for per-frame reactive colors.

Maps pitch class (chroma) to hue, spectral brightness to saturation,
and onsets to glow/pulse effects.
"""

import numpy as np
import colorsys
from typing import Tuple


# Pitch class to hue mapping (color wheel, 0-1)
# Based on synesthetic associations and visual appeal
PITCH_TO_HUE = {
    0:  0.00,   # C  → Red
    1:  0.05,   # C# → Red-Orange
    2:  0.08,   # D  → Orange
    3:  0.12,   # D# → Yellow-Orange
    4:  0.17,   # E  → Yellow
    5:  0.33,   # F  → Green
    6:  0.45,   # F# → Cyan
    7:  0.55,   # G  → Blue
    8:  0.65,   # G# → Blue-Indigo
    9:  0.75,   # A  → Violet
    10: 0.85,   # A# → Magenta
    11: 0.92,   # B  → Pink
}

# Stem-specific saturation multiplier (some stems are naturally less colorful)
STEM_SATURATION = {
    "drums": 0.7,    # Slightly muted - rhythm section
    "bass": 0.8,     # Warm but not overwhelming
    "vocals": 1.0,   # Full saturation - focus point
    "guitar": 0.9,   # Vivid
    "piano": 0.85,   # Slightly softer
    "other": 0.75,   # Background
}


def chroma_to_hue(chroma: np.ndarray) -> float:
    """
    Convert a 12-element chroma vector to a single hue value.
    
    Blends hues weighted by the energy of each pitch class.
    
    Args:
        chroma: Array of shape (12,) with energy per pitch class
        
    Returns:
        Hue value between 0 and 1
    """
    # Normalize chroma to sum to 1 (or handle silence)
    total = np.sum(chroma)
    if total < 1e-6:
        return 0.0  # Default to red on silence
    
    weights = chroma / total
    
    # Weighted circular mean (hue is circular)
    sin_sum = 0.0
    cos_sum = 0.0
    for pitch_class, weight in enumerate(weights):
        hue = PITCH_TO_HUE[pitch_class]
        angle = hue * 2 * np.pi
        sin_sum += weight * np.sin(angle)
        cos_sum += weight * np.cos(angle)
    
    # Convert back to hue
    mean_angle = np.arctan2(sin_sum, cos_sum)
    hue = (mean_angle / (2 * np.pi)) % 1.0
    
    return hue


def get_frame_color(
    stem_name: str,
    chroma: np.ndarray,
    spectral_brightness: float,
    onset: float,
    base_saturation: float = 0.8,
    base_lightness: float = 0.5,
) -> str:
    """
    Compute the color for one stem at one frame.
    
    Args:
        stem_name: Name of the stem (for saturation adjustment)
        chroma: 12-element array with pitch class energies for this frame
        spectral_brightness: 0-1 spectral centroid (higher = brighter sound)
        onset: 0-1 onset strength (1 = transient detected)
        base_saturation: Base saturation before modulation
        base_lightness: Base lightness before modulation
        
    Returns:
        Hex color string (e.g., "#FF6B35")
    """
    # 1. Hue from chroma (pitch)
    hue = chroma_to_hue(chroma)
    
    # 2. Saturation from spectral brightness + stem adjustment
    stem_sat_mult = STEM_SATURATION.get(stem_name, 0.8)
    # Brighter sounds are more saturated
    saturation = base_saturation * stem_sat_mult * (0.6 + 0.4 * spectral_brightness)
    saturation = min(1.0, max(0.2, saturation))  # Clamp
    
    # 3. Lightness with onset glow
    # Onsets cause a brief brightness pulse
    lightness = base_lightness + onset * 0.25
    lightness = min(0.85, max(0.3, lightness))  # Clamp
    
    # Convert HSL to RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    
    # Convert to hex
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"


def get_stem_frame_color(envelope, frame_idx: int) -> str:
    """
    Get the color for a stem at a specific frame.
    
    Convenience function that takes a StemEnvelope and frame index.
    
    Args:
        envelope: StemEnvelope with chroma, spectral_brightness, onsets
        frame_idx: Current frame index
        
    Returns:
        Hex color string
    """
    # Get features for this frame
    frame_idx = min(frame_idx, envelope.chroma.shape[1] - 1)
    frame_idx = min(frame_idx, len(envelope.spectral_brightness) - 1)
    frame_idx = min(frame_idx, len(envelope.onsets) - 1)
    
    chroma = envelope.chroma[:, frame_idx]
    brightness = envelope.spectral_brightness[frame_idx]
    onset = envelope.onsets[frame_idx]
    
    return get_frame_color(
        stem_name=envelope.name,
        chroma=chroma,
        spectral_brightness=brightness,
        onset=onset,
    )


def get_dominant_pitch_name(chroma: np.ndarray) -> str:
    """Get the name of the dominant pitch class (for debugging)."""
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitch_names[np.argmax(chroma)]
