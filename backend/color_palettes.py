"""
Color palette system for key-based visualization.

Maps musical keys to curated color palettes for emotionally-coherent visuals.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class KeyInfo:
    """Detected musical key information."""
    key: str           # Key name (e.g., "C", "F#")
    mode: str          # "major" or "minor"
    confidence: float  # 0.0 to 1.0


# Default stem colors (fallback)
DEFAULT_COLORS = {
    "drums": "#FF6B35",    # Orange
    "bass": "#C41E3A",     # Deep Red  
    "vocals": "#FFD700",   # Gold
    "guitar": "#1E90FF",   # Electric Blue
    "piano": "#9B59B6",    # Purple
    "other": "#20B2AA",    # Teal
}

# Key-based color palettes
# Each key has major (brighter) and minor (darker/moodier) variants
# Colors assigned based on synesthetic associations and color theory

KEY_PALETTES = {
    # White keys - warm to neutral
    "C": {
        "major": {  # Pure, bright, clean
            "drums": "#FFE066",   # Bright yellow
            "bass": "#FF8C42",    # Warm orange
            "vocals": "#FFFFFF",  # Pure white
            "guitar": "#7EC8E3",  # Sky blue
            "piano": "#98D8AA",   # Mint green
            "other": "#DDA0DD",   # Plum
        },
        "minor": {  # Melancholic, introspective
            "drums": "#4A6FA5",   # Steel blue
            "bass": "#2E4057",    # Dark slate
            "vocals": "#C9D6DF",  # Silver
            "guitar": "#6B7AA1",  # Dusty violet
            "piano": "#5C7AEA",   # Periwinkle
            "other": "#8E94F2",   # Soft indigo
        },
    },
    "D": {
        "major": {  # Triumphant, victorious
            "drums": "#FFD700",   # Gold
            "bass": "#FF6B35",    # Tangerine
            "vocals": "#FFF8DC",  # Cornsilk
            "guitar": "#FF9F1C",  # Amber
            "piano": "#F4D03F",   # Sunflower
            "other": "#FFBF69",   # Peach
        },
        "minor": {  # Serious, stoic
            "drums": "#8B4513",   # Saddle brown
            "bass": "#5D4037",    # Coffee
            "vocals": "#A0826D",  # Dusty rose
            "guitar": "#6D4C41",  # Cocoa
            "piano": "#8D6E63",   # Taupe
            "other": "#795548",   # Bronze
        },
    },
    "E": {
        "major": {  # Bright, joyful, guitaristic
            "drums": "#FF5722",   # Deep orange
            "bass": "#E53935",    # Red
            "vocals": "#FFEB3B",  # Yellow
            "guitar": "#FF7043",  # Coral
            "piano": "#FFA726",   # Orange
            "other": "#FFCC80",   # Light orange
        },
        "minor": {  # Blues, gritty
            "drums": "#B71C1C",   # Dark red
            "bass": "#4A0000",    # Maroon
            "vocals": "#FF5252",  # Salmon red
            "guitar": "#D32F2F",  # Crimson
            "piano": "#C62828",   # Fire red
            "other": "#EF5350",   # Light crimson
        },
    },
    "F": {
        "major": {  # Pastoral, natural
            "drums": "#4CAF50",   # Green
            "bass": "#2E7D32",    # Forest green
            "vocals": "#C8E6C9",  # Pale green
            "guitar": "#66BB6A",  # Light green
            "piano": "#81C784",   # Sage
            "other": "#A5D6A7",   # Mint
        },
        "minor": {  # Mysterious, deep
            "drums": "#1B5E20",   # Dark green
            "bass": "#004D40",    # Teal dark
            "vocals": "#26A69A",  # Teal
            "guitar": "#00695C",  # Dark cyan
            "piano": "#00897B",   # Eucalyptus
            "other": "#4DB6AC",   # Aqua
        },
    },
    "G": {
        "major": {  # Rustic, folk, sunny
            "drums": "#FFC107",   # Amber
            "bass": "#FF9800",    # Orange
            "vocals": "#FFECB3",  # Cream
            "guitar": "#FFB300",  # Gold amber
            "piano": "#FFD54F",   # Light gold
            "other": "#FFE082",   # Pale gold
        },
        "minor": {  # Contemplative
            "drums": "#5D4037",   # Brown
            "bass": "#3E2723",    # Dark brown
            "vocals": "#A1887F",  # Taupe
            "guitar": "#6D4C41",  # Coffee
            "piano": "#795548",   # Mocha
            "other": "#8D6E63",   # Dusty brown
        },
    },
    "A": {
        "major": {  # Bright, declaration
            "drums": "#E91E63",   # Pink
            "bass": "#C2185B",    # Dark pink
            "vocals": "#F8BBD9",  # Light pink
            "guitar": "#EC407A",  # Rose
            "piano": "#F06292",   # Flamingo
            "other": "#F48FB1",   # Soft pink
        },
        "minor": {  # The classic (Do I Wanna Know?)
            "drums": "#263238",   # Charcoal
            "bass": "#37474F",    # Dark gray-blue
            "vocals": "#78909C",  # Steel
            "guitar": "#455A64",  # Slate
            "piano": "#546E7A",   # Blue-gray
            "other": "#607D8B",   # Cool gray
        },
    },
    "B": {
        "major": {  # Intense, electric  
            "drums": "#9C27B0",   # Purple
            "bass": "#7B1FA2",    # Deep purple
            "vocals": "#E1BEE7",  # Lavender
            "guitar": "#AB47BC",  # Orchid
            "piano": "#BA68C8",   # Light purple
            "other": "#CE93D8",   # Soft violet
        },
        "minor": {  # Dark, dramatic
            "drums": "#4A148C",   # Deep violet
            "bass": "#311B92",    # Indigo
            "vocals": "#7E57C2",  # Medium purple
            "guitar": "#5E35B1",  # Deep lavender
            "piano": "#673AB7",   # Violet
            "other": "#9575CD",   # Soft purple
        },
    },
    # Black keys - sharps/flats
    "C#": {
        "major": {
            "drums": "#00BCD4",   # Cyan
            "bass": "#0097A7",    # Dark cyan
            "vocals": "#B2EBF2",  # Light cyan
            "guitar": "#00ACC1",  # Teal cyan
            "piano": "#26C6DA",   # Turquoise
            "other": "#4DD0E1",   # Light turquoise
        },
        "minor": {
            "drums": "#006064",   # Dark teal
            "bass": "#004D40",    # Deep teal
            "vocals": "#4DB6AC",  # Sea green
            "guitar": "#00796B",  # Jungle green
            "piano": "#00897B",   # Persian green
            "other": "#26A69A",   # Aqua marine
        },
    },
    "D#": {
        "major": {
            "drums": "#3F51B5",   # Indigo
            "bass": "#303F9F",    # Dark indigo
            "vocals": "#C5CAE9",  # Lavender gray
            "guitar": "#5C6BC0",  # Slate blue
            "piano": "#7986CB",   # Light slate
            "other": "#9FA8DA",   # Periwinkle
        },
        "minor": {
            "drums": "#1A237E",   # Navy
            "bass": "#0D47A1",    # Royal blue dark
            "vocals": "#5472D3",  # Cornflower
            "guitar": "#283593",  # Deep blue
            "piano": "#3949AB",   # Dark cornflower
            "other": "#3F51B5",   # Indigo
        },
    },
    "F#": {
        "major": {
            "drums": "#FF5722",   # Deep orange
            "bass": "#E64A19",    # Burnt orange
            "vocals": "#FFCCBC",  # Peach
            "guitar": "#FF7043",  # Coral
            "piano": "#FF8A65",   # Light coral
            "other": "#FFAB91",   # Salmon
        },
        "minor": {  # Intense, brooding
            "drums": "#BF360C",   # Rust
            "bass": "#8D6E63",    # Warm gray
            "vocals": "#FF8A65",  # Coral pink
            "guitar": "#D84315",  # Dark orange
            "piano": "#E64A19",   # Burnt sienna
            "other": "#FF5722",   # Vermillion
        },
    },
    "G#": {
        "major": {
            "drums": "#8BC34A",   # Light green
            "bass": "#689F38",    # Olive
            "vocals": "#DCEDC8",  # Tea green
            "guitar": "#7CB342",  # Lawn green
            "piano": "#9CCC65",   # Yellow green
            "other": "#AED581",   # Pistachio
        },
        "minor": {
            "drums": "#33691E",   # Dark olive
            "bass": "#1B5E20",    # Forest
            "vocals": "#7CB342",  # Leaf
            "guitar": "#558B2F",  # Moss
            "piano": "#689F38",   # Fern
            "other": "#8BC34A",   # Grass
        },
    },
    "A#": {
        "major": {
            "drums": "#2196F3",   # Blue
            "bass": "#1976D2",    # Dark blue
            "vocals": "#BBDEFB",  # Light blue
            "guitar": "#42A5F5",  # Sky blue
            "piano": "#64B5F6",   # Baby blue
            "other": "#90CAF9",   # Powder blue
        },
        "minor": {
            "drums": "#0D47A1",   # Deep blue
            "bass": "#1565C0",    # True blue
            "vocals": "#64B5F6",  # Carolina blue
            "guitar": "#1976D2",  # Denim
            "piano": "#1E88E5",   # Azure
            "other": "#2196F3",   # Dodger blue
        },
    },
}

# Enharmonic equivalents (flats map to their sharp equivalents)
ENHARMONIC_MAP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


def get_palette_for_key(key_info: Optional[KeyInfo]) -> dict[str, str]:
    """
    Get stem colors based on detected musical key.
    
    Args:
        key_info: Detected key information, or None for defaults
        
    Returns:
        Dictionary mapping stem names to hex color codes
    """
    if key_info is None:
        return DEFAULT_COLORS.copy()
    
    # Handle enharmonic equivalents
    key = ENHARMONIC_MAP.get(key_info.key, key_info.key)
    
    # Look up palette
    if key in KEY_PALETTES:
        mode_palettes = KEY_PALETTES[key]
        if key_info.mode in mode_palettes:
            return mode_palettes[key_info.mode].copy()
    
    # Fallback to defaults
    return DEFAULT_COLORS.copy()


def format_key_name(key_info: KeyInfo) -> str:
    """Format key info for display (e.g., 'A minor', 'C major')."""
    return f"{key_info.key} {key_info.mode}"
