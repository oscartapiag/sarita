"""
Audio stem separation using Demucs.

Separates audio into 6 stems: drums, bass, vocals, guitar, piano, and other.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf
import numpy as np
import librosa




# Cache file to track separated stems
CACHE_FILE = Path("temp/.stems_cache.json")

# Stem names that Demucs 6-stem model outputs
STEM_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]


@dataclass
class StemPaths:
    """Container for paths to separated stems."""
    drums: Path
    bass: Path
    vocals: Path
    guitar: Path
    piano: Path
    other: Path
    
    def all(self) -> list[Path]:
        """Return all stem paths as a list."""
        return [self.drums, self.bass, self.vocals, self.guitar, self.piano, self.other]
    
    def as_dict(self) -> dict[str, Path]:
        """Return stems as a dictionary."""
        return {
            "drums": self.drums,
            "bass": self.bass,
            "vocals": self.vocals,
            "guitar": self.guitar,
            "piano": self.piano,
            "other": self.other,
        }


def get_file_hash(file_path: Path) -> str:
    """Get a hash of a file for cache identification."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_cache() -> dict:
    """Load the stems cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """Save the stems cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cached_stems(file_hash: str) -> Optional[StemPaths]:
    """
    Check if stems for this file have already been separated.
    
    Returns StemPaths if cached and all files exist, None otherwise.
    """
    cache = load_cache()
    if file_hash in cache:
        cached = cache[file_hash]
        stems = StemPaths(
            drums=Path(cached["drums"]),
            bass=Path(cached["bass"]),
            vocals=Path(cached["vocals"]),
            guitar=Path(cached["guitar"]),
            piano=Path(cached["piano"]),
            other=Path(cached["other"]),
        )
        # Verify all files still exist
        if all(p.exists() for p in stems.all()):
            return stems
        # Files were deleted, remove from cache
        del cache[file_hash]
        save_cache(cache)
    return None


def add_to_cache(file_hash: str, stems: StemPaths) -> None:
    """Add separated stems to the cache."""
    cache = load_cache()
    cache[file_hash] = {
        "drums": str(stems.drums),
        "bass": str(stems.bass),
        "vocals": str(stems.vocals),
        "guitar": str(stems.guitar),
        "piano": str(stems.piano),
        "other": str(stems.other),
    }
    save_cache(cache)


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    use_cuda: bool = True,
    model_name: str = "htdemucs_6s",
) -> StemPaths:
    """
    Separate an audio file into stems using Demucs.
    
    Args:
        audio_path: Path to the input audio file (WAV, MP3, etc.)
        output_dir: Directory to save the separated stems
        use_cuda: Whether to use GPU acceleration
        model_name: Demucs model to use (htdemucs_6s for 6 stems, htdemucs for 4)
        
    Returns:
        StemPaths object containing paths to the separated stems
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        RuntimeError: If separation fails
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check cache first
    file_hash = get_file_hash(audio_path)
    cached_stems = get_cached_stems(file_hash)
    if cached_stems:
        print(f"   └── Found in cache!")
        return cached_stems
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    if use_cuda and torch.cuda.is_available():
        device = "cuda"
        print(f"   └── Using GPU: {torch.cuda.get_device_name(0)}")
    elif use_cuda and torch.backends.mps.is_available():
        device = "mps"
        print(f"   └── Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print(f"   └── Using CPU (this will be slower)")
    
    # Import demucs here to avoid slow import on startup
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    print(f"   └── Loading model: {model_name}...")
    model = get_model(model_name)
    model.to(device)
    model.eval()
    
    # Load audio
    print(f"   └── Loading audio...")
    
    # Load with librosa (returns mono by default, so we set mono=False)
    audio_np, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    
    # Handle mono audio
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=0)
    elif audio_np.shape[0] > 2:
        audio_np = audio_np[:2]
    
    waveform = torch.from_numpy(audio_np).float()
    
    # Demucs expects stereo audio
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    
    # Resample if needed (Demucs expects 44100 Hz)
    if sample_rate != model.samplerate:
        print(f"   └── Resampling from {sample_rate}Hz to {model.samplerate}Hz...")
        waveform_np = waveform.numpy()
        waveform_np = librosa.resample(waveform_np, orig_sr=sample_rate, target_sr=model.samplerate)
        waveform = torch.from_numpy(waveform_np).float()
    
    # Add batch dimension and move to device
    waveform = waveform.unsqueeze(0).to(device)
    
    # Separate
    print(f"   └── Separating stems (this may take a few minutes)...")
    with torch.no_grad():
        sources = apply_model(model, waveform, device=device, progress=True)
    
    # Sources shape: (batch, num_sources, channels, samples)
    sources = sources.squeeze(0).cpu()
    
    # Save stems
    print(f"   └── Saving stems...")
    stem_base = audio_path.stem
    stem_paths = {}
    
    for i, stem_name in enumerate(model.sources):
        stem_path = output_dir / f"{stem_base}_{stem_name}.wav"
        # Save using soundfile (channels, samples) -> need to transpose to (samples, channels)
        sf.write(stem_path, sources[i].numpy().T, model.samplerate)
        stem_paths[stem_name] = stem_path
    
    stems = StemPaths(
        drums=stem_paths.get("drums", output_dir / f"{stem_base}_drums.wav"),
        bass=stem_paths.get("bass", output_dir / f"{stem_base}_bass.wav"),
        vocals=stem_paths.get("vocals", output_dir / f"{stem_base}_vocals.wav"),
        guitar=stem_paths.get("guitar", output_dir / f"{stem_base}_guitar.wav"),
        piano=stem_paths.get("piano", output_dir / f"{stem_base}_piano.wav"),
        other=stem_paths.get("other", output_dir / f"{stem_base}_other.wav"),
    )
    
    # Add to cache
    add_to_cache(file_hash, stems)
    
    print(f"   └── Done! Stems saved to: {output_dir}")
    return stems


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.separator <audio_file>")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    output_dir = Path("temp/stems")
    
    print(f"🎵 Separating: {audio_path.name}")
    stems = separate_stems(audio_path, output_dir)
    
    print()
    print("📁 Stems created:")
    for name, path in stems.as_dict().items():
        print(f"   {name}: {path}")
