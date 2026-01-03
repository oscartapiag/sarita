"""
YouTube audio downloader using yt-dlp.

Downloads audio from YouTube URLs and converts to WAV format for processing.
"""

import json
import re
from pathlib import Path
from typing import Optional

import yt_dlp
from tqdm import tqdm


# Cache file to track downloaded videos
CACHE_FILE = Path("temp/.download_cache.json")


def sanitize_filename(name: str) -> str:
    """Remove or replace characters that are problematic in filenames."""
    # Remove/replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    # Trim and limit length
    return sanitized.strip()[:200]


def load_cache() -> dict:
    """Load the download cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    """Save the download cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cached_path(video_id: str) -> Optional[Path]:
    """
    Check if a video has already been downloaded.
    
    Returns the path to the cached file if it exists, None otherwise.
    """
    cache = load_cache()
    if video_id in cache:
        cached_path = Path(cache[video_id])
        if cached_path.exists():
            return cached_path
        # File was deleted, remove from cache
        del cache[video_id]
        save_cache(cache)
    return None


def add_to_cache(video_id: str, file_path: Path) -> None:
    """Add a downloaded video to the cache."""
    cache = load_cache()
    cache[video_id] = str(file_path)
    save_cache(cache)


class DownloadProgressBar:
    """Progress bar wrapper for yt-dlp downloads."""
    
    def __init__(self):
        self.pbar: Optional[tqdm] = None
        self.downloaded_bytes = 0
    
    def hook(self, d: dict):
        """yt-dlp progress hook callback."""
        if d['status'] == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            
            if self.pbar is None and total > 0:
                self.pbar = tqdm(
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc='   Downloading',
                    leave=False
                )
            
            if self.pbar is not None:
                # Update by delta since last call
                delta = downloaded - self.downloaded_bytes
                if delta > 0:
                    self.pbar.update(delta)
                self.downloaded_bytes = downloaded
                
        elif d['status'] == 'finished':
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            self.downloaded_bytes = 0


def get_video_info(url: str) -> dict:
    """
    Extract video metadata from a YouTube URL without downloading.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary containing video info (id, title, duration, etc.)
        
    Raises:
        yt_dlp.DownloadError: If the video cannot be accessed
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            'id': info.get('id', ''),
            'title': info.get('title', 'Unknown'),
            'duration': info.get('duration', 0),
            'uploader': info.get('uploader', 'Unknown'),
            'view_count': info.get('view_count', 0),
        }


def get_video_title(url: str) -> str:
    """
    Get just the video title from a YouTube URL.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video title as a string
    """
    info = get_video_info(url)
    return info['title']


def download_audio(url: str, output_dir: Path, show_progress: bool = True) -> Path:
    """
    Download audio from a YouTube URL and convert to WAV.
    
    Checks the cache first to avoid re-downloading.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the downloaded audio
        show_progress: Whether to show a progress bar
        
    Returns:
        Path to the downloaded WAV file
        
    Raises:
        yt_dlp.DownloadError: If download fails
        FileNotFoundError: If FFmpeg is not installed
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info first (including ID for cache lookup)
    ydl_opts = {'quiet': True, 'no_warnings': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info.get('id', '')
        title = info.get('title', 'Unknown')
    
    # Check cache
    cached_path = get_cached_path(video_id)
    if cached_path:
        if show_progress:
            print(f"   └── Found in cache: {cached_path}")
        return cached_path
    
    # Setup progress hook
    progress = DownloadProgressBar() if show_progress else None
    
    # Download
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',  # Best quality
        }],
        'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'progress_hooks': [progress.hook] if progress else [],
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
        sanitized_title = sanitize_filename(title)
        wav_path = output_dir / f"{sanitized_title}.wav"
        
        # yt-dlp might use the original title (not sanitized) for the file
        # So we need to find the actual file
        original_path = output_dir / f"{title}.wav"
        if original_path.exists() and not wav_path.exists():
            original_path.rename(wav_path)
        
        # If still not found, try to find any wav file just created
        if not wav_path.exists():
            wav_files = list(output_dir.glob("*.wav"))
            if wav_files:
                wav_path = wav_files[-1]  # Get most recent
        
        # Add to cache
        add_to_cache(video_id, wav_path)
        
        return wav_path


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.downloader <youtube_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = Path("temp")
    
    print(f"🔗 Fetching video info...")
    info = get_video_info(url)
    print(f"   Title: {info['title']}")
    print(f"   Duration: {info['duration']}s")
    print()
    
    print(f"📥 Downloading audio...")
    wav_path = download_audio(url, output_dir)
    print(f"   └── Saved to: {wav_path}")
