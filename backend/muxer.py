"""
Audio/video muxer using FFmpeg.

Combines rendered video with original audio to create the final output.
"""

import subprocess
import shutil
from pathlib import Path


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    overwrite: bool = True,
) -> Path:
    """
    Combine video and audio into a single file.
    
    Args:
        video_path: Path to the video file (without audio)
        audio_path: Path to the audio file
        output_path: Path for the final output
        overwrite: Whether to overwrite existing output
        
    Returns:
        Path to the muxed output file
        
    Raises:
        FileNotFoundError: If input files don't exist or FFmpeg not found
        subprocess.CalledProcessError: If FFmpeg fails
    """
    if not check_ffmpeg():
        raise FileNotFoundError("FFmpeg not found. Please install FFmpeg.")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),      # Video input
        "-i", str(audio_path),      # Audio input
        "-c:v", "copy",             # Copy video stream (no re-encode)
        "-c:a", "aac",              # Encode audio as AAC
        "-b:a", "192k",             # Audio bitrate
        "-shortest",                # Match shortest stream duration
        "-map", "0:v:0",            # Use video from first input
        "-map", "1:a:0",            # Use audio from second input
        str(output_path),
    ]
    
    # Run FFmpeg
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    
    return output_path


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python -m backend.muxer <video> <audio> <output>")
        sys.exit(1)
    
    video = Path(sys.argv[1])
    audio = Path(sys.argv[2])
    output = Path(sys.argv[3])
    
    print(f"🎬 Muxing video and audio...")
    result = mux_audio_video(video, audio, output)
    print(f"   └── Output: {result}")
