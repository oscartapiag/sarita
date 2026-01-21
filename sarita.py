#!/usr/bin/env python3
"""
Sarita - AM-Style Music Visualizer

Generates oscilloscope-style videos from audio files or YouTube URLs by
separating stems and rendering animated waveforms for each instrument.
"""

import re
import click
from pathlib import Path

from backend.downloader import get_video_title, download_audio, sanitize_filename
from backend.separator import separate_stems
from backend.analyzer import analyze_stems, detect_key
from backend.renderer import render_oscilloscope
from backend.muxer import mux_audio_video


# Project directories
TEMP_DIR = Path("temp")
AUDIO_DIR = Path("temp/audio")
STEMS_DIR = Path("temp/stems")
OUTPUT_DIR = Path("output")


def is_youtube_url(url: str) -> bool:
    """Check if the input string is a YouTube URL."""
    youtube_patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(https?://)?(www\.)?youtu\.be/[\w-]+',
        r'(https?://)?(www\.)?youtube\.com/shorts/[\w-]+',
        r'(https?://)?music\.youtube\.com/watch\?v=[\w-]+',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


@click.command()
@click.argument("input_source", type=str)
@click.option(
    "-o", "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output video file path (defaults to input name + .mp4)"
)
@click.option(
    "--no-cuda",
    is_flag=True,
    default=False,
    help="Disable GPU acceleration for stem separation"
)
@click.option(
    "--keep-stems",
    is_flag=True,
    default=False,
    help="Keep separated stem audio files after rendering"
)
@click.option(
    "--quality",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="Render quality (affects resolution and framerate)"
)
@click.option(
    "--show-labels",
    is_flag=True,
    default=False,
    help="Show instrument labels on the visualization"
)
@click.option(
    "--preview",
    type=float,
    default=None,
    help="Only render first N seconds (for quick testing, e.g. --preview 30)"
)
@click.option(
    "--classic",
    is_flag=True,
    default=False,
    help="Use classic vertical waveform mode instead of Lissajous curves"
)
def main(input_source: str, output: Path, no_cuda: bool, keep_stems: bool, quality: str, show_labels: bool, preview: float, classic: bool):
    """
    Generate an oscilloscope music video from INPUT_SOURCE.
    
    INPUT_SOURCE can be a local audio file (mp3, wav, flac) or a YouTube URL.
    
    Examples:
    
        python sarita.py song.mp3 -o video.mp4
        
        python sarita.py "https://youtube.com/watch?v=..." -o video.mp4
    """
    click.echo(f"🎵 sarita - Music Visualizer")
    click.echo(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Determine input type and derive output name if not specified
    if is_youtube_url(input_source):
        click.echo("🔗 Detected YouTube URL")
        
        # Get video title for output filename
        click.echo("📝 Fetching video title...")
        try:
            video_title = get_video_title(input_source)
            click.echo(f"   └── {video_title}")
        except Exception as e:
            click.echo(f"❌ Error fetching video info: {e}")
            raise SystemExit(1)
        
        if output is None:
            output = OUTPUT_DIR / f"{sanitize_filename(video_title)}.mp4"
        
        click.echo(f"Input:   {input_source}")
        click.echo(f"Output:  {output}")
        click.echo(f"GPU:     {'Disabled' if no_cuda else 'Enabled'}")
        click.echo(f"Quality: {quality}")
        click.echo()
        
        click.echo("⏳ Step 0/5: Downloading audio from YouTube...")
        try:
            audio_path = download_audio(input_source, AUDIO_DIR, show_progress=True)
            click.echo(f"   └── Saved to: {audio_path}")
        except Exception as e:
            click.echo(f"❌ Error downloading audio: {e}")
            raise SystemExit(1)
    else:
        audio_path = Path(input_source)
        if not audio_path.exists():
            click.echo(f"❌ Error: File not found: {audio_path}")
            raise SystemExit(1)
        
        # Derive output name from input file if not specified
        if output is None:
            output = OUTPUT_DIR / f"{audio_path.stem}.mp4"
        
        click.echo(f"Input:   {input_source}")
        click.echo(f"Output:  {output}")
        click.echo(f"GPU:     {'Disabled' if no_cuda else 'Enabled'}")
        click.echo(f"Quality: {quality}")
        click.echo()
        
        click.echo(f"📁 Using local file: {audio_path.name}")
    
    click.echo()
    
    # Step 1: Separate stems
    click.echo("⏳ Step 1/4: Separating stems with Demucs...")
    try:
        stems = separate_stems(audio_path, STEMS_DIR, use_cuda=not no_cuda)
        click.echo(f"   └── Created: drums, bass, vocals, guitar, piano, other")
    except Exception as e:
        click.echo(f"❌ Error separating stems: {e}")
        raise SystemExit(1)
    
    # Step 2: Analyze audio
    # Match analysis FPS to render FPS for proper sync (Option A)
    fps_map = {"low": 24, "medium": 30, "high": 60}
    render_fps = fps_map[quality]
    
    click.echo("⏳ Step 2/4: Analyzing audio with Librosa...")
    try:
        analysis = analyze_stems(stems.as_dict(), target_fps=render_fps)
        click.echo(f"   └── Duration: {analysis.duration:.1f}s, {int(analysis.duration * analysis.fps)} frames")
        
        # Detect musical key from original audio
        click.echo("   └── Detecting musical key...")
        try:
            key_info = detect_key(audio_path)
            analysis.key_info = key_info
            click.echo(f"   └── 🎵 Key: {key_info.key} {key_info.mode} (confidence: {key_info.confidence:.0%})")
        except Exception as key_err:
            click.echo(f"   └── ⚠️ Could not detect key: {key_err}")
            # Continue without key detection
    except Exception as e:
        click.echo(f"❌ Error analyzing audio: {e}")
        raise SystemExit(1)
    
    # Step 3: Render visualization with Manim
    click.echo("⏳ Step 3/4: Rendering oscilloscope with Manim...")
    if preview:
        click.echo(f"   └── Preview mode: rendering first {preview}s only")
    try:
        temp_video_path = TEMP_DIR / f"{output.stem}_video.mp4"
        rendered_video = render_oscilloscope(
            analysis,
            temp_video_path,
            quality=quality,
            show_labels=show_labels,
            preview_duration=preview,
            use_lissajous=not classic,
        )
        click.echo(f"   └── Rendered to: {rendered_video}")
    except Exception as e:
        click.echo(f"❌ Error rendering video: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    
    # Step 4: Mux audio and video
    click.echo("⏳ Step 4/4: Muxing audio with video...")
    try:
        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)
        final_video = mux_audio_video(rendered_video, audio_path, output)
        click.echo(f"   └── Output: {final_video}")
    except Exception as e:
        click.echo(f"❌ Error muxing audio/video: {e}")
        raise SystemExit(1)
    
    # Cleanup
    if not keep_stems:
        click.echo("🧹 Cleaning up temporary files...")
        # TODO: Remove stem files and temp downloads
        pass
    
    click.echo()
    click.echo(f"✅ Done! Video saved to: {output}")


if __name__ == "__main__":
    main()
