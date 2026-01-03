# Sarita

**A music visualizer that generates oscilloscope-style videos from any audio file or YouTube URL.**

Inspired by the iconic "Do I Wanna Know?" music video by Arctic Monkeys, Sarita separates audio into stems (drums, bass, vocals, guitar, piano and other) and renders each as an animated waveform line using Manim.

---

## Features

- 🎵 **AI-Powered Stem Separation** — Uses [Demucs](https://github.com/facebookresearch/demucs) to isolate instruments
- 📊 **Per-Stem Oscilloscope Lines** — Each instrument gets its own animated waveform
- 🎬 **Manim-Powered Rendering** — Smooth, beautiful animations with ManimCE
- 🔗 **YouTube URL Support** — Paste a link, download audio automatically
- 🎧 **Local File Support** — Works with MP3, WAV, FLAC, and more

---

## Requirements

- **Python 3.10+**
- **FFmpeg** (must be installed and in PATH)
- **CUDA-capable GPU** (recommended for Demucs, 6GB+ VRAM)
- **LaTeX** (optional, for Manim text rendering)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sarita.git
   cd sarita
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify FFmpeg is installed:
   ```bash
   ffmpeg -version
   ```

---

## Usage

### From a local file:
```bash
python sarita.py song.mp3 -o output.mp4
```

### From a YouTube URL:
```bash
python sarita.py "https://www.youtube.com/watch?v=..." -o output.mp4
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output` | Output video path | Auto: `<input_name>.mp4` |
| `--no-cuda` | Disable GPU acceleration | Uses GPU if available |
| `--keep-stems` | Keep separated stem files | Deletes after render |
| `--quality` | Render quality: `low`, `medium`, `high` | `medium` |

---

## How It Works

1. **Download** *(if URL)* — yt-dlp fetches the audio from YouTube
2. **Separate** — Demucs AI splits the audio into 4 stems: drums, bass, vocals, other
3. **Analyze** — Librosa extracts amplitude envelopes from each stem
4. **Render** — Manim animates oscilloscope lines synced to the audio
5. **Mux** — FFmpeg combines the animation with the original audio

---

## Roadmap

- [x] Project setup
- [ ] YouTube audio download (yt-dlp)
- [ ] Stem separation (Demucs)
- [ ] Amplitude analysis (Librosa)
- [ ] Oscilloscope rendering (Manim)
- [ ] Audio/video muxing (FFmpeg)
- [ ] Web UI (FastAPI) — *future*

---

## ⚠️ Disclaimer

This tool includes the ability to download audio from YouTube via yt-dlp. 

- Downloading copyrighted content may violate YouTube's Terms of Service
- This feature is intended for personal, non-commercial use only
- Users are responsible for ensuring they have the right to download and use any content
- The developers do not condone copyright infringement

**Use responsibly and respect content creators' rights.**

---

## Credits

- **Demucs** by Meta AI Research
- **ManimCE** by the Manim Community
- **yt-dlp** for YouTube audio extraction
- **Librosa** for audio analysis
- Inspired by the Arctic Monkeys' "Do I Wanna Know?" visualizer


---

## License

MIT
