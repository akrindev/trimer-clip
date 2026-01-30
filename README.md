# Trimer-Clip Agent Skills

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-Compatible-green.svg)](https://agentskills.io)

> **AI-powered video editing for short-form content creation**

Trimer-Clip enables AI agents to automatically find viral moments in long videos (podcasts, vlogs, gaming, tutorials) and create short-form content for TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels - all in 9:16 portrait format.

Built following the [Agent Skills](https://agentskills.io) open standard.

## ✨ Features

- 🎬 **Automatic Video Clipping**: Detect highlights, funny moments, and viral-worthy segments
- 📱 **Multi-Platform Support**: Export for TikTok, YouTube Shorts, Instagram Reels (9:16 portrait)
- 🤖 **AI-Powered Detection**: Scene changes, laughter, sentiment/emotion, transcript analysis
- 🗣️ **Advanced Speaker Diarization**: pyannote-audio integration for 35% better accuracy
- 🧠 **Smart AI Decisions**: Automatically selects best tools based on content type
- 🎤 **Flexible Transcription**: Whisper (local) or Gemini API (gemini-flash-lite-latest)
- ⚡ **Fully Automated**: AI agents can process entire workflow end-to-end
- 📥 **YouTube Download**: Download videos directly from YouTube URLs

## 🚀 Quick Start

### One-Line Installation

```bash
pip install git+https://github.com/akrindev/trimer-clip.git
```

### Single Command Autocut

```bash
# From YouTube URL - fully automatic
python -m trimer_clip.autocut "https://youtube.com/watch?v=VIDEO_ID" --platform tiktok

# From local file
python -m trimer_clip.autocut video.mp4 --num-clips 5 --platform shorts
```

## 📦 Available Skills

| Skill | Description | Status |
|--------|-------------|--------|
| `youtube-downloader` | Download videos from YouTube URLs | ✅ Ready |
| `video-transcriber` | Transcribe audio using Whisper or Gemini API | ✅ Ready |
| `speaker-diarization` | Advanced speaker diarization with pyannote-audio | ✅ Ready |
| `scene-detector` | Detect scene changes and shot boundaries | ✅ Ready |
| `laughter-detector` | Find humorous/laughing segments | ✅ Ready |
| `sentiment-analyzer` | Analyze emotion and sentiment | ✅ Ready |
| `highlight-scanner` | Combine all signals to find viral moments | ✅ Ready |
| `video-trimmer` | Trim/cut videos by timestamp | ✅ Ready |
| `portrait-resizer` | Convert to 9:16 vertical format | ✅ Ready |
| `subtitle-overlay` | Add captions to video clips | ✅ Ready |
| `autocut-shorts` | Main orchestration skill for full workflow | ✅ Ready |
| `batch-processor` | Process multiple videos at once | ✅ Ready |

## 📋 Requirements

- Python 3.9+
- FFmpeg
- 4GB+ RAM (8GB+ recommended for pyannote)
- Optional: CUDA-compatible GPU for faster processing

## 🔧 Setup

### 1. Clone Repository

```bash
git clone https://github.com/akrindev/trimer-clip.git
cd trimer-clip
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

**Verify Installation:**
```bash
ffmpeg -version
```

### 4. Set API Keys (Optional)

Create a `.env` file in the project root:

```bash
# For Gemini API transcription (optional)
GEMINI_API_KEY=your-gemini-api-key

# For pyannote speaker diarization (optional)
HUGGINGFACE_TOKEN=your-huggingface-token
```

Or set environment variables:
```bash
export GEMINI_API_KEY="your-api-key"
export HUGGINGFACE_TOKEN="your-token"
```

**How to get tokens:**
- **Gemini API**: Get key at [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HuggingFace**: 
  1. Create account at [huggingface.co](https://huggingface.co)
  2. Generate token at [hf.co/settings/tokens](https://hf.co/settings/tokens)
  3. Accept terms at [pyannote/speaker-diarization-community-1](https://hf.co/pyannote/speaker-diarization-community-1)

## 🎯 Usage Examples

### Example 1: Podcast to TikTok

```bash
# Auto-detects it's a podcast and uses pyannote for speaker diarization
python skills/autocut-shorts/scripts/autocut.py podcast.mp4 \
  --num-clips 5 \
  --platform tiktok \
  --style tiktok
```

### Example 2: YouTube Video to Shorts

```bash
# Downloads and processes in one command
python skills/autocut-shorts/scripts/autocut.py \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --platform shorts \
  --num-clips 10
```

### Example 3: Focus on Specific Speaker

```bash
# Extract only host's segments (requires pyannote diarization)
python skills/autocut-shorts/scripts/autocut.py interview.mp4 \
  --diarization-model pyannote \
  --focus-speaker SPEAKER_00 \
  --platform tiktok
```

### Example 4: Batch Processing

```bash
# Process multiple YouTube URLs
echo "https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3" > urls.txt

python skills/batch-processor/scripts/batch_from_urls.py \
  --urls urls.txt \
  --num-clips 5 \
  --platform reels
```

### Example 5: Individual Skills

```bash
# Just download a YouTube video
python skills/youtube-downloader/scripts/download.py \
  "https://youtube.com/watch?v=VIDEO_ID" \
  --output video.mp4

# Just transcribe with Whisper
python skills/video-transcriber/scripts/transcribe.py video.mp4 \
  --model whisper \
  --whisper-model medium

# Just detect scenes
python skills/scene-detector/scripts/detect_scenes.py video.mp4

# Just resize to portrait
python skills/portrait-resizer/scripts/resize_to_portrait.py video.mp4
```

## 🗣️ Speaker Diarization

Trimer-Clip includes **pyannote-audio** integration for state-of-the-art speaker diarization:

- **35% better accuracy** than cloud API alternatives
- **Local processing** - no data leaves your machine
- **Overlapping speech detection** - identifies when people talk simultaneously
- **Automatic AI selection** - chooses pyannote vs Gemini based on content type

### Smart AI Selection

The AI agent automatically selects the best diarization method:

| Content Type | Auto-Selected | Reason |
|-------------|---------------|---------|
| **Podcast** | pyannote | Multi-speaker, high accuracy needed |
| **Interview** | pyannote | Precise speaker separation |
| **Panel Discussion** | pyannote | Handles 4+ speakers, overlapping speech |
| **Vlog** | Gemini | Single speaker, faster |
| **Tutorial** | Gemini | Single speaker, speed priority |
| **Gaming** | Gemini | Usually 1-2 speakers |

### Manual Override

```bash
# Force pyannote for accurate multi-speaker detection
python skills/autocut-shorts/scripts/autocut.py podcast.mp4 \
  --diarization-model pyannote

# Use Gemini for single speaker content (faster)
python skills/autocut-shorts/scripts/autocut.py vlog.mp4 \
  --diarization-model gemini

# Skip diarization entirely
python skills/autocut-shorts/scripts/autocut.py tutorial.mp4 \
  --diarization-model none
```

## 🎤 Transcription Options

### Whisper (Local) - Free

```bash
python skills/video-transcriber/scripts/transcribe.py video.mp4 \
  --model whisper \
  --whisper-model medium
```

- **Models**: tiny (fastest), base, small, medium, large-v3 (most accurate)
- **Cost**: 100% free
- **Privacy**: Everything stays local
- **Best for**: Sensitive content, high-volume processing

### Gemini API (Cloud)

```bash
python skills/video-transcriber/scripts/transcribe.py video.mp4 \
  --model gemini \
  --speaker-diarization \
  --emotion-detection
```

- **Model**: gemini-flash-lite-latest
- **Features**: Speaker diarization, emotion detection, context understanding
- **Best for**: Maximum accuracy, viral segment analysis
- **Cost**: Pay-per-use (Google AI pricing)

## 📊 Virality Scoring

The highlight scanner scores clips based on:

```
Virality Score = 
  35% × Transcript (hooks, viral phrases) +
  25% × Laughter (humorous moments) +
  25% × Sentiment (emotional peaks) +
  15% × Scenes (visual transitions)
```

**Score Interpretation:**
- **0.8-1.0**: Premium viral potential
- **0.6-0.8**: High potential
- **0.4-0.6**: Good potential

## 🎬 Output Specifications

- **Format**: MP4 (H.264 + AAC)
- **Resolution**: 1080×1920 or 720×1280 (9:16 portrait)
- **Duration**: 15-60 seconds (auto-detected)
- **Captions**: Burned-in with platform-specific styling
- **FPS**: 30fps
- **Bitrate**: 4000k video, 128k audio

## 🔌 Integration with AI Agents

### Claude Code / OpenCode

```bash
# Navigate to project
cd trimer-clip

# Claude automatically discovers skills
claude "Make 5 TikTok clips from this podcast"
```

### Gemini CLI

```bash
# Gemini uses the SKILL.md files for context
gemini "Extract viral moments from video.mp4"
```

### Skills CLI

```bash
npx skills add https://github.com/akrindev/trimer-clip
```

## 🐛 Troubleshooting

### FFmpeg not found

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### pyannote model download fails

```bash
# Accept user agreement at HuggingFace
# Visit: https://hf.co/pyannote/speaker-diarization-community-1
# Click "Access repository" and accept terms
```

### Out of memory

```bash
# Use smaller Whisper model
python skills/video-transcriber/scripts/transcribe.py video.mp4 \
  --model whisper \
  --whisper-model tiny  # Uses less memory

# Or skip diarization
python skills/autocut-shorts/scripts/autocut.py video.mp4 \
  --diarization-model none
```

### GPU not detected

```bash
# Install CUDA-enabled PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📚 Documentation

- **[SKILLS.md](SKILLS.md)** - Detailed skill reference
- **[references/README.md](references/README.md)** - FFmpeg commands, platform specs, troubleshooting
- **[Skill Documentation](skills/)** - Each skill has its own SKILL.md

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) first.

### Development Setup

```bash
git clone https://github.com/akrindev/trimer-clip.git
cd trimer-clip
pip install -e ".[dev]"
pre-commit install
```

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [OpenAI Whisper](https://github.com/openai/whisper) - Transcription
- [Google Gemini](https://ai.google.dev/gemini-api) - Cloud transcription
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Agent Skills](https://agentskills.io) - Standard for AI agent tools

## 👤 Author

**Syakirin Amin**
- GitHub: [@akrindev](https://github.com/akrindev)
- Email: akrinmin@gmail.com

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Trimer-Clip Agent Skills** - AI-powered video editing for short-form content creation
