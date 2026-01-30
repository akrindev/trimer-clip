# Trimer-Clip Agent Skills

Agent Skills for AI-powered video editing. These skills enable AI agents to automatically find viral moments in long videos (podcasts, vlogs, gaming, tutorials) and create short-form content (TikTok, YouTube Shorts, Instagram Reels) in 9:16 portrait format.

Built following the [Agent Skills](https://agentskills.io) open standard.

## Features

- **Automatic Video Clipping**: Detect highlights, funny moments, and viral-worthy segments
- **Multi-Platform Support**: Export for TikTok, YouTube Shorts, Instagram Reels (9:16 portrait)
- **AI-Powered Detection**: Scene changes, laughter, sentiment/emotion, transcript analysis
- **Advanced Speaker Diarization**: pyannote-audio integration for accurate multi-speaker detection
- **Smart AI Decisions**: Automatically selects best tools based on content type
- **Flexible Transcription**: Whisper (local) or Gemini API (gemini-flash-lite-latest)
- **Fully Automated**: AI agents can process entire workflow end-to-end
- **YouTube Download**: Download videos directly from YouTube URLs

## Available Skills

| Skill | Description |
|--------|-------------|
| `youtube-downloader` | Download videos from YouTube URLs |
| `video-transcriber` | Transcribe audio using Whisper or Gemini API |
| `speaker-diarization` | Advanced speaker diarization with pyannote-audio |
| `scene-detector` | Detect scene changes and shot boundaries |
| `laughter-detector` | Find humorous/laughing segments |
| `sentiment-analyzer` | Analyze emotion and sentiment |
| `highlight-scanner` | Combine all signals to find viral moments |
| `video-trimmer` | Trim/cut videos by timestamp |
| `portrait-resizer` | Convert to 9:16 vertical format |
| `subtitle-overlay` | Add captions to video clips |
| `autocut-shorts` | Main orchestration skill for full workflow |
| `batch-processor` | Process multiple videos at once |

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

Ensure FFmpeg is installed and available in your PATH:

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### 3. Set API Keys (Optional)

For Gemini API transcription:

```bash
export GEMINI_API_KEY="your-api-key"
```

For pyannote speaker diarization (get token at [huggingface.co](https://huggingface.co)):

```bash
export HUGGINGFACE_TOKEN="your-token"
```

Or create a `.env` file:
```
GEMINI_API_KEY=your-api-key
HUGGINGFACE_TOKEN=your-token
```

## Usage with AI Agents

### 1. Skills CLI (`npx skills`)

```bash
npx skills add https://github.com/your-org/trimer-clip
```

### 2. ClawdHub & Moltbot

```bash
# Install Moltbot and ClawdHub CLI
npm install -g moltbot clawdhub

# Initialize Moltbot
moltbot onboard

# Install trimer-clip skills
clawdhub install autocut-shorts --workdir ~/molt
```

### 3. Manual Installation

Agents like Claude Code, Gemini CLI, and OpenCode automatically discover skills in `skills/` directories.

```bash
git clone https://github.com/your-org/trimer-clip.git
```

## Example Workflow

User asks: *"Make 5 TikTok clips from this YouTube podcast interview"*

Agent (using `autocut-shorts` skill):
1. Downloads video
2. Transcribes audio (Whisper or Gemini)
3. Performs speaker diarization (pyannote - auto-selected for podcasts)
4. Detects scene changes
5. Finds laughter segments
6. Analyzes sentiment/emotions
7. Scans transcript for hooks and viral moments
8. Scores and selects 5 best segments
9. Trims each segment
10. Resizes to 9:16 portrait (smart crop)
11. Adds styled captions with speaker labels
12. Exports 5 MP4 files ready for TikTok

**Quick Start:**
```bash
# Single command - everything automatic
python skills/autocut-shorts/scripts/autocut.py "https://youtube.com/watch?v=VIDEO_ID" --platform tiktok

# From local file
python skills/autocut-shorts/scripts/autocut.py video.mp4 --num-clips 5 --platform shorts
```

## Speaker Diarization

Trimer-Clip now includes **pyannote-audio** integration for state-of-the-art speaker diarization:

- **35% better accuracy** than cloud API alternatives
- **Local processing** - no data leaves your machine
- **Overlapping speech detection** - identifies when people talk simultaneously
- **Automatic AI selection** - chooses pyannote vs Gemini based on content type

### When to Use pyannote vs Gemini

**AI automatically selects based on context:**

| Content Type | Recommended | Reason |
|-------------|-------------|---------|
| Podcast | pyannote | Multi-speaker, high accuracy needed |
| Interview | pyannote | Precise speaker separation |
| Panel Discussion | pyannote | Handles 4+ speakers, overlapping speech |
| Vlog | Gemini | Single speaker, faster |
| Tutorial | Gemini | Single speaker, speed priority |

**Override manually:**
```bash
# Force pyannote for accurate multi-speaker detection
python skills/autocut-shorts/scripts/autocut.py podcast.mp4 --diarization-model pyannote

# Use Gemini for single speaker content (faster)
python skills/autocut-shorts/scripts/autocut.py vlog.mp4 --diarization-model gemini

# Extract only specific speaker
python skills/autocut-shorts/scripts/autocut.py podcast.mp4 --focus-speaker SPEAKER_00
```

## Transcription Options

### Whisper (Local)

- **Models**: tiny, base, small, medium, large-v3
- **Cost**: Free
- **Privacy**: 100% local
- **Best for**: Sensitive content, high volume processing

### Gemini API (Cloud)

- **Model**: gemini-flash-lite-latest
- **Features**: Speaker diarization, emotion detection, context understanding
- **Best for**: Highest quality, viral potential analysis

## Default Output Specifications

- **Format**: MP4 (H.264 + AAC)
- **Resolution**: 1080x1920 or 720x1280 (9:16 portrait)
- **Duration**: 15-60 seconds (auto-detected)
- **Captions**: Burned-in, styled for readability

## License

MIT

## About

Trimer-Clip Agent Skills - AI-powered video editing for short-form content creation
