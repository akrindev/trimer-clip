# Trimer-Clip Agent Skills

Complete collection of video editing and autocut skills for AI agents.

## Overview

Trimer-Clip is a comprehensive suite of Agent Skills that enable AI agents to automatically create short-form content (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels) from long videos.

## Skills Reference

### 1. youtube-downloader
**Purpose**: Download videos from YouTube URLs

**Key Features**:
- Download from YouTube URLs
- Extract audio only
- Support for playlists
- Multiple quality options
- Subtitle download

**Scripts**:
- `scripts/download.py` - Download single video
- `scripts/download_playlist.py` - Download entire playlist

### 2. video-transcriber
**Purpose**: Transcribe audio from videos

**Key Features**:
- **Whisper** (local): tiny, base, small, medium, large-v3 models
- **Gemini API** (cloud): gemini-flash-lite-latest
- Speaker diarization (Gemini)
- Emotion detection (Gemini)
- Multiple output formats (SRT, VTT, JSON)

**Scripts**:
- `scripts/transcribe.py` - Transcribe audio
- `scripts/analyze.py` - Analyze audio content

### 3. scene-detector
**Purpose**: Detect scene changes and shot boundaries

**Key Features**:
- Adaptive detection algorithm
- Configurable threshold
- Scene splitting
- Cut point identification

**Scripts**:
- `scripts/detect_scenes.py` - Detect scenes in video

### 4. laughter-detector
**Purpose**: Detect laughter and humorous moments

**Key Features**:
- Keyword-based detection
- Audio feature analysis
- Confidence scoring
- Segment extraction

**Scripts**:
- `scripts/detect_laughter.py` - Detect laughter segments

### 5. sentiment-analyzer
**Purpose**: Analyze sentiment and emotion

**Key Features**:
- Positive/negative/neutral detection
- Emotional peak identification
- Keyword-based analysis
- AI-powered analysis (Gemini)

**Scripts**:
- `scripts/analyze_sentiment.py` - Analyze sentiment

### 6. highlight-scanner
**Purpose**: Find viral-worthy highlights

**Key Features**:
- Combined analysis (transcript + laughter + sentiment + scenes)
- Virality scoring algorithm
- Ranking and selection
- Multi-signal integration

**Scripts**:
- `scripts/find_highlights.py` - Find highlight segments

### 7. video-trimmer
**Purpose**: Trim and cut videos

**Key Features**:
- Time-based trimming
- Stream copy (fast)
- Re-encode (quality)
- Multiple segment trimming

**Scripts**:
- `scripts/trim.py` - Trim single segment
- `scripts/trim_multiple.py` - Trim multiple segments

### 8. portrait-resizer
**Purpose**: Convert to 9:16 portrait format

**Key Features**:
- Smart crop (focus on subjects)
- Center crop
- Letterbox mode
- 1080x1920 resolution

**Scripts**:
- `scripts/resize_to_portrait.py` - Resize video
- `scripts/batch_resize.py` - Batch resize

### 9. subtitle-overlay
**Purpose**: Add burned-in subtitles

**Key Features**:
- SRT/VTT support
- Platform-specific styles (TikTok, Shorts, Reels)
- Customizable fonts and colors
- Position control

**Scripts**:
- `scripts/add_subtitles.py` - Add subtitles to video

### 10. autocut-shorts
**Purpose**: Main orchestration for complete workflow

**Key Features**:
- Downloads video (if URL)
- Transcribes audio
- Detects highlights (combined analysis)
- Trims segments
- Resizes to portrait
- Adds subtitles
- Exports clips

**Scripts**:
- `scripts/autocut.py` - Full autocut workflow
- `scripts/quick_cut.py` - Quick cut from timestamps

### 11. batch-processor
**Purpose**: Batch process multiple videos

**Key Features**:
- Multiple video processing
- Parallel execution
- Consolidated reporting
- Error handling

**Scripts**:
- `scripts/batch_process.py` - Batch process
- `scripts/batch_from_urls.py` - Batch from URLs

## Skill Dependencies

```
autocut-shorts (orchestrator)
├── youtube-downloader
├── video-transcriber
├── scene-detector (optional)
├── laughter-detector (optional)
├── sentiment-analyzer (optional)
├── highlight-scanner
├── video-trimmer
├── portrait-resizer
└── subtitle-overlay
```

## Quick Start

### Basic Usage

1. **Download and autocut from YouTube:**
```bash
python skills/autocut-shorts/scripts/autocut.py "https://youtube.com/watch?v=VIDEO_ID"
```

2. **Autocut local video:**
```bash
python skills/autocut-shorts/scripts/autocut.py video.mp4 --num-clips 5
```

3. **Create TikTok clips:**
```bash
python skills/autocut-shorts/scripts/autocut.py video.mp4 --platform tiktok --style tiktok
```

### Step-by-Step Workflow

1. **Transcribe video:**
```bash
python skills/video-transcriber/scripts/transcribe.py video.mp4
```

2. **Find highlights:**
```bash
python skills/highlight-scanner/scripts/find_highlights.py video.mp4 --transcript-path video.srt --num-clips 5
```

3. **Process each clip:**
```bash
# Trim
python skills/video-trimmer/scripts/trim.py video.mp4 --start 30 --end 60 -o clip.mp4

# Resize
python skills/portrait-resizer/scripts/resize_to_portrait.py clip.mp4 -o clip_portrait.mp4

# Add subtitles
python skills/subtitle-overlay/scripts/add_subtitles.py clip_portrait.mp4 --subtitle video.srt -o final.mp4
```

## Configuration

### Environment Variables

```bash
# Gemini API (for transcription and analysis)
export GEMINI_API_KEY="your-api-key"

# Optional: Vertex AI settings
export GOOGLE_PROJECT_ID="your-project-id"
export GOOGLE_LOCATION="us-central1"
```

### Model Selection

**Transcription:**
- `auto` - Automatically selects best model
- `whisper` - Use local Whisper (privacy, free)
- `gemini` - Use Gemini API (quality, features)

**Whisper Models:**
- `tiny` - Fastest, lowest accuracy
- `base` - Fast, good accuracy
- `small` - Balanced
- `medium` - Good accuracy
- `large-v3` - Best accuracy, slowest

## Output Specifications

### Video Format
- **Container**: MP4
- **Video Codec**: H.264 (libx264)
- **Audio Codec**: AAC
- **Resolution**: 1080x1920 (9:16 portrait)
- **Frame Rate**: 30fps
- **Bitrate**: 4000k

### Clip Duration
- **TikTok**: 15-60 seconds
- **YouTube Shorts**: 15-60 seconds
- **Instagram Reels**: 15-90 seconds
- **Facebook Reels**: 15-90 seconds

## Virality Scoring

### Algorithm
```
virality_score = (
    transcript_score × 0.35 +    # Hooks, viral phrases
    laughter_score × 0.25 +      # Humorous moments
    sentiment_score × 0.25 +     # Emotional peaks
    scene_score × 0.15           # Visual transitions
)
```

### Score Interpretation
- **0.8-1.0**: Premium viral potential
- **0.6-0.8**: High potential
- **0.4-0.6**: Good potential
- **0.2-0.4**: Moderate potential
- **0.0-0.2**: Low potential

## Platform Presets

### TikTok
- Resolution: 1080x1920
- Duration: 15-60s
- Style: Arial Bold, 28px, white with black outline

### YouTube Shorts
- Resolution: 1080x1920
- Duration: 15-60s
- Style: Roboto, 26px, white with black outline

### Instagram Reels
- Resolution: 1080x1920
- Duration: 15-90s
- Style: SF Pro, 24px, white with black outline

## Performance Benchmarks

| Video Length | Processing Time |
|--------------|-----------------|
| 1 minute | 30-60 seconds |
| 10 minutes | 3-5 minutes |
| 30 minutes | 8-12 minutes |
| 1 hour | 15-25 minutes |

## Tips and Best Practices

1. **Use Gemini for best results** - Higher quality transcription and better highlight detection
2. **Process in portrait last** - More efficient workflow
3. **Keep 2-3 second buffers** - Provides context around highlights
4. **Test different platforms** - Performance varies by audience
5. **Batch process overnight** - For large video libraries
6. **Monitor virality scores** - Filter by 0.6+ for premium content
7. **Use scene detection** - Helps with clean cuts
8. **Add laughter detection** - Improves viral potential for comedy content

## References

- Agent Skills Standard: https://agentskills.io
- FFmpeg: https://ffmpeg.org
- Whisper: https://github.com/openai/whisper
- Gemini API: https://ai.google.dev/gemini-api
- yt-dlp: https://github.com/yt-dlp/yt-dlp

## Support

For issues and feature requests, visit the project repository.
