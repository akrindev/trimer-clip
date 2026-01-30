# Trimer-Clip References

## FFmpeg Commands Reference

### Basic Operations

**Get video info:**
```bash
ffprobe -v quiet -print_format json -show_format -show_streams video.mp4
```

**Extract audio:**
```bash
ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
```

**Trim video:**
```bash
# Stream copy (fast)
ffmpeg -i video.mp4 -ss 30 -t 10 -c copy output.mp4

# Re-encode (quality)
ffmpeg -i video.mp4 -ss 30 -t 10 -c:v libx264 -preset fast -crf 23 output.mp4
```

**Resize to portrait:**
```bash
ffmpeg -i video.mp4 -vf "crop=1080:1920:420:0" -c:a copy output.mp4
```

**Add subtitles:**
```bash
ffmpeg -i video.mp4 -vf "subtitles=subtitle.srt:force_style='FontName=Plus Jakarta Sans,FontSize=24'" -c:a copy output.mp4
```

### Advanced Filters

**Smart crop:**
```bash
ffmpeg -i video.mp4 -vf "cropdetect=24:16:0" -c:a copy output.mp4
```

**Scene detection:**
```bash
ffmpeg -i video.mp4 -vf "select='gt(scene,0.3)',showinfo" -f null -
```

## Video Processing Pipeline

```
Input Video
    |
    v
[Download from YouTube] (if URL)
    |
    v
[Extract Audio]
    |
    v
[Transcribe Audio]
    |
    v
[Analyze Content]
    - Scene Detection
    - Laughter Detection
    - Sentiment Analysis
    - Transcript Analysis
    |
    v
[Find Highlights]
    |
    v
[Trim Segments]
    |
    v
[Resize to Portrait]
    |
    v
[Add Subtitles]
    |
    v
Output Clips (Ready for upload)
```

## Platform Specifications

### TikTok
- **Aspect Ratio**: 9:16 (portrait)
- **Resolution**: 1080×1920
- **Duration**: 15 seconds - 3 minutes
- **Optimal**: 15-60 seconds
- **Format**: MP4 or MOV
- **Codec**: H.264
- **Audio**: AAC
- **File Size**: ≤ 287.6 MB (iOS), ≤ 72 MB (Android)

### YouTube Shorts
- **Aspect Ratio**: 9:16 (portrait)
- **Resolution**: 1080×1920
- **Duration**: Up to 60 seconds
- **Format**: MP4
- **Codec**: H.264
- **Audio**: AAC
- **File Size**: ≤ 2GB

### Instagram Reels
- **Aspect Ratio**: 9:16 (portrait)
- **Resolution**: 1080×1920
- **Duration**: 15-90 seconds
- **Format**: MP4 or MOV
- **Codec**: H.264
- **Audio**: AAC
- **File Size**: ≤ 4GB

### Facebook Reels
- **Aspect Ratio**: 9:16 (portrait)
- **Resolution**: 1080×1920
- **Duration**: Up to 90 seconds
- **Format**: MP4
- **Codec**: H.264
- **Audio**: AAC
- **File Size**: ≤ 1GB

## Viral Content Guidelines

### Hook Patterns
1. **"You won't believe..."** - Creates curiosity
2. **"This changed everything..."** - Implies transformation
3. **"The secret to..."** - Offers value
4. **"What nobody tells you..."** - Exclusive information
5. **"I made a huge mistake..."** - Story hook
6. **"Stop doing this..."** - Contrarian advice

### High-Engagement Moments
1. **Surprise reveals** - Unexpected outcomes
2. **Emotional peaks** - Strong reactions
3. **Humorous moments** - Laughter and comedy
4. **Quick tips** - Value in seconds
5. **Visual transformations** - Before/after
6. **Controversial takes** - Opinionated content

### Optimal Clip Structure
1. **0-3s**: Hook (attention grabber)
2. **3-15s**: Setup (context building)
3. **15-45s**: Content (main value)
4. **45-60s**: Call-to-action (engagement)

## Detection Algorithms

### Scene Detection
```python
# Content-aware detection
detector = ContentDetector(threshold=0.3, min_scene_len=0.5)
scenes = detect(video_path, detector)
```

### Laughter Detection
```python
# Keyword-based
def detect_laughter(transcript):
    keywords = ['laugh', 'laughter', 'ha ha', 'haha', 'lol']
    for keyword in keywords:
        if keyword in transcript.lower():
            return True
    return False
```

### Sentiment Analysis
```python
# Emotion keywords
emotions = {
    'positive': ['amazing', 'incredible', 'fantastic', 'love', 'happy'],
    'negative': ['terrible', 'awful', 'hate', 'angry', 'sad'],
    'surprise': ['wow', 'oh my god', 'unbelievable', 'shocked']
}
```

## API Reference

### Gemini API

**Transcription:**
```python
client = genai.Client(api_key=GEMINI_API_KEY)
file = client.files.upload(path=audio_path)
response = client.models.generate_content(
    model="gemini-flash-lite-latest",
    contents=["Transcribe this audio", file],
    config=GenerateContentConfig(audio_timestamp=True)
)
```

**Analysis:**
```python
response = client.models.generate_content(
    model="gemini-flash-lite-latest",
    contents=["Analyze this audio for viral segments", file]
)
```

### Whisper API

```python
import whisper
model = whisper.load_model("medium")
result = model.transcribe(audio_path)
```

## Troubleshooting

### Common Issues

**FFmpeg not found:**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

**Whisper model download fails:**
```bash
# Pre-download models
whisper audio.mp3 --model medium
```

**Gemini API rate limit:**
- Use Whisper for high-volume processing
- Implement retry logic
- Add delays between requests

**Memory issues with large videos:**
- Split video into segments first
- Process segments separately
- Use stream copy when possible

## Best Practices

### Workflow Optimization
1. Download all videos first
2. Transcribe in batch
3. Run detection modules in parallel
4. Process clips sequentially
5. Clean up temp files

### Quality Control
1. Preview highlights before trimming
2. Check virality scores (0.6+ recommended)
3. Verify portrait crop focuses on subject
4. Test subtitle readability
5. Validate output format

### Performance Tips
1. Use GPU for Whisper (if available)
2. Stream copy instead of re-encode when possible
3. Process multiple videos in parallel
4. Use faster-whisper for speed
5. Cache transcription results

## Resources

### Documentation
- FFmpeg: https://ffmpeg.org/documentation.html
- Whisper: https://github.com/openai/whisper
- Gemini: https://ai.google.dev/gemini-api/docs
- yt-dlp: https://github.com/yt-dlp/yt-dlp

### Tools
- PySceneDetect: https://www.scenedetect.com/
- MoviePy: https://zulko.github.io/moviepy/
- OpenCV: https://docs.opencv.org/

### Inspiration
- OpusClip: https://www.opus.pro/
- Vizard.ai: https://vizard.ai/
