---
name: sentiment-analyzer
description: Analyze sentiment and emotion in audio/video content. Use when you want to identify emotional peaks, detect positive/negative sentiment, find reaction moments, or analyze the emotional journey throughout the video. Supports both transcript-based and AI-based emotion detection.
allowed-tools: Bash(ffmpeg:*) Bash(python:*)
compatibility: Requires optional ML models for AI-based detection
metadata:
  version: "1.0"
  methods: "Transcript Keywords + AI Emotion Detection"
---

# Sentiment Analyzer

This skill enables AI agents to analyze sentiment and emotion in video content.

## When to Use

- User wants to identify emotional peaks in content
- Detecting positive/negative/neutral sentiment
- Finding reaction moments (surprise, excitement, anger)
- Analyzing the emotional journey throughout the video
- Creating highlight clips from emotional moments

## Detection Methods

### 1. Transcript-Based Keyword Detection

Analyzes transcript for emotion keywords:

**Positive:** excited, amazing, incredible, wow, love, happy, joy
**Negative:** terrible, awful, hate, sad, angry, frustrated, disappointing
**Surprise:** oh my god, what!, unbelievable, shocked, can't believe
**Enthusiasm:** let's go, come on, yes!, awesome, fantastic

### 2. AI-Based Emotion Detection (Gemini)

Uses Gemini API for advanced emotion analysis:
- Context-aware sentiment detection
- Emotion classification (happy, sad, angry, surprised, neutral)
- Intensity scoring
- Speaker emotion tracking

### 3. Audio Feature-Based Detection

Analyzes audio characteristics:
- Pitch variations
- Volume/loudness changes
- Speech rate changes
- Energy patterns

## Available Scripts

### `scripts/analyze_sentiment.py`

Analyze sentiment in video content.

**Usage:**
```bash
python skills/sentiment-analyzer/scripts/analyze_sentiment.py <video_path> [options]
```

**Options:**
- `--method`: Analysis method (keywords, ai, audio) - default: keywords
- `--transcript-path`: Path to transcript SRT/VTT file (for keyword detection)
- `--output, -o`: Output JSON path (default: `<video_path>_sentiment.json`)
- `--window-size`: Analysis window size in seconds - default: 5.0

**Examples:**

Analyze from transcript:
```bash
python skills/sentiment-analyzer/scripts/analyze_sentiment.py video.mp4 --transcript-path video.srt
```

Analyze with Gemini AI:
```bash
python skills/sentiment-analyzer/scripts/analyze_sentiment.py video.mp4 --method ai
```

### `scripts/find_emotional_peaks.py`

Find emotional peaks and significant moments.

**Usage:**
```bash
python skills/sentiment-analyzer/scripts/find_emotional_peaks.py <video_path> [options]
```

**Options:**
- `--threshold`: Peak detection threshold (0.0-1.0) - default: 0.7
- `--emotion-type`: Filter by emotion type (positive, negative, surprise, all) - default: all
- `--output, -o`: Output JSON path

**Example:**
```bash
python skills/sentiment-analyzer/scripts/find_emotional_peaks.py video.mp4 --threshold 0.8 --emotion-type positive
```

## Output Format

```json
{
  "video_path": "video.mp4",
  "method": "keywords",
  "overall_sentiment": {
    "positive": 0.45,
    "negative": 0.15,
    "neutral": 0.40
  },
  "emotional_peaks": [
    {
      "timestamp": 23.5,
      "duration": 3.2,
      "emotion": "positive",
      "intensity": 0.85,
      "text": "This is absolutely incredible!",
      "keywords": ["incredible"]
    },
    {
      "timestamp": 67.0,
      "duration": 2.8,
      "emotion": "surprise",
      "intensity": 0.90,
      "text": "Oh my god, I can't believe this!",
      "keywords": ["oh my god", "can't believe"]
    }
  ],
  "sentiment_timeline": [
    {
      "start": 0.0,
      "end": 30.0,
      "dominant_emotion": "neutral",
      "intensity": 0.4
    },
    {
      "start": 30.0,
      "end": 60.0,
      "dominant_emotion": "positive",
      "intensity": 0.7
    }
  ]
}
```

## Emotion Keywords

### Positive Emotions
- amazing, incredible, fantastic, awesome, wonderful, great, excellent
- love, happy, joy, excited, thrilled, delighted, pleased
- perfect, brilliant, outstanding, impressive, remarkable

### Negative Emotions
- terrible, awful, horrible, bad, terrible, disappointing
- hate, angry, frustrated, annoyed, upset, sad, depressed
- worst, disgusting, pathetic, useless, failed

### Surprise/Excitement
- wow, oh my god, what!, unbelievable, shocked, stunned
- surprised, amazed, astonished, incredible, no way
- excited, thrilled, pumped, fired up, let's go

### Neutral/Calm
- okay, alright, fine, good, normal, regular, standard

## Integration with Other Skills

After sentiment analysis, you can use these skills:

- `highlight-scanner`: Combine sentiment with other signals
- `video-trimmer`: Create clips from emotional peaks
- `autocut-shorts`: Full workflow for creating short clips

## Common Workflow

1. User provides video file
2. Transcribe using `video-transcriber`
3. Analyze sentiment using this skill
4. Find emotional peaks
5. Create short clips from emotional moments

## Tips

- Emotional peaks are excellent for viral content
- Positive + surprise emotions have highest viral potential
- Combine with laughter detection for even better results
- Consider surrounding context (3-5 seconds before/after)
- High-intensity emotions (>0.8) are premium clip candidates

## References

- Sentiment analysis research papers
- Emotion detection in NLP
- Audio emotion analysis techniques
