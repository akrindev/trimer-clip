---
name: youtube-downloader
description: Download videos from YouTube URLs. Use when user wants to download a YouTube video for processing, editing, or transcription. Supports different quality options, audio-only extraction, and playlist downloads.
allowed-tools: Bash(yt-dlp:*)
compatibility: Requires yt-dlp and FFmpeg
metadata:
  version: "1.0"
  platforms: "YouTube"
---

# YouTube Downloader

This skill enables AI agents to download videos from YouTube URLs for further processing, editing, or transcription.

## When to Use

- User wants to download a YouTube video
- User provides a YouTube URL that needs to be processed
- User wants to extract audio from a YouTube video
- User wants to download a YouTube playlist

## Setup

Ensure yt-dlp is installed:

```bash
pip install yt-dlp
```

## Available Scripts

### `scripts/download.py`

Download a video from YouTube URL.

**Usage:**
```bash
python skills/youtube-downloader/scripts/download.py <url> [options]
```

**Options:**
- `--output, -o`: Output path (default: `./downloads/%(title)s.%(ext)s`)
- `--format, -f`: Video format (default: `best[ext=mp4]`)
- `--audio-only`: Extract audio only
- `--audio-format`: Audio format when using --audio-only (default: `mp3`)
- `--quality`: Quality preset (best, 1080, 720, 480, 360)
- `--playlist`: Download entire playlist
- `--subtitle`: Download subtitles
- `--write-description`: Write video description
- `--write-info-json`: Write video metadata as JSON

**Examples:**

Download video in best quality:
```bash
python skills/youtube-downloader/scripts/download.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Download with custom output path:
```bash
python skills/youtube-downloader/scripts/download.py "https://www.youtube.com/watch?v=VIDEO_ID" --output "./my_video.mp4"
```

Extract audio only:
```bash
python skills/youtube-downloader/scripts/download.py "https://www.youtube.com/watch?v=VIDEO_ID" --audio-only --audio-format wav
```

Download 720p video:
```bash
python skills/youtube-downloader/scripts/download.py "https://www.youtube.com/watch?v=VIDEO_ID" --quality 720
```

Download with subtitles:
```bash
python skills/youtube-downloader/scripts/download.py "https://www.youtube.com/watch?v=VIDEO_ID" --subtitle --write-info-json
```

### `scripts/download_playlist.py`

Download entire YouTube playlist.

**Usage:**
```bash
python skills/youtube-downloader/scripts/download_playlist.py <playlist_url> [options]
```

**Options:**
- `--output, -o`: Output directory (default: `./downloads/%(playlist_title)s/%(title)s.%(ext)s`)
- `--start`: Start downloading from this video number
- `--end`: End downloading at this video number
- `--format`: Video format
- All other options from `download.py`

**Example:**
```bash
python skills/youtube-downloader/scripts/download_playlist.py "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

## Output

### Video Download

Returns a JSON with download information:

```json
{
  "success": true,
  "video_path": "/path/to/video.mp4",
  "title": "Video Title",
  "duration": 1234.5,
  "uploader": "Channel Name",
  "upload_date": "20240101",
  "view_count": 1000000,
  "thumbnail": "/path/to/thumbnail.jpg"
}
```

### Audio Download

Returns:
```json
{
  "success": true,
  "audio_path": "/path/to/audio.mp3",
  "title": "Video Title",
  "duration": 1234.5
}
```

## Quality Presets

| Preset | Resolution | Description |
|---------|-------------|-------------|
| `best` | Best available | Highest quality |
| `1080` | 1920x1080 | Full HD |
| `720` | 1280x720 | HD |
| `480` | 854x480 | SD |
| `360` | 640x360 | Low quality |

## Integration with Other Skills

After downloading, you can use these skills:

- `video-transcriber`: Transcribe the downloaded video
- `scene-detector`: Detect scene changes
- `portrait-resizer`: Convert to 9:16 portrait format
- `autocut-shorts`: Full workflow for creating short clips

## Common Workflow

1. User provides YouTube URL
2. Download video using this skill
3. Transcribe using `video-transcriber`
4. Create short clips using `autocut-shorts`

## Error Handling

- **Network errors**: Retry up to 3 times
- **Invalid URL**: Return clear error message
- **Video not available**: Return video status
- **Age-restricted**: May require authentication

## Tips

- Use `--audio-only` when you only need the transcript (saves bandwidth)
- Use `--write-info-json` to get video metadata for analysis
- For long videos, consider downloading in segments
- Use `--subtitle` if available video has captions

## References

- yt-dlp documentation: https://github.com/yt-dlp/yt-dlp
- YouTube URL formats: Regular, Short, Playlist, Embed
