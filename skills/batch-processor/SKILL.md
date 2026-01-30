---
name: batch-processor
description: Process multiple videos in batch mode for efficiency. Supports batch download from YouTube URLs, batch autocut for multiple videos, and batch export to multiple platforms. Generates consolidated reports with all clips.
allowed-tools: Bash(ffmpeg:*) Bash(yt-dlp:*) Bash(python:*)
compatibility: Requires all trimer-clip dependencies
metadata:
  version: "1.0"
---

# Batch Processor

This skill enables batch processing of multiple videos for maximum efficiency.

## When to Use

- Processing multiple YouTube videos at once
- Batch converting podcast episodes to shorts
- Repurposing entire video libraries
- Creating content at scale
- Processing seasonal content (holiday, event videos)

## Available Scripts

### `scripts/batch_process.py`

Process multiple videos in batch.

**Usage:**
```bash
python skills/batch-processor/scripts/batch_process.py --input <json_file> [options]
```

**Input JSON Format:**
```json
[
  {
    "source": "https://youtube.com/watch?v=VIDEO1",
    "source_type": "youtube",
    "num_clips": 5,
    "platform": "tiktok"
  },
  {
    "source": "./videos/video2.mp4",
    "source_type": "file",
    "num_clips": 3,
    "platform": "shorts"
  }
]
```

**Options:**
- `--input`: JSON file with video list
- `--output-dir`: Output directory (default: `./batch_output/`)
- `--parallel`: Number of parallel processes (default: 1)
- `--transcription-model`: Transcription model (auto, whisper, gemini)

**Example:**
```bash
python skills/batch-processor/scripts/batch_process.py --input videos.json --parallel 2
```

### `scripts/batch_from_urls.py`

Download and process multiple YouTube URLs.

**Usage:**
```bash
python skills/batch-processor/scripts/batch_from_urls.py --urls <file> [options]
```

**URLs File Format:**
```
https://youtube.com/watch?v=VIDEO1
https://youtube.com/watch?v=VIDEO2
https://youtube.com/watch?v=VIDEO3
```

**Options:**
- `--urls`: File with YouTube URLs (one per line)
- `--num-clips`: Clips per video (default: 5)
- `--platform`: Target platform (default: tiktok)
- All other autocut options

**Example:**
```bash
python skills/batch-processor/scripts/batch_from_urls.py --urls urls.txt --num-clips 5 --platform shorts
```

## Output

### Directory Structure
```
batch_output/
  2024-01-30/
    video1/
      video1_tiktok_001.mp4
      video1_tiktok_002.mp4
      report.json
    video2/
      video2_shorts_001.mp4
      video2_shorts_002.mp4
      report.json
    batch_report.json
```

### Batch Report

```json
{
  "batch_id": "batch_20240130_120000",
  "total_videos": 10,
  "successful": 8,
  "failed": 2,
  "total_clips": 40,
  "output_dir": "./batch_output/2024-01-30/",
  "processing_time": 1800.5,
  "results": [
    {
      "source": "https://youtube.com/watch?v=...",
      "status": "success",
      "clips": 5,
      "output_dir": "batch_output/2024-01-30/video1/"
    },
    {
      "source": "https://youtube.com/watch?v=...",
      "status": "failed",
      "error": "Transcription failed"
    }
  ]
}
```

## Performance

- **Sequential processing**: 1 video at a time
- **Parallel processing**: 2-4 videos simultaneously (based on CPU cores)
- **Estimated time**: (video duration × 2) per video

## Error Handling

- **Individual failures**: Continues with next video
- **Partial success**: Reports successful vs failed
- **Retry logic**: Retries failed videos once
- **Progress tracking**: Shows progress during processing

## Tips

- Use parallel processing for large batches
- Process overnight for big libraries
- Check batch report for failed videos
- Retry failed videos individually
- Monitor disk space for large batches

## References

- Parallel processing in Python
- Batch workflow optimization
