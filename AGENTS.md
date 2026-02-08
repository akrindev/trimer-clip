# Agent Workflow Notes

Use these defaults for local testing and manual verification.

## Workflow Summary

Local-first pipeline:

1. Download (if URL)
2. Trim or detect highlights
3. Portrait resize (`--mode smart`)
4. Transcribe locally with Whisper tiny (fast validation)
5. Generate ASS karaoke (word-level)
6. Burn-in subtitles (use ASS styles)

Use API transcription only if local tiny fails or is too slow.

## Local-First Transcription

- Prefer local Whisper tiny for quick tests.
- Command example:

```bash
python skills/video-transcriber/scripts/transcribe.py video.mp4 \
  --model whisper --whisper-model tiny --format json
```

## Word-Level Captions

- Use word timestamps JSON to build ASS karaoke subtitles.
- Keep ASS styling (do not override): pass `--use-ass-style`.

## API Fallback Order (when needed)

1. OpenAI Whisper API (`--model openai`) with `OPENAI_API_KEY`
2. Google Speech-to-Text (`--model google`) with `GOOGLE_APPLICATION_CREDENTIALS`

## Portrait Resize

- Use `--mode smart` for face-aware crop.
- MediaPipe is optional; OpenCV Haar is the fallback.

## Subtitle Safe Area

- Keep captions above platform UI (safe area margin already increased in ASS presets).

## Output Structure

Each source video gets its own run folder. Each clip has a dedicated folder with a metadata JSON file.

```
shorts/
  <video_slug>_<YYYYMMDD-HHMMSS>/
    clip_001/
      master.mp4
      data.json
    clip_002/
      master.mp4
      data.json
```

## Clip Metadata (data.json)

Each clip folder includes a `data.json` with fields similar to yt-short-clipper:

- `title`, `hook_text`, `start_time`, `end_time`, `duration_seconds`
- `has_hook`, `has_captions`, `subtitle_mode`
- `rank`, `virality_score`, `text`, `reasoning`, `confidence`, `scores`
- `timestamps` (start/end + suggested clip range)
- `clip_filename`, `output_path`, `platform`
- `source` (type, path, url, title, uploader, duration, upload_date)
- `transcription` (model + model details)
- `youtube_title`, `youtube_description`, `youtube_tags`, `youtube_url`, `youtube_video_id`
