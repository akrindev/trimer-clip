#!/usr/bin/env python3
import sys
import os
import json
import argparse
import time
import re
import types
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper
from shared.subtitle_utils import build_karaoke_ass, load_word_timestamps, slice_words
from shared.video_utils import create_srt, format_srt_time, sanitize_filename

SKILLS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SKILLS_ROOT))


def _alias_skill_package(name: str, directory: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(directory)]
    sys.modules[name] = module


def _register_skill_aliases() -> None:
    aliases = {
        "youtube_downloader": "youtube-downloader",
        "video_transcriber": "video-transcriber",
        "speaker_diarization": "speaker-diarization",
        "scene_detector": "scene-detector",
        "laughter_detector": "laughter-detector",
        "sentiment_analyzer": "sentiment-analyzer",
        "highlight_scanner": "highlight-scanner",
        "video_trimmer": "video-trimmer",
        "portrait_resizer": "portrait-resizer",
        "subtitle_overlay": "subtitle-overlay",
        "autocut_shorts": "autocut-shorts",
    }
    for alias, folder in aliases.items():
        target = SKILLS_ROOT / folder
        if target.exists():
            _alias_skill_package(alias, target)


_register_skill_aliases()


def select_diarization_model(
    video_path: str, user_request: str = "", explicit_choice: str = "auto"
) -> str:
    """Intelligently select diarization model based on context."""

    if explicit_choice != "auto":
        return explicit_choice

    user_lower = user_request.lower()

    # Use pyannote for multi-speaker content
    multi_speaker_keywords = [
        "podcast",
        "interview",
        "panel",
        "debate",
        "discussion",
        "conversation",
        "talk show",
    ]
    if any(keyword in user_lower for keyword in multi_speaker_keywords):
        return "pyannote"

    # Use pyannote when accuracy is explicitly requested
    accuracy_keywords = ["accurate", "precise", "high quality", "best", "professional"]
    if any(keyword in user_lower for keyword in accuracy_keywords):
        return "pyannote"

    # Use pyannote for overlapping speech detection
    overlap_keywords = ["overlapping", "talk over", "interruption", "heated", "argument"]
    if any(keyword in user_lower for keyword in overlap_keywords):
        return "pyannote"

    # Use pyannote for privacy
    privacy_keywords = ["private", "offline", "local", "sensitive", "confidential"]
    if any(keyword in user_lower for keyword in privacy_keywords):
        return "pyannote"

    # Use gemini for single speaker / simple content
    single_speaker_keywords = ["vlog", "tutorial", "monologue", "single", "solo", "quick", "fast"]
    if any(keyword in user_lower for keyword in single_speaker_keywords):
        return "gemini"

    # Default: check audio characteristics
    try:
        ffmpeg = FFmpegWrapper()
        duration = ffmpeg.get_duration(video_path)

        # Longer content more likely to be multi-speaker
        if duration > 600:  # 10+ minutes
            return "pyannote"
    except:
        pass

    # Default to gemini for speed if uncertain
    return "gemini"


def _slugify(value: str) -> str:
    value = value.lower()
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "video"


def _extract_youtube_id(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = parsed.netloc.lower()
    path = parsed.path

    if "youtu.be" in host:
        video_id = path.strip("/")
        return video_id or None

    if "youtube.com" in host:
        if path == "/watch":
            query = parse_qs(parsed.query)
            return query.get("v", [None])[0]

        for prefix in ["/shorts/", "/embed/", "/v/"]:
            if path.startswith(prefix):
                parts = path.split("/")
                if len(parts) > 2:
                    return parts[2] or None

    return None


def _build_run_output_dir(base_dir: str, video_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = sanitize_filename(video_name)
    slug = _slugify(safe_name)
    run_dir = Path(base_dir) / f"{slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_clip_metadata(
    clip_dir: Path,
    highlight: dict,
    clip_start: float,
    clip_end: float,
    output_path: str,
    subtitle_mode: str,
    context: dict,
) -> str:
    source = context.get("source", {}) if isinstance(context, dict) else {}
    source_info = source.get("info", {}) if isinstance(source, dict) else {}

    youtube_url = source_info.get("url")
    if not isinstance(youtube_url, str):
        youtube_url = None
    youtube_video_id = source_info.get("video_id")
    if not youtube_video_id:
        youtube_video_id = _extract_youtube_id(youtube_url)
    youtube_title = source_info.get("title")
    youtube_description = source_info.get("description")
    youtube_tags = source_info.get("tags") if isinstance(source_info.get("tags"), list) else []

    title = source_info.get("title") or Path(source.get("path", output_path)).stem
    hook_text = highlight.get("hook_text")

    metadata = {
        "title": title,
        "hook_text": hook_text,
        "start_time": format_srt_time(clip_start),
        "end_time": format_srt_time(clip_end),
        "duration_seconds": round(clip_end - clip_start, 2),
        "has_hook": bool(hook_text),
        "has_captions": subtitle_mode in ["auto", "word", "segment"],
        "subtitle_mode": subtitle_mode,
        "youtube_title": youtube_title or title,
        "youtube_description": youtube_description,
        "youtube_tags": youtube_tags,
        "youtube_url": youtube_url,
        "youtube_video_id": youtube_video_id,
        "rank": highlight.get("rank"),
        "virality_score": highlight.get("virality_score"),
        "text": highlight.get("text"),
        "reasoning": highlight.get("reasoning"),
        "confidence": highlight.get("confidence"),
        "scores": highlight.get("scores"),
        "timestamps": {
            "start": highlight.get("start_time"),
            "end": highlight.get("end_time"),
            "suggested_start": clip_start,
            "suggested_end": clip_end,
        },
        "clip_filename": Path(output_path).name,
        "output_path": output_path,
        "platform": context.get("platform"),
        "source": {
            "type": source.get("type"),
            "path": source.get("path"),
            "url": source_info.get("url"),
            "title": source_info.get("title"),
            "uploader": source_info.get("uploader"),
            "duration": source_info.get("duration"),
            "upload_date": source_info.get("upload_date"),
        },
        "transcription": {
            "model": context.get("transcription_model"),
            "whisper_model": context.get("whisper_model")
            if context.get("transcription_model") == "whisper"
            else None,
            "openai_model": context.get("openai_model")
            if context.get("transcription_model") == "openai"
            else None,
            "google_model": context.get("google_model")
            if context.get("transcription_model") == "google"
            else None,
        },
    }

    metadata_path = clip_dir / "data.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return str(metadata_path)


def autocut(
    source: str,
    source_type: str = "auto",
    num_clips: int = 5,
    min_duration: float = 15,
    max_duration: float = 60,
    platform: str = "tiktok",
    output_dir: str = "./shorts/",
    transcription_model: str = "auto",
    whisper_model: str = "large-v3",
    openai_model: str = "whisper-1",
    google_model: str = None,
    diarization_model: str = "auto",
    huggingface_token: str = None,
    focus_speaker: str = None,
    gemini_api_key: str = None,
    skip_transcribe: bool = False,
    skip_diarization: bool = False,
    skip_scenes: bool = False,
    skip_laughter: bool = False,
    skip_sentiment: bool = False,
    transcript_path: str = None,
    word_timestamps_path: str = None,
    subtitle_mode: str = "auto",
    style: str = None,
    user_request: str = "",
) -> dict:
    """Main autocut workflow with intelligent diarization selection."""

    start_time = time.time()
    timings = {}

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        step_start = time.time()

        if source_type == "auto":
            if source.startswith(("http://", "https://", "www.")):
                source_type = "youtube"
            else:
                source_type = "file"

        video_path = source
        video_info = {}

        if source_type == "youtube":
            video_path, video_info = download_video(source, output_dir)
            if not video_path:
                return {"success": False, "error": "Failed to download video"}

        if not Path(video_path).exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        timings["download"] = time.time() - step_start

        video_name = video_info.get("title") if isinstance(video_info, dict) else None
        if not video_name:
            video_name = Path(video_path).stem

        run_output_dir = _build_run_output_dir(output_dir, video_name)
        output_dir = str(run_output_dir)

        # Intelligent diarization model selection
        selected_diarization = select_diarization_model(video_path, user_request, diarization_model)

        step_start = time.time()

        subtitle_mode = (subtitle_mode or "auto").lower()
        needs_word_timestamps = subtitle_mode in ["auto", "word"]
        word_timestamps = None
        word_timestamps_file = word_timestamps_path
        transcript_file = None

        effective_transcription_model = transcription_model
        if needs_word_timestamps and transcription_model in ["auto", "gemini"]:
            effective_transcription_model = "whisper"

        if not skip_transcribe:
            if transcript_path and Path(transcript_path).exists():
                if transcript_path.endswith(".json"):
                    word_timestamps_file = transcript_path
                else:
                    transcript_file = transcript_path
            else:
                if needs_word_timestamps:
                    word_timestamps_file = transcribe_video(
                        video_path,
                        model=effective_transcription_model,
                        whisper_model=whisper_model,
                        openai_model=openai_model,
                        google_model=google_model,
                        output_format="json",
                        gemini_api_key=gemini_api_key,
                    )
                else:
                    transcript_file = transcribe_video(
                        video_path,
                        model=effective_transcription_model,
                        whisper_model=whisper_model,
                        openai_model=openai_model,
                        google_model=google_model,
                        output_format="srt",
                        gemini_api_key=gemini_api_key,
                    )

        if word_timestamps_file and Path(word_timestamps_file).exists():
            transcript_file = transcript_file or _ensure_srt_from_word_json(
                word_timestamps_file, output_path=f"{video_path}.srt"
            )
            word_timestamps = load_word_timestamps(word_timestamps_file)

        if skip_transcribe:
            if transcript_path and Path(transcript_path).exists():
                if transcript_path.endswith(".json") and not word_timestamps:
                    word_timestamps_file = transcript_path
                    transcript_file = _ensure_srt_from_word_json(word_timestamps_file)
                    word_timestamps = load_word_timestamps(word_timestamps_file)
                else:
                    transcript_file = transcript_path

            if word_timestamps_path and Path(word_timestamps_path).exists() and not word_timestamps:
                word_timestamps_file = word_timestamps_path
                transcript_file = transcript_file or _ensure_srt_from_word_json(
                    word_timestamps_file
                )
                word_timestamps = load_word_timestamps(word_timestamps_file)

        if not transcript_file:
            return {"success": False, "error": "Transcript file not found"}

        if needs_word_timestamps and not word_timestamps:
            subtitle_mode = "segment"

        timings["transcription"] = time.time() - step_start

        # Speaker diarization
        step_start = time.time()
        diarization_file = None

        if not skip_diarization and selected_diarization != "none":
            diarization_file = diarize_video(
                video_path,
                model=selected_diarization,
                huggingface_token=huggingface_token,
                gemini_api_key=gemini_api_key,
            )

            if diarization_file:
                timings["diarization"] = time.time() - step_start

        # Detection modules
        step_start = time.time()

        scenes_file = None
        if not skip_scenes:
            scenes_file = detect_scenes(video_path)

        laughter_file = None
        if not skip_laughter:
            laughter_file = detect_laughter(video_path, transcript_file)

        sentiment_file = None
        if not skip_sentiment:
            sentiment_file = analyze_sentiment(video_path, transcript_file)

        # Find highlights with speaker context
        highlights = find_highlights(
            video_path,
            transcript_file,
            diarization_file,
            scenes_file,
            laughter_file,
            sentiment_file,
            num_clips,
            min_duration,
            max_duration,
            focus_speaker=focus_speaker,
        )

        timings["analysis"] = time.time() - step_start

        if not highlights:
            return {"success": False, "error": "No highlights found"}

        # Process clips
        step_start = time.time()

        metadata_context = {
            "source": {"type": source_type, "path": video_path, "info": video_info},
            "platform": platform,
            "style": style or platform,
            "subtitle_mode": subtitle_mode,
            "transcription_model": effective_transcription_model,
            "whisper_model": whisper_model,
            "openai_model": openai_model,
            "google_model": google_model,
        }

        clips = process_clips(
            video_path,
            highlights,
            output_dir,
            platform,
            style or platform,
            transcript_file,
            diarization_file,
            word_timestamps=word_timestamps,
            subtitle_mode=subtitle_mode,
            metadata_context=metadata_context,
        )

        timings["processing"] = time.time() - step_start

        total_time = time.time() - start_time

        return {
            "success": True,
            "source": {"type": source_type, "path": video_path, "info": video_info},
            "processing": {
                "transcription_model": effective_transcription_model,
                "whisper_model": whisper_model
                if effective_transcription_model == "whisper"
                else None,
                "openai_model": openai_model if effective_transcription_model == "openai" else None,
                "diarization_model": selected_diarization,
                "subtitle_mode": subtitle_mode,
                "platform": platform,
                "num_clips_requested": num_clips,
                "num_clips_generated": len(clips),
            },
            "results": {"clips": clips, "output_dir": output_dir},
            "performance": {
                "total_time": round(total_time, 2),
                "download_time": round(timings.get("download", 0), 2),
                "transcription_time": round(timings.get("transcription", 0), 2),
                "diarization_time": round(timings.get("diarization", 0), 2),
                "analysis_time": round(timings.get("analysis", 0), 2),
                "processing_time": round(timings.get("processing", 0), 2),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e), "source": source}


def download_video(url: str, output_dir: str) -> tuple:
    """Download video from YouTube."""
    try:
        from youtube_downloader.scripts.download import download_video as yt_download

        result = yt_download(url, output_path=f"{output_dir}/%(title)s.%(ext)s")

        if result["success"]:
            video_path = result.get("video_path")
            if not video_path:
                candidates = list(Path(output_dir).glob("*.mp4"))
                if candidates:
                    latest = max(candidates, key=lambda p: p.stat().st_mtime)
                    video_path = str(latest)
                    result["video_path"] = video_path
            return video_path, result
        return None, {}

    except Exception as e:
        print(f"Download error: {e}")
        return None, {}


def transcribe_video(
    video_path: str,
    model: str,
    whisper_model: str = "medium",
    openai_model: str = "whisper-1",
    google_model: str = None,
    output_format: str = "srt",
    gemini_api_key: str = None,
) -> str:
    """Transcribe video audio."""
    try:
        from video_transcriber.scripts.transcribe import transcribe_video

        output_path = f"{video_path}.{output_format}"
        result = transcribe_video(
            video_path,
            model=model,
            whisper_model=whisper_model,
            openai_model=openai_model,
            google_model=google_model,
            output_path=output_path,
            format=output_format,
        )

        if result["success"]:
            return result["output_path"]
        return None

    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def _load_segments_from_json(json_path: str) -> list:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if isinstance(data, dict):
        data = data.get("segments", [])

    if not isinstance(data, list):
        return []

    segments = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "start" in item and "end" in item and "text" in item:
            segments.append(item)
    return segments


def _ensure_srt_from_word_json(json_path: str, output_path: str = None) -> str:
    segments = _load_segments_from_json(json_path)
    if not segments:
        return None

    if output_path is None:
        output_path = str(Path(json_path).with_suffix(".srt"))

    subtitles = []
    for idx, segment in enumerate(segments, 1):
        subtitles.append(
            {
                "index": idx,
                "start": float(segment.get("start", 0)),
                "end": float(segment.get("end", 0)),
                "text": str(segment.get("text", "")).strip(),
            }
        )

    create_srt(subtitles, output_path)
    return output_path


def diarize_video(
    video_path: str, model: str, huggingface_token: str = None, gemini_api_key: str = None
) -> str:
    """Perform speaker diarization."""
    try:
        if model == "pyannote":
            from speaker_diarization.scripts.diarize import diarize_video as pyannote_diarize

            result = pyannote_diarize(
                video_path, output_format="json", huggingface_token=huggingface_token
            )

            if isinstance(result, dict) and result.get("success"):
                output_path = f"{video_path}_diarization.json"
                with open(output_path, "w") as f:
                    json.dump(result, f)
                return output_path

        elif model == "gemini":
            # Use Gemini for diarization
            from video_transcriber.scripts.transcribe import transcribe_video

            result = transcribe_video(
                video_path,
                model="gemini",
                output_path=f"{video_path}_gemini.srt",
                format="srt",
                speaker_diarization=True,
            )

            if result["success"]:
                return result["output_path"]

        return None

    except Exception as e:
        print(f"Diarization error: {e}")
        return None


def detect_scenes(video_path: str) -> str:
    """Detect scene changes."""
    try:
        from scene_detector.scripts.detect_scenes import detect_scenes as sd

        output_path = f"{video_path}_scenes.json"
        result = sd(video_path)

        if result["success"]:
            with open(output_path, "w") as f:
                json.dump(result, f)
            return output_path
        return None

    except Exception as e:
        print(f"Scene detection error: {e}")
        return None


def detect_laughter(video_path: str, transcript_path: str) -> str:
    """Detect laughter segments."""
    try:
        from laughter_detector.scripts.detect_laughter import detect_laughter as ld

        output_path = f"{video_path}_laughter.json"
        result = ld(video_path, transcript_path=transcript_path)

        if result["success"]:
            with open(output_path, "w") as f:
                json.dump(result, f)
            return output_path
        return None

    except Exception as e:
        print(f"Laughter detection error: {e}")
        return None


def analyze_sentiment(video_path: str, transcript_path: str) -> str:
    """Analyze sentiment."""
    try:
        from sentiment_analyzer.scripts.analyze_sentiment import analyze_sentiment as sa

        output_path = f"{video_path}_sentiment.json"
        result = sa(video_path, transcript_path=transcript_path)

        if result["success"]:
            with open(output_path, "w") as f:
                json.dump(result, f)
            return output_path
        return None

    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return None


def find_highlights(
    video_path: str,
    transcript_path: str,
    diarization_path: str,
    scenes_path: str,
    laughter_path: str,
    sentiment_path: str,
    num_clips: int,
    min_duration: float,
    max_duration: float,
    focus_speaker: str = None,
) -> list:
    """Find highlight segments with speaker context."""
    try:
        from highlight_scanner.scripts.find_highlights import find_highlights as fh

        result = fh(
            video_path,
            transcript_path=transcript_path,
            scenes_path=scenes_path,
            laughter_path=laughter_path,
            sentiment_path=sentiment_path,
            num_clips=num_clips,
            min_duration=min_duration,
            max_duration=max_duration,
        )

        if result["success"]:
            highlights = result["highlights"]

            # Filter by specific speaker if requested
            if focus_speaker and diarization_path:
                with open(diarization_path, "r") as f:
                    diarization = json.load(f)

                # Adjust highlights to focus on specific speaker's segments
                speaker_segments = [
                    s for s in diarization.get("segments", []) if s["speaker"] == focus_speaker
                ]

                if speaker_segments:
                    highlights = _adjust_highlights_for_speaker(highlights, speaker_segments)

            return highlights
        return []

    except Exception as e:
        print(f"Highlight finding error: {e}")
        return []


def _adjust_highlights_for_speaker(highlights: list, speaker_segments: list) -> list:
    """Adjust highlight timestamps to focus on specific speaker."""
    adjusted = []

    for highlight in highlights:
        # Find overlapping speaker segments
        for seg in speaker_segments:
            if seg["start"] <= highlight["end_time"] and seg["end"] >= highlight["start_time"]:
                # Adjust to focus on speaker
                adjusted_start = max(highlight["start_time"], seg["start"] - 2)
                adjusted_end = min(highlight["end_time"], seg["end"] + 2)

                if adjusted_end - adjusted_start >= 10:  # Minimum 10 seconds
                    adjusted.append(
                        {
                            **highlight,
                            "start_time": adjusted_start,
                            "end_time": adjusted_end,
                            "duration": adjusted_end - adjusted_start,
                            "speaker_focus": True,
                        }
                    )
                    break

    return adjusted if adjusted else highlights


def process_clips(
    video_path: str,
    highlights: list,
    output_dir: str,
    platform: str,
    style: str,
    subtitle_path: str,
    diarization_path: str = None,
    word_timestamps: list = None,
    subtitle_mode: str = "auto",
    metadata_context: dict = None,
) -> list:
    """Process clips through trim, resize, and subtitle."""
    clips = []

    use_word_subtitles = subtitle_mode in ["auto", "word"] and word_timestamps
    context = metadata_context or {}

    try:
        from video_trimmer.scripts.trim import trim_video
        from portrait_resizer.scripts.resize_to_portrait import resize_to_portrait
        from subtitle_overlay.scripts.add_subtitles import add_subtitles

        for i, highlight in enumerate(highlights):
            clip_id = f"clip_{i + 1:03d}"
            clip_dir = Path(output_dir) / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)

            clip_start = float(highlight["suggested_clip_start"])
            clip_end = float(highlight["suggested_clip_end"])

            trimmed_path = str(clip_dir / "trimmed.mp4")
            trim_result = trim_video(
                video_path,
                str(clip_start),
                str(clip_end),
                trimmed_path,
                reencode=False,
            )

            if not trim_result["success"]:
                continue

            portrait_path = str(clip_dir / "portrait.mp4")
            resize_result = resize_to_portrait(
                trimmed_path, output_path=portrait_path, mode="smart"
            )

            if not resize_result["success"]:
                continue

            final_path = str(clip_dir / "master.mp4")
            subtitle_source = subtitle_path
            force_style = True
            ass_path = None

            if use_word_subtitles:
                clip_words = slice_words(word_timestamps, clip_start, clip_end)
                if clip_words:
                    output_resolution = resize_result.get("output_resolution", {})
                    video_width = output_resolution.get("width", 1080)
                    video_height = output_resolution.get("height", 1920)
                    ass_path = str(clip_dir / "captions.ass")
                    ass_file = build_karaoke_ass(
                        clip_words,
                        ass_path,
                        video_width=video_width,
                        video_height=video_height,
                        style=style,
                    )
                    if ass_file:
                        subtitle_source = ass_file
                        force_style = False

            subtitle_result = add_subtitles(
                portrait_path,
                subtitle_source,
                final_path,
                style=style,
                force_style=force_style,
            )

            if subtitle_result["success"]:
                metadata_path = _write_clip_metadata(
                    clip_dir,
                    highlight,
                    clip_start,
                    clip_end,
                    final_path,
                    subtitle_mode,
                    context,
                )
                clips.append(
                    {
                        "rank": highlight["rank"],
                        "filename": Path(final_path).name,
                        "start_time": highlight["start_time"],
                        "end_time": highlight["end_time"],
                        "duration": highlight["duration"],
                        "virality_score": highlight["virality_score"],
                        "output_path": final_path,
                        "clip_dir": str(clip_dir),
                        "metadata_path": metadata_path,
                    }
                )

            if Path(trimmed_path).exists():
                Path(trimmed_path).unlink()
            if Path(portrait_path).exists():
                Path(portrait_path).unlink()
            if ass_path and Path(ass_path).exists():
                Path(ass_path).unlink()

        return clips

    except Exception as e:
        print(f"Clip processing error: {e}")
        return clips


def main():
    parser = argparse.ArgumentParser(description="Autocut shorts from video")
    parser.add_argument("source", help="Video file path or YouTube URL")
    parser.add_argument("--source-type", choices=["auto", "file", "youtube"], default="auto")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=15)
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument(
        "--platform", choices=["tiktok", "shorts", "reels", "facebook"], default="tiktok"
    )
    parser.add_argument("--output-dir", default="./shorts/")
    parser.add_argument(
        "--transcription-model",
        choices=["auto", "whisper", "gemini", "openai", "google"],
        default="auto",
    )
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        default="large-v3",
    )
    parser.add_argument("--openai-model", default="whisper-1")
    parser.add_argument("--google-model", default=None)
    parser.add_argument(
        "--diarization-model", choices=["auto", "pyannote", "gemini", "none"], default="auto"
    )
    parser.add_argument("--huggingface-token", default=os.getenv("HUGGINGFACE_TOKEN"))
    parser.add_argument("--focus-speaker", help="Focus on specific speaker (SPEAKER_00, etc.)")
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"))
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-diarization", action="store_true")
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--skip-laughter", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--transcript-path")
    parser.add_argument("--word-timestamps-path")
    parser.add_argument("--subtitle-mode", choices=["auto", "word", "segment"], default="auto")
    parser.add_argument("--style", choices=["tiktok", "shorts", "reels", "default"])

    args = parser.parse_args()

    result = autocut(
        source=args.source,
        source_type=args.source_type,
        num_clips=args.num_clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        platform=args.platform,
        output_dir=args.output_dir,
        transcription_model=args.transcription_model,
        whisper_model=args.whisper_model,
        openai_model=args.openai_model,
        google_model=args.google_model,
        diarization_model=args.diarization_model,
        huggingface_token=args.huggingface_token,
        focus_speaker=args.focus_speaker,
        gemini_api_key=args.gemini_api_key,
        skip_transcribe=args.skip_transcribe,
        skip_diarization=args.skip_diarization,
        skip_scenes=args.skip_scenes,
        skip_laughter=args.skip_laughter,
        skip_sentiment=args.skip_sentiment,
        transcript_path=args.transcript_path,
        word_timestamps_path=args.word_timestamps_path,
        subtitle_mode=args.subtitle_mode,
        style=args.style,
        user_request=" ".join(sys.argv),  # Capture full command for context
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
