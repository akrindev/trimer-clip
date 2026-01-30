#!/usr/bin/env python3
import sys
import os
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "skills"))


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


def autocut(
    source: str,
    source_type: str = "auto",
    num_clips: int = 5,
    min_duration: float = 15,
    max_duration: float = 60,
    platform: str = "tiktok",
    output_dir: str = "./shorts/",
    transcription_model: str = "auto",
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

        # Intelligent diarization model selection
        selected_diarization = select_diarization_model(video_path, user_request, diarization_model)

        step_start = time.time()

        if not skip_transcribe:
            if transcript_path and Path(transcript_path).exists():
                transcript_file = transcript_path
            else:
                transcript_file = transcribe_video(
                    video_path, model=transcription_model, gemini_api_key=gemini_api_key
                )

            if not transcript_file:
                return {"success": False, "error": "Transcription failed"}
        else:
            transcript_file = transcript_path

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

        clips = process_clips(
            video_path,
            highlights,
            output_dir,
            platform,
            style or platform,
            transcript_file,
            diarization_file,
        )

        timings["processing"] = time.time() - step_start

        total_time = time.time() - start_time

        return {
            "success": True,
            "source": {"type": source_type, "path": video_path, "info": video_info},
            "processing": {
                "transcription_model": transcription_model,
                "diarization_model": selected_diarization,
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
            return result.get("video_path"), result
        return None, {}

    except Exception as e:
        print(f"Download error: {e}")
        return None, {}


def transcribe_video(video_path: str, model: str, gemini_api_key: str = None) -> str:
    """Transcribe video audio."""
    try:
        from video_transcriber.scripts.transcribe import transcribe_video

        result = transcribe_video(
            video_path, model=model, output_path=f"{video_path}.srt", format="srt"
        )

        if result["success"]:
            return result["output_path"]
        return None

    except Exception as e:
        print(f"Transcription error: {e}")
        return None


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
) -> list:
    """Process clips through trim, resize, and subtitle."""
    clips = []

    try:
        from video_trimmer.scripts.trim import trim_video
        from portrait_resizer.scripts.resize_to_portrait import resize_to_portrait
        from subtitle_overlay.scripts.add_subtitles import add_subtitles

        for i, highlight in enumerate(highlights):
            base_name = Path(video_path).stem

            trimmed_path = f"{output_dir}/{base_name}_trimmed_{i + 1:03d}.mp4"
            trim_result = trim_video(
                video_path,
                str(highlight["suggested_clip_start"]),
                str(highlight["suggested_clip_end"]),
                trimmed_path,
                reencode=False,
            )

            if not trim_result["success"]:
                continue

            portrait_path = f"{output_dir}/{base_name}_portrait_{i + 1:03d}.mp4"
            resize_result = resize_to_portrait(
                trimmed_path, output_path=portrait_path, mode="smart"
            )

            if not resize_result["success"]:
                continue

            final_path = f"{output_dir}/{base_name}_{platform}_{i + 1:03d}.mp4"
            subtitle_result = add_subtitles(portrait_path, subtitle_path, final_path, style=style)

            if subtitle_result["success"]:
                clips.append(
                    {
                        "rank": highlight["rank"],
                        "filename": Path(final_path).name,
                        "start_time": highlight["start_time"],
                        "end_time": highlight["end_time"],
                        "duration": highlight["duration"],
                        "virality_score": highlight["virality_score"],
                        "output_path": final_path,
                    }
                )

            if Path(trimmed_path).exists():
                Path(trimmed_path).unlink()
            if Path(portrait_path).exists():
                Path(portrait_path).unlink()

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
        "--transcription-model", choices=["auto", "whisper", "gemini"], default="auto"
    )
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
        style=args.style,
        user_request=" ".join(sys.argv),  # Capture full command for context
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
