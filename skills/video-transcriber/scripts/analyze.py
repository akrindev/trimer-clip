#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper
from shared.audio_utils import extract_audio_from_video
from models.gemini_transcriber import GeminiTranscriber
import tempfile


def analyze_video(
    video_path: str, analysis_type: str = "viral", num_segments: int = 5, model: str = "gemini"
) -> dict:
    """Analyze video audio content."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    if model != "gemini":
        return {"success": False, "error": "Analysis only supported with Gemini API"}

    try:
        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        extract_audio_from_video(video_path, temp_audio_path, sample_rate=16000)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"success": False, "error": "GEMINI_API_KEY environment variable not set"}

        transcriber = GeminiTranscriber(api_key=api_key, model="gemini-flash-lite-latest")

        result = transcriber.analyze_audio(temp_audio_path, analysis_type=analysis_type)

        os.unlink(temp_audio_path)

        return {
            "success": True,
            "analysis_type": analysis_type,
            "analysis": result["analysis"],
            "video_path": video_path,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def main():
    parser = argparse.ArgumentParser(description="Analyze video audio content")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--analysis-type", choices=["viral", "summary", "emotions", "questions"], default="viral"
    )
    parser.add_argument("--num-segments", type=int, default=5)
    parser.add_argument("--model", choices=["gemini"], default="gemini")

    args = parser.parse_args()

    result = analyze_video(
        video_path=args.video_path,
        analysis_type=args.analysis_type,
        num_segments=args.num_segments,
        model=args.model,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
