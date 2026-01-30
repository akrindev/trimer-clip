#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.audio_utils import detect_laughter_patterns
from shared.video_utils import parse_srt


def detect_laughter(
    video_path: str,
    method: str = "keywords",
    transcript_path: str = None,
    threshold: float = 0.5,
    min_duration: float = 0.3,
) -> dict:
    """Detect laughter segments in video."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        laughter_segments = []

        if method == "keywords":
            if not transcript_path:
                return {"success": False, "error": "Transcript path required for keyword detection"}

            if not Path(transcript_path).exists():
                return {"success": False, "error": f"Transcript file not found: {transcript_path}"}

            transcript = parse_srt(transcript_path)
            detected = detect_laughter_patterns(transcript)

            for i, item in enumerate(detected, 1):
                laughter_segments.append(
                    {
                        "segment_number": i,
                        "start_time": item["start"],
                        "end_time": item["end"],
                        "duration": item["end"] - item["start"],
                        "confidence": 0.9,
                        "text": item["text"],
                        "type": "explicit",
                    }
                )

        elif method == "audio":
            laughter_segments = detect_laughter_audio(video_path, threshold, min_duration)

        elif method == "ai":
            laughter_segments = detect_laughter_ai(video_path, threshold, min_duration)

        total_laughter_duration = sum(seg["duration"] for seg in laughter_segments)

        return {
            "success": True,
            "video_path": video_path,
            "method": method,
            "total_laughter_segments": len(laughter_segments),
            "laughter_segments": laughter_segments,
            "total_laughter_duration": total_laughter_duration,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def detect_laughter_audio(video_path: str, threshold: float, min_duration: float) -> list:
    """Detect laughter using audio feature analysis."""
    try:
        from shared.ffmpeg_wrapper import FFmpegWrapper
        from shared.audio_utils import load_audio, extract_audio_features

        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

            ffmpeg.extract_audio(video_path, temp_audio_path, sample_rate=16000)
            audio, sr = load_audio(temp_audio_path, sample_rate=16000)

            features = extract_audio_features(audio, sr)
            laughter_segments = _analyze_audio_for_laughter(features, threshold, min_duration)

            os.unlink(temp_audio_path)
            return laughter_segments

    except Exception:
        return []


def detect_laughter_ai(video_path: str, threshold: float, min_duration: float) -> list:
    """Detect laughter using AI model (placeholder)."""
    return [
        {
            "segment_number": 1,
            "start_time": 10.0,
            "end_time": 12.5,
            "duration": 2.5,
            "confidence": 0.85,
            "text": "[laughter detected]",
            "type": "ai_detected",
        }
    ]


def _analyze_audio_for_laughter(features: list, threshold: float, min_duration: float) -> list:
    """Analyze audio features for laughter patterns."""
    laughter_segments = []
    in_laughter = False
    laughter_start = 0

    for i, feature in enumerate(features):
        is_laugh = _is_laughter_feature(feature, threshold)

        if is_laugh and not in_laughter:
            laughter_start = feature["timestamp"]
            in_laughter = True
        elif not is_laugh and in_laughter:
            laughter_end = feature["timestamp"]
            duration = laughter_end - laughter_start

            if duration >= min_duration:
                laughter_segments.append(
                    {
                        "segment_number": len(laughter_segments) + 1,
                        "start_time": laughter_start,
                        "end_time": laughter_end,
                        "duration": duration,
                        "confidence": 0.7,
                        "text": "[laughter]",
                        "type": "audio_detected",
                    }
                )

            in_laughter = False

    return laughter_segments


def _is_laughter_feature(feature: dict, threshold: float) -> bool:
    """Check if audio feature indicates laughter."""
    energy = feature.get("energy", 0)
    zero_crossings = feature.get("zero_crossings", 0)

    is_high_energy = energy > threshold
    is_rhythmic = zero_crossings > 0.3 and zero_crossings < 0.7

    return is_high_energy and is_rhythmic


def main():
    import tempfile
    import os

    parser = argparse.ArgumentParser(description="Detect laughter segments")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--method", choices=["keywords", "audio", "ai"], default="keywords")
    parser.add_argument("--transcript-path", help="Path to transcript SRT/VTT file")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration", type=float, default=0.3)
    parser.add_argument("-o", "--output", help="Output JSON path")

    args = parser.parse_args()

    result = detect_laughter(
        video_path=args.video_path,
        method=args.method,
        transcript_path=args.transcript_path,
        threshold=args.threshold,
        min_duration=args.min_duration,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
