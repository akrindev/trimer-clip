#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper


def extract_speaker_segments(
    video_path: str,
    diarization_path: str,
    speaker: str = None,
    min_segment_duration: float = 5.0,
    context: float = 2.0,
    output_dir: str = "./speaker_segments/",
) -> dict:
    """Extract video segments for specific speakers."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video not found: {video_path}"}

    if not Path(diarization_path).exists():
        return {"success": False, "error": f"Diarization file not found: {diarization_path}"}

    try:
        with open(diarization_path, "r") as f:
            diarization = json.load(f)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        segments = diarization.get("segments", [])
        speakers_to_extract = [speaker] if speaker else list(diarization.get("speakers", {}).keys())

        results = []
        ffmpeg = FFmpegWrapper()

        for target_speaker in speakers_to_extract:
            speaker_segments = [s for s in segments if s["speaker"] == target_speaker]

            if not speaker_segments:
                continue

            valid_segments = [s for s in speaker_segments if s["duration"] >= min_segment_duration]

            if not valid_segments:
                continue

            for i, seg in enumerate(valid_segments):
                start = max(0, seg["start"] - context)
                end = seg["end"] + context

                output_path = Path(output_dir) / f"{target_speaker}_segment_{i + 1:03d}.mp4"

                ffmpeg.trim_video(video_path, str(output_path), start, end, reencode=False)

                results.append(
                    {
                        "speaker": target_speaker,
                        "segment_index": i + 1,
                        "start": start,
                        "end": end,
                        "duration": end - start,
                        "output_path": str(output_path),
                    }
                )

        return {
            "success": True,
            "video_path": video_path,
            "speakers_extracted": speakers_to_extract,
            "total_segments": len(results),
            "output_dir": output_dir,
            "segments": results,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Extract speaker segments")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("diarization_path", help="Path to diarization JSON")
    parser.add_argument("--speaker", help="Specific speaker to extract")
    parser.add_argument("--min-segment-duration", type=float, default=5.0)
    parser.add_argument("--context", type=float, default=2.0)
    parser.add_argument("--output-dir", default="./speaker_segments/")

    args = parser.parse_args()

    result = extract_speaker_segments(
        video_path=args.video_path,
        diarization_path=args.diarization_path,
        speaker=args.speaker,
        min_segment_duration=args.min_segment_duration,
        context=args.context,
        output_dir=args.output_dir,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
