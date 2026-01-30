#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper
from shared.video_utils import time_to_seconds


def trim_video(
    video_path: str,
    start_time: str,
    end_time: str,
    output_path: str = None,
    reencode: bool = False,
    codec: str = "libx264",
    quality: str = "fast",
) -> dict:
    """Trim video to specified time range."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        ffmpeg = FFmpegWrapper()

        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)

        duration = ffmpeg.get_duration(video_path)

        start_sec = max(0, min(start_sec, duration))
        end_sec = max(start_sec, min(end_sec, duration))

        if output_path is None:
            base_name = Path(video_path).stem
            output_path = f"{base_name}_trimmed.mp4"

        success = ffmpeg.trim_video(
            video_path,
            output_path,
            start_sec,
            end_sec,
            reencode=reencode,
            vcodec=codec if reencode else None,
        )

        if success:
            return {
                "success": True,
                "input_path": video_path,
                "output_path": output_path,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": end_sec - start_sec,
                "mode": "reencode" if reencode else "stream_copy",
            }
        else:
            return {"success": False, "error": "FFmpeg trim failed", "input_path": video_path}

    except Exception as e:
        return {"success": False, "error": str(e), "input_path": video_path}


def trim_multiple(
    video_path: str,
    segments: list,
    output_dir: str = "./clips/",
    copy: bool = True,
    prefix: str = "clip",
) -> dict:
    """Trim multiple segments from video."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []

        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", start + 30)

            output_path = Path(output_dir) / f"{prefix}_{i:03d}.mp4"

            result = trim_video(
                video_path, str(start), str(end), str(output_path), reencode=not copy
            )

            results.append(result)

        successful = sum(1 for r in results if r["success"])

        return {
            "success": successful == len(results),
            "total_segments": len(segments),
            "successful_segments": successful,
            "output_dir": output_dir,
            "results": results,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "input_path": video_path}


def main():
    parser = argparse.ArgumentParser(description="Trim video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-s", "--start", required=True, help="Start time (seconds or HH:MM:SS)")
    parser.add_argument("-e", "--end", required=True, help="End time (seconds or HH:MM:SS)")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--reencode", action="store_true", help="Re-encode video")
    parser.add_argument("--codec", default="libx264", help="Video codec for re-encoding")
    parser.add_argument("--quality", default="fast", help="Quality preset")
    parser.add_argument("--copy", action="store_true", help="Use stream copy")

    args = parser.parse_args()

    result = trim_video(
        video_path=args.video_path,
        start_time=args.start,
        end_time=args.end,
        output_path=args.output,
        reencode=args.reencode,
        codec=args.codec,
        quality=args.quality,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
