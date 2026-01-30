#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper


def resize_to_portrait(
    video_path: str,
    width: int = 1080,
    height: int = 1920,
    mode: str = "smart",
    focus_x: float = None,
    focus_y: float = None,
    output_path: str = None,
    quality: str = "fast",
) -> dict:
    """Convert video to 9:16 portrait format."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        ffmpeg = FFmpegWrapper()

        input_width, input_height = ffmpeg.get_resolution(video_path)

        if input_height > input_width:
            return {
                "success": False,
                "error": "Video is already in portrait format",
                "input_path": video_path,
                "input_resolution": {"width": input_width, "height": input_height},
            }

        if mode == "smart":
            if focus_x is None or focus_y is None:
                focus_x, focus_y = detect_focus_point(video_path)

            success = ffmpeg.crop_to_portrait(
                video_path,
                output_path or f"{Path(video_path).stem}_portrait.mp4",
                width,
                height,
                focus_point=(focus_x * input_width, focus_y * input_height),
            )

        elif mode == "center":
            success = ffmpeg.crop_to_portrait(
                video_path,
                output_path or f"{Path(video_path).stem}_portrait.mp4",
                width,
                height,
                focus_point=(input_width / 2, input_height / 2),
            )

        elif mode == "letterbox":
            success = ffmpeg.resize_video(
                video_path,
                output_path or f"{Path(video_path).stem}_portrait.mp4",
                width,
                height,
                crop_mode="center",
                fill_color="black",
            )

        else:
            return {"success": False, "error": f"Unknown mode: {mode}"}

        if success:
            return {
                "success": True,
                "input_path": video_path,
                "output_path": output_path or f"{Path(video_path).stem}_portrait.mp4",
                "input_resolution": {"width": input_width, "height": input_height},
                "output_resolution": {"width": width, "height": height},
                "mode": mode,
                "focus_point": {"x": focus_x, "y": focus_y} if focus_x else None,
            }
        else:
            return {"success": False, "error": "FFmpeg resize failed", "input_path": video_path}

    except Exception as e:
        return {"success": False, "error": str(e), "input_path": video_path}


def detect_focus_point(video_path: str) -> tuple:
    """Detect focus point using simple heuristics."""
    try:
        ffmpeg = FFmpegWrapper()
        width, height = ffmpeg.get_resolution(video_path)

        center_x = width / 2
        center_y = height / 2

        focus_x = center_x / width
        focus_y = center_y / height

        return focus_x, focus_y

    except Exception:
        return 0.5, 0.5


def batch_resize(input_dir: str, output_dir: str = "./portrait/", **kwargs) -> dict:
    """Resize multiple videos to portrait format."""

    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        video_files = list(Path(input_dir).glob("*.mp4"))
        video_files.extend(Path(input_dir).glob("*.mov"))
        video_files.extend(Path(input_dir).glob("*.avi"))

        results = []

        for video_file in video_files:
            output_path = Path(output_dir) / f"{video_file.stem}_portrait.mp4"

            result = resize_to_portrait(str(video_file), output_path=str(output_path), **kwargs)

            results.append(result)

        successful = sum(1 for r in results if r["success"])

        return {
            "success": successful == len(results),
            "total_videos": len(video_files),
            "successful_resizes": successful,
            "output_dir": output_dir,
            "results": results,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Resize video to portrait format")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--mode", choices=["smart", "center", "letterbox"], default="smart")
    parser.add_argument("--focus-x", type=float)
    parser.add_argument("--focus-y", type=float)
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--quality", default="fast")

    args = parser.parse_args()

    result = resize_to_portrait(
        video_path=args.video_path,
        width=args.width,
        height=args.height,
        mode=args.mode,
        focus_x=args.focus_x,
        focus_y=args.focus_y,
        output_path=args.output,
        quality=args.quality,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
