#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper


STYLE_PRESETS = {
    "tiktok": {
        "font": "Plus Jakarta Sans",
        "font_size": 8,  # Smaller for portrait
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 1,
        "position": "bottom",
    },
    "shorts": {
        "font": "Plus Jakarta Sans",
        "font_size": 6,  # Smaller for portrait
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 1,
        "position": "bottom",
    },
    "reels": {
        "font": "Plus Jakarta Sans",
        "font_size": 6,  # Smaller for portrait
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 1,
        "position": "bottom",
    },
    "default": {
        "font": "Plus Jakarta Sans",
        "font_size": 8,  # Smaller for portrait
        "font_color": "white",
        "outline_color": "black",
        "outline_width": 1,
        "position": "bottom",
    },
}


def add_subtitles(
    video_path: str,
    subtitle_path: str,
    output_path: str = None,
    style: str = "default",
    font: str = None,
    font_size: int = None,
    font_color: str = None,
    outline_color: str = None,
    outline_width: int = None,
    position: str = None,
    no_outline: bool = False,
    force_style: bool = True,
) -> dict:
    """Add burned-in subtitles to video."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    if not Path(subtitle_path).exists():
        return {"success": False, "error": f"Subtitle file not found: {subtitle_path}"}

    try:
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["default"])

        final_font = font or preset["font"]
        final_font_size = font_size or preset["font_size"]
        final_font_color = font_color or preset["font_color"]
        final_outline_color = outline_color or preset["outline_color"]
        final_outline_width = 0 if no_outline else (outline_width or preset["outline_width"])
        final_position = position or preset["position"]

        if output_path is None:
            output_path = f"{Path(video_path).stem}_subtitled.mp4"

        ffmpeg = FFmpegWrapper()

        success = ffmpeg.add_subtitle(
            video_path,
            subtitle_path,
            output_path,
            font_name=final_font,
            font_size=final_font_size,
            font_color=final_font_color,
            outline_color=final_outline_color,
            outline_width=final_outline_width,
            position=final_position,
            force_style=force_style,
        )

        if success:
            subtitle_count = count_subtitles(subtitle_path)

            return {
                "success": True,
                "input_video": video_path,
                "input_subtitle": subtitle_path,
                "output_video": output_path,
                "style": {
                    "preset": style,
                    "font": final_font,
                    "font_size": final_font_size,
                    "font_color": final_font_color,
                    "outline_color": final_outline_color,
                    "outline_width": final_outline_width,
                    "position": final_position,
                },
                "subtitle_count": subtitle_count,
            }
        else:
            return {
                "success": False,
                "error": "FFmpeg subtitle overlay failed",
                "input_video": video_path,
            }

    except Exception as e:
        return {"success": False, "error": str(e), "input_video": video_path}


def count_subtitles(subtitle_path: str) -> int:
    """Count number of subtitle entries."""
    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            content = f.read()

        if subtitle_path.endswith(".srt"):
            return content.strip().count("\n\n") + 1
        elif subtitle_path.endswith(".vtt"):
            lines = content.split("\n")
            return sum(1 for line in lines if "-->" in line)
        else:
            return 0
    except:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Add subtitles to video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-s", "--subtitle", required=True, help="Path to subtitle file")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument(
        "--style", choices=["tiktok", "shorts", "reels", "default"], default="default"
    )
    parser.add_argument("--font", help="Font name")
    parser.add_argument("--font-size", type=int, help="Font size")
    parser.add_argument("--font-color", help="Font color")
    parser.add_argument("--outline-color", help="Outline color")
    parser.add_argument("--outline-width", type=int, help="Outline width")
    parser.add_argument("--position", choices=["bottom", "top", "center"], help="Text position")
    parser.add_argument("--no-outline", action="store_true", help="Disable outline")
    parser.add_argument(
        "--use-ass-style",
        action="store_true",
        help="Use styles defined in ASS subtitle file",
    )

    args = parser.parse_args()

    result = add_subtitles(
        video_path=args.video_path,
        subtitle_path=args.subtitle,
        output_path=args.output,
        style=args.style,
        font=args.font,
        font_size=args.font_size,
        font_color=args.font_color,
        outline_color=args.outline_color,
        outline_width=args.outline_width,
        position=args.position,
        no_outline=args.no_outline,
        force_style=not args.use_ass_style,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
