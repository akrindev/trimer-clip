#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

try:
    from scenedetect import detect, ContentDetector, split_video_ffmpeg

    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

from shared.ffmpeg_wrapper import FFmpegWrapper


def detect_scenes(
    video_path: str,
    threshold: float = 0.3,
    min_scene_len: float = 0.5,
    split: bool = False,
    output_dir: str = "./clips/",
) -> dict:
    """Detect scene changes in video."""

    if not SCENEDETECT_AVAILABLE:
        return {
            "success": False,
            "error": "PySceneDetect is not installed. Install with: pip install scenedetect[opencv]",
        }

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        ffmpeg = FFmpegWrapper()
        duration = ffmpeg.get_duration(video_path)

        scene_list = detect(
            video_path, ContentDetector(threshold=threshold, min_scene_len=int(min_scene_len * 30))
        )

        scenes = []
        for i, scene in enumerate(scene_list, 1):
            # scene is a tuple of (start_timecode, end_timecode)
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scenes.append(
                {
                    "scene_number": i,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "content_type": "adaptive",
                }
            )

        result = {
            "success": True,
            "video_path": video_path,
            "duration": duration,
            "total_scenes": len(scenes),
            "scenes": scenes,
        }

        if split:
            output_path = split_video_by_scenes(video_path, scenes, output_dir)
            result["split_clips"] = output_path

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def split_video_by_scenes(video_path: str, scenes: list, output_dir: str) -> str:
    """Split video into scene clips."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_name = Path(video_path).stem
    clip_paths = []

    for scene in scenes:
        start = scene["start_time"]
        end = scene["end_time"]
        scene_num = scene["scene_number"]

        clip_name = f"{base_name}_scene_{scene_num:03d}.mp4"
        clip_path = Path(output_dir) / clip_name

        ffmpeg = FFmpegWrapper()
        ffmpeg.trim_video(video_path, str(clip_path), start, end, reencode=False)

        clip_paths.append(str(clip_path))

    return str(Path(output_dir).absolute())


def main():
    parser = argparse.ArgumentParser(description="Detect scene changes in video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min-scene-len", type=float, default=0.5)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--output-dir", default="./clips/")

    args = parser.parse_args()

    result = detect_scenes(
        video_path=args.video_path,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
        split=args.split,
        output_dir=args.output_dir,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
