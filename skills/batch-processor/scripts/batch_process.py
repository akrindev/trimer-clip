#!/usr/bin/env python3
import sys
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "skills"))

from autocut_shorts.scripts.autocut import autocut


def batch_process(
    input_file: str,
    output_dir: str = "./batch_output/",
    parallel: int = 1,
    transcription_model: str = "auto",
) -> dict:
    """Process multiple videos in batch."""

    try:
        with open(input_file, "r") as f:
            videos = json.load(f)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d")
        batch_dir = Path(output_dir) / timestamp
        batch_dir.mkdir(parents=True, exist_ok=True)

        results = []
        start_time = time.time()

        if parallel > 1:
            with ProcessPoolExecutor(max_workers=parallel) as executor:
                futures = {}

                for i, video in enumerate(videos):
                    video_dir = batch_dir / f"video_{i + 1:03d}"
                    future = executor.submit(
                        autocut,
                        source=video["source"],
                        source_type=video.get("source_type", "auto"),
                        num_clips=video.get("num_clips", 5),
                        platform=video.get("platform", "tiktok"),
                        output_dir=str(video_dir),
                        transcription_model=transcription_model,
                    )
                    futures[future] = video

                for future in as_completed(futures):
                    video = futures[future]
                    try:
                        result = future.result()
                        results.append(
                            {
                                "source": video["source"],
                                "status": "success" if result["success"] else "failed",
                                "result": result,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {"source": video["source"], "status": "failed", "error": str(e)}
                        )
        else:
            for i, video in enumerate(videos):
                video_dir = batch_dir / f"video_{i + 1:03d}"

                result = autocut(
                    source=video["source"],
                    source_type=video.get("source_type", "auto"),
                    num_clips=video.get("num_clips", 5),
                    platform=video.get("platform", "tiktok"),
                    output_dir=str(video_dir),
                    transcription_model=transcription_model,
                )

                results.append(
                    {
                        "source": video["source"],
                        "status": "success" if result["success"] else "failed",
                        "result": result,
                    }
                )

        total_time = time.time() - start_time
        successful = sum(1 for r in results if r["status"] == "success")
        total_clips = sum(
            r["result"].get("processing", {}).get("num_clips_generated", 0)
            for r in results
            if r["status"] == "success"
        )

        batch_report = {
            "success": True,
            "batch_id": f"batch_{timestamp}_{int(time.time())}",
            "total_videos": len(videos),
            "successful": successful,
            "failed": len(videos) - successful,
            "total_clips": total_clips,
            "output_dir": str(batch_dir),
            "processing_time": round(total_time, 2),
            "results": results,
        }

        with open(batch_dir / "batch_report.json", "w") as f:
            json.dump(batch_report, f, indent=2)

        return batch_report

    except Exception as e:
        return {"success": False, "error": str(e)}


def batch_from_urls(
    urls_file: str,
    num_clips: int = 5,
    platform: str = "tiktok",
    output_dir: str = "./batch_output/",
    **kwargs,
) -> dict:
    """Process videos from URLs file."""

    try:
        with open(urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        videos = [
            {"source": url, "source_type": "youtube", "num_clips": num_clips, "platform": platform}
            for url in urls
        ]

        temp_file = Path(output_dir) / "temp_batch.json"
        with open(temp_file, "w") as f:
            json.dump(videos, f)

        result = batch_process(str(temp_file), output_dir, **kwargs)

        temp_file.unlink()

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch process videos")
    parser.add_argument("--input", help="JSON file with video list")
    parser.add_argument("--output-dir", default="./batch_output/")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--transcription-model", choices=["auto", "whisper", "gemini"], default="auto"
    )

    args = parser.parse_args()

    if args.input:
        result = batch_process(
            input_file=args.input,
            output_dir=args.output_dir,
            parallel=args.parallel,
            transcription_model=args.transcription_model,
        )
    else:
        result = {"success": False, "error": "No input file specified"}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
