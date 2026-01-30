#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from video_trimmer.scripts.trim import trim_multiple


def main():
    parser = argparse.ArgumentParser(description="Trim multiple segments")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--segments", required=True, help="JSON file with segments")
    parser.add_argument("--output-dir", default="./clips/", help="Output directory")
    parser.add_argument("--copy", action="store_true", help="Use stream copy")
    parser.add_argument("--prefix", default="clip", help="Filename prefix")

    args = parser.parse_args()

    with open(args.segments, "r") as f:
        segments = json.load(f)

    result = trim_multiple(
        video_path=args.video_path,
        segments=segments,
        output_dir=args.output_dir,
        copy=args.copy,
        prefix=args.prefix,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
