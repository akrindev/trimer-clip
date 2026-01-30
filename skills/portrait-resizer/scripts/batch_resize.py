#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from portrait_resizer.scripts.resize_to_portrait import batch_resize


def main():
    parser = argparse.ArgumentParser(description="Batch resize videos to portrait")
    parser.add_argument("--input-dir", required=True, help="Directory with videos")
    parser.add_argument("--output-dir", default="./portrait/")
    parser.add_argument("--mode", choices=["smart", "center", "letterbox"], default="smart")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--quality", default="fast")

    args = parser.parse_args()

    result = batch_resize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        width=args.width,
        height=args.height,
        quality=args.quality,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
