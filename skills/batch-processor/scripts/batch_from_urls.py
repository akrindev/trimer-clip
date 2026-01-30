#!/usr/bin/env python3
import sys
import argparse

from batch_process import batch_from_urls


def main():
    parser = argparse.ArgumentParser(description="Batch process from URLs")
    parser.add_argument("--urls", required=True, help="File with YouTube URLs")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument(
        "--platform", choices=["tiktok", "shorts", "reels", "facebook"], default="tiktok"
    )
    parser.add_argument("--output-dir", default="./batch_output/")
    parser.add_argument("--parallel", type=int, default=1)

    args = parser.parse_args()

    result = batch_from_urls(
        urls_file=args.urls,
        num_clips=args.num_clips,
        platform=args.platform,
        output_dir=args.output_dir,
        parallel=args.parallel,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
