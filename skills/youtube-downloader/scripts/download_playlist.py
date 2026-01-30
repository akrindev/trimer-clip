#!/usr/bin/env python3
import yt_dlp
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def download_playlist(
    url: str,
    output_path: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    format: str = "best[ext=mp4]/best",
    audio_only: bool = False,
    audio_format: str = "mp3",
) -> Dict[str, Any]:
    """Download entire YouTube playlist."""

    ydl_opts = {
        "format": format,
        "outtmpl": output_path
        or "./downloads/%(playlist_title)s/%(playlist_index)03d-%(title)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "retries": 3,
        "ignoreerrors": True,
    }

    if start is not None:
        ydl_opts["playliststart"] = start
    if end is not None:
        ydl_opts["playlistend"] = end

    if audio_only:
        ydl_opts.update(
            {
                "format": f"bestaudio[ext={audio_format}]/bestaudio",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": audio_format,
                    }
                ],
            }
        )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            result = {
                "success": True,
                "playlist_title": info.get("title"),
                "playlist_id": info.get("id"),
                "playlist_url": url,
                "total_videos": len(info.get("entries", [])),
                "downloaded_videos": sum(1 for e in info.get("entries", []) if e),
                "entries": [],
            }

            for entry in info.get("entries", []):
                if entry:
                    result["entries"].append(
                        {
                            "index": entry.get("playlist_index"),
                            "title": entry.get("title"),
                            "url": entry.get("url"),
                            "duration": entry.get("duration"),
                            "success": True,
                        }
                    )

            return result

    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


def main():
    parser = argparse.ArgumentParser(description="Download YouTube playlist")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("--start", type=int, help="Start downloading from this video number")
    parser.add_argument("--end", type=int, help="End downloading at this video number")
    parser.add_argument("-f", "--format", default="best[ext=mp4]", help="Video format")
    parser.add_argument("--audio-only", action="store_true", help="Extract audio only")
    parser.add_argument(
        "--audio-format", default="mp3", help="Audio format when using --audio-only"
    )

    args = parser.parse_args()

    result = download_playlist(
        url=args.url,
        output_path=args.output,
        start=args.start,
        end=args.end,
        format=args.format,
        audio_only=args.audio_only,
        audio_format=args.audio_format,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
