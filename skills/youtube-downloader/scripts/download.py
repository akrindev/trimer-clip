#!/usr/bin/env python3
import yt_dlp
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def download_video(
    url: str,
    output_path: Optional[str] = None,
    format: str = "best[ext=mp4]/best",
    quality: Optional[str] = None,
    audio_only: bool = False,
    audio_format: str = "mp3",
    subtitle: bool = False,
    write_description: bool = False,
    write_info_json: bool = False,
) -> Dict[str, Any]:
    """Download video from YouTube URL."""

    if quality:
        quality_formats = {
            "best": "best[ext=mp4]/best",
            "1080": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "720": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "480": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best",
            "360": "bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best",
        }
        format = quality_formats.get(quality, format)

    ydl_opts = {
        "format": format,
        "outtmpl": output_path or "./downloads/%(title)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "retries": 3,
    }

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

    if subtitle:
        ydl_opts.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en", "en-US", "id"],
            }
        )

    if write_description:
        ydl_opts["writedescription"] = True

    if write_info_json:
        ydl_opts["writeinfojson"] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            result = {
                "success": True,
                "title": info.get("title"),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "url": info.get("webpage_url") or url,
                "video_id": info.get("id"),
                "description": info.get("description"),
                "tags": info.get("tags") or [],
            }

            if audio_only:
                # yt-dlp might have changed the extension during post-processing
                audio_path = info.get("_filename")
                result["audio_path"] = str(audio_path) if audio_path else None
            else:
                video_path = info.get("_filename")
                result["video_path"] = str(video_path) if video_path else None

                thumbnail = info.get("thumbnail")
                if thumbnail:
                    result["thumbnail"] = thumbnail

            return result

    except Exception as e:
        return {"success": False, "error": str(e), "url": url}


def main():
    parser = argparse.ArgumentParser(description="Download video from YouTube")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-f", "--format", default="best[ext=mp4]", help="Video format")
    parser.add_argument(
        "--quality", choices=["best", "1080", "720", "480", "360"], help="Quality preset"
    )
    parser.add_argument("--audio-only", action="store_true", help="Extract audio only")
    parser.add_argument(
        "--audio-format", default="mp3", help="Audio format when using --audio-only"
    )
    parser.add_argument("--subtitle", action="store_true", help="Download subtitles")
    parser.add_argument("--write-description", action="store_true", help="Write video description")
    parser.add_argument(
        "--write-info-json", action="store_true", help="Write video metadata as JSON"
    )

    args = parser.parse_args()

    result = download_video(
        url=args.url,
        output_path=args.output,
        format=args.format,
        quality=args.quality,
        audio_only=args.audio_only,
        audio_format=args.audio_format,
        subtitle=args.subtitle,
        write_description=args.write_description,
        write_info_json=args.write_info_json,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
