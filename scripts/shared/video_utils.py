from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import re


def parse_srt_time(srt_time: str) -> float:
    """Convert SRT timestamp to seconds."""
    pattern = r"(\d+):(\d+):(\d+),(\d+)"
    match = re.match(pattern, srt_time)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    return 0.0


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(srt_path: str) -> List[Dict[str, Any]]:
    """Parse SRT subtitle file."""
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    subtitles = []

    for block in blocks:
        lines = block.split("\n")
        if len(lines) >= 3:
            try:
                idx = int(lines[0])
                time_line = lines[1]
                text = "\n".join(lines[2:])

                start_end = time_line.split(" --> ")
                start_time = parse_srt_time(start_end[0])
                end_time = parse_srt_time(start_end[1])

                subtitles.append({"index": idx, "start": start_time, "end": end_time, "text": text})
            except (ValueError, IndexError):
                continue

    return subtitles


def create_srt(subtitles: List[Dict[str, Any]], output_path: str) -> None:
    """Create SRT subtitle file from subtitle list."""
    with open(output_path, "w", encoding="utf-8") as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{format_srt_time(sub['start'])} --> {format_srt_time(sub['end'])}\n")
            f.write(f"{sub['text']}\n\n")


def time_to_seconds(time_str: str) -> float:
    """Convert various time formats to seconds."""
    if ":" in time_str:
        parts = list(map(float, time_str.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
    return float(time_str)


def find_overlapping_segments(
    segments: List[Dict[str, float]], max_gap: float = 2.0
) -> List[Dict[str, float]]:
    """Merge overlapping or nearby segments."""
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda x: x["start"])
    merged = [sorted_segments[0].copy()]

    for seg in sorted_segments[1:]:
        last = merged[-1]
        if seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            merged.append(seg.copy())

    return merged


def clip_segment(segment: Dict[str, float], min_duration: float, max_duration: float) -> bool:
    """Check if segment is within duration bounds."""
    duration = segment["end"] - segment["start"]
    return min_duration <= duration <= max_duration


def split_long_segment(segment: Dict[str, float], max_duration: float) -> List[Dict[str, float]]:
    """Split long segment into multiple chunks."""
    duration = segment["end"] - segment["start"]
    if duration <= max_duration:
        return [segment.copy()]

    chunks = []
    current_start = segment["start"]

    while current_start < segment["end"]:
        current_end = min(current_start + max_duration, segment["end"])
        chunks.append(
            {"start": current_start, "end": current_end, "score": segment.get("score", 0)}
        )
        current_start = current_end

    return chunks


def calculate_overlap(seg1: Dict[str, float], seg2: Dict[str, float]) -> float:
    """Calculate overlap duration between two segments."""
    overlap_start = max(seg1["start"], seg2["start"])
    overlap_end = min(seg1["end"], seg2["end"])

    if overlap_start >= overlap_end:
        return 0.0

    return overlap_end - overlap_start


def combine_scores(
    segments: List[Dict[str, float]], weights: Dict[str, float]
) -> List[Dict[str, float]]:
    """Combine multiple score sources into final score."""
    for seg in segments:
        final_score = 0
        for key, weight in weights.items():
            if key in seg:
                final_score += seg[key] * weight
        seg["final_score"] = final_score

    return segments


def validate_video_file(video_path: str) -> bool:
    """Check if video file is valid."""
    path = Path(video_path)
    return path.exists() and path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def get_platform_presets(platform: str) -> Dict[str, Any]:
    """Get platform-specific video presets."""
    presets = {
        "tiktok": {
            "width": 1080,
            "height": 1920,
            "duration_range": (15, 60),
            "fps": 30,
            "bitrate": "4000k",
            "audio_bitrate": "128k",
        },
        "youtube_shorts": {
            "width": 1080,
            "height": 1920,
            "duration_range": (15, 60),
            "fps": 30,
            "bitrate": "4000k",
            "audio_bitrate": "128k",
        },
        "instagram_reels": {
            "width": 1080,
            "height": 1920,
            "duration_range": (15, 90),
            "fps": 30,
            "bitrate": "4000k",
            "audio_bitrate": "128k",
        },
        "facebook_reels": {
            "width": 1080,
            "height": 1920,
            "duration_range": (15, 90),
            "fps": 30,
            "bitrate": "4000k",
            "audio_bitrate": "128k",
        },
    }
    return presets.get(platform.lower(), presets["tiktok"])


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for file system."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename[:200]


def generate_output_filename(
    original_name: str, segment_index: int, start_time: float, platform: str = "tiktok"
) -> str:
    """Generate output filename for clip."""
    base_name = Path(original_name).stem
    sanitized = sanitize_filename(base_name)
    timestamp_str = f"{int(start_time // 60):02d}_{int(start_time % 60):02d}"
    return f"{sanitized}_{platform}_{segment_index + 1}_{timestamp_str}.mp4"
