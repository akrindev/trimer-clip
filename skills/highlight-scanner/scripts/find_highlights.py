#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.video_utils import parse_srt, clip_segment


DEFAULT_WEIGHTS = {"transcript": 0.35, "laughter": 0.25, "sentiment": 0.25, "scenes": 0.15}

VIRAL_KEYWORDS = [
    "you won't believe",
    "this changes everything",
    "the secret to",
    "what nobody tells you",
    "i made a huge mistake",
    "this is illegal",
    "the plot twist",
    "and then it happened",
    "but here's the catch",
    "the most important part",
    "wait for it",
    "watch till the end",
    "comment if you agree",
    "like if you've",
    "this is crazy",
    "mind blown",
    "insane",
    "unbelievable",
    "shocking",
    "epic",
    "viral",
    "trending",
    "must watch",
    "game changer",
]

HOOK_PATTERNS = [
    "let me tell you",
    "here's the thing",
    "the truth about",
    "what i learned",
    "i discovered",
    "the problem with",
    "why you should",
    "never do this",
    "always remember",
    "the biggest mistake",
    "how to",
    "tips for",
    "tricks to",
]


def find_highlights(
    video_path: str,
    transcript_path: str = None,
    scenes_path: str = None,
    laughter_path: str = None,
    sentiment_path: str = None,
    num_clips: int = 5,
    min_duration: float = 15,
    max_duration: float = 60,
    weights: dict = None,
) -> dict:
    """Find viral-worthy highlight segments."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    if weights is None:
        weights = DEFAULT_WEIGHTS

    try:
        segments = []

        if transcript_path:
            segments = analyze_transcript(transcript_path, segments)

        if scenes_path:
            segments = integrate_scenes(scenes_path, segments)

        if laughter_path:
            segments = integrate_laughter(laughter_path, segments)

        if sentiment_path:
            segments = integrate_sentiment(sentiment_path, segments)

        segments = calculate_virality_scores(segments, weights)
        segments = filter_by_duration(segments, min_duration, max_duration)
        segments = rank_and_select(segments, num_clips)

        highlights = format_highlights(segments)

        return {
            "success": True,
            "video_path": video_path,
            "total_segments_analyzed": len(segments),
            "num_clips_requested": num_clips,
            "highlights": highlights,
            "analysis_summary": {
                "avg_virality_score": sum(s.get("virality_score", 0) for s in highlights)
                / len(highlights)
                if highlights
                else 0,
                "total_highlight_duration": sum(h["duration"] for h in highlights),
                "best_segment_start": highlights[0]["start_time"] if highlights else 0,
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def analyze_transcript(transcript_path: str, existing_segments: list) -> list:
    """Analyze transcript for viral content."""
    transcript = parse_srt(transcript_path)

    segments = existing_segments.copy()

    for item in transcript:
        text = item["text"].lower()
        start = item["start"]
        end = item["end"]

        score = 0.0
        matched_keywords = []

        for keyword in VIRAL_KEYWORDS:
            if keyword in text:
                score += 0.2
                matched_keywords.append(keyword)

        for pattern in HOOK_PATTERNS:
            if pattern in text:
                score += 0.15
                matched_keywords.append(pattern)

        if score > 0:
            existing = next((s for s in segments if abs(s["start"] - start) < 2), None)

            if existing:
                existing["transcript_score"] = max(
                    existing.get("transcript_score", 0), min(score, 1.0)
                )
                existing["keywords"].extend(matched_keywords)
            else:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "transcript_score": min(score, 1.0),
                        "keywords": matched_keywords,
                        "text": item["text"],
                        "laughter_score": 0,
                        "sentiment_score": 0,
                        "scene_score": 0,
                    }
                )

    return segments


def integrate_scenes(scenes_path: str, segments: list) -> list:
    """Integrate scene detection data."""
    try:
        with open(scenes_path, "r") as f:
            scenes_data = json.load(f)

        scenes = scenes_data.get("scenes", [])

        for scene in scenes:
            start = scene["start_time"]
            end = scene["end_time"]

            existing = next((s for s in segments if abs(s["start"] - start) < 2), None)

            if existing:
                existing["scene_score"] = max(existing.get("scene_score", 0), 0.5)
                if existing["end"] < end:
                    existing["end"] = end
            else:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "scene_score": 0.5,
                        "transcript_score": 0,
                        "laughter_score": 0,
                        "sentiment_score": 0,
                        "keywords": ["scene_change"],
                        "text": "[Scene change]",
                    }
                )

        return segments

    except Exception:
        return segments


def integrate_laughter(laughter_path: str, segments: list) -> list:
    """Integrate laughter detection data."""
    try:
        with open(laughter_path, "r") as f:
            laughter_data = json.load(f)

        laughter_segments = laughter_data.get("laughter_segments", [])

        for laugh in laughter_segments:
            start = laugh["start_time"]
            end = laugh["end_time"]
            confidence = laugh.get("confidence", 0.7)

            existing = next((s for s in segments if abs(s["start"] - start) < 2), None)

            if existing:
                existing["laughter_score"] = max(existing.get("laughter_score", 0), confidence)
                if existing["end"] < end:
                    existing["end"] = end
            else:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "laughter_score": confidence,
                        "transcript_score": 0,
                        "sentiment_score": 0,
                        "scene_score": 0,
                        "keywords": ["laughter"],
                        "text": laugh.get("text", "[Laughter]"),
                    }
                )

        return segments

    except Exception:
        return segments


def integrate_sentiment(sentiment_path: str, segments: list) -> list:
    """Integrate sentiment analysis data."""
    try:
        with open(sentiment_path, "r") as f:
            sentiment_data = json.load(f)

        peaks = sentiment_data.get("emotional_peaks", [])

        for peak in peaks:
            start = peak["timestamp"]
            end = start + peak.get("duration", 3)
            intensity = peak.get("intensity", 0.5)

            existing = next((s for s in segments if abs(s["start"] - start) < 2), None)

            if existing:
                existing["sentiment_score"] = max(existing.get("sentiment_score", 0), intensity)
                if existing["end"] < end:
                    existing["end"] = end
            else:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "sentiment_score": intensity,
                        "transcript_score": 0,
                        "laughter_score": 0,
                        "scene_score": 0,
                        "keywords": [peak.get("emotion", "emotion")],
                        "text": peak.get("text", "[Emotional peak]"),
                    }
                )

        return segments

    except Exception:
        return segments


def calculate_virality_scores(segments: list, weights: dict) -> list:
    """Calculate final virality scores."""
    for seg in segments:
        transcript = seg.get("transcript_score", 0)
        laughter = seg.get("laughter_score", 0)
        sentiment = seg.get("sentiment_score", 0)
        scenes = seg.get("scene_score", 0)

        virality = (
            transcript * weights.get("transcript", 0.35)
            + laughter * weights.get("laughter", 0.25)
            + sentiment * weights.get("sentiment", 0.25)
            + scenes * weights.get("scenes", 0.15)
        )

        seg["virality_score"] = round(virality, 3)

    return segments


def filter_by_duration(segments: list, min_duration: float, max_duration: float) -> list:
    """Filter segments by duration constraints."""
    filtered = []

    for seg in segments:
        duration = seg["end"] - seg["start"]

        if min_duration <= duration <= max_duration:
            filtered.append(seg)
        elif duration > max_duration:
            seg["end"] = seg["start"] + max_duration
            filtered.append(seg)

    return filtered


def rank_and_select(segments: list, num_clips: int) -> list:
    """Rank and select top segments."""
    segments.sort(key=lambda x: x["virality_score"], reverse=True)

    selected = []
    used_ranges = []

    for seg in segments:
        if len(selected) >= num_clips:
            break

        overlaps = False
        for used in used_ranges:
            if seg["start"] < used["end"] and seg["end"] > used["start"]:
                overlap_duration = min(seg["end"], used["end"]) - max(seg["start"], used["start"])
                if overlap_duration > 5:
                    overlaps = True
                    break

        if not overlaps:
            selected.append(seg)
            used_ranges.append({"start": seg["start"], "end": seg["end"]})

    return selected


def format_highlights(segments: list) -> list:
    """Format segments as highlights output."""
    highlights = []

    for i, seg in enumerate(segments, 1):
        duration = seg["end"] - seg["start"]

        reasoning_parts = []
        if seg.get("transcript_score", 0) > 0.5:
            reasoning_parts.append("hook/viral content")
        if seg.get("laughter_score", 0) > 0.5:
            reasoning_parts.append("humor")
        if seg.get("sentiment_score", 0) > 0.5:
            reasoning_parts.append("strong emotion")
        if seg.get("scene_score", 0) > 0.3:
            reasoning_parts.append("scene transition")

        reasoning = " + ".join(reasoning_parts) if reasoning_parts else "multiple factors"

        confidence = (
            "high"
            if seg["virality_score"] > 0.7
            else "medium"
            if seg["virality_score"] > 0.4
            else "low"
        )

        highlights.append(
            {
                "rank": i,
                "start_time": round(seg["start"], 2),
                "end_time": round(seg["end"], 2),
                "duration": round(duration, 2),
                "virality_score": seg["virality_score"],
                "scores": {
                    "transcript": round(seg.get("transcript_score", 0), 2),
                    "laughter": round(seg.get("laughter_score", 0), 2),
                    "sentiment": round(seg.get("sentiment_score", 0), 2),
                    "scenes": round(seg.get("scene_score", 0), 2),
                },
                "text": seg.get("text", "")[:200],
                "reasoning": f"Contains {reasoning}",
                "suggested_clip_start": round(max(0, seg["start"] - 2), 2),
                "suggested_clip_end": round(seg["end"] + 2, 2),
                "confidence": confidence,
            }
        )

    return highlights


def main():
    parser = argparse.ArgumentParser(description="Find viral-worthy highlights")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--transcript-path", help="Path to transcript SRT/VTT")
    parser.add_argument("--scenes-path", help="Path to scenes JSON")
    parser.add_argument("--laughter-path", help="Path to laughter JSON")
    parser.add_argument("--sentiment-path", help="Path to sentiment JSON")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=15)
    parser.add_argument("--max-duration", type=float, default=60)
    parser.add_argument("-o", "--output", help="Output JSON path")

    args = parser.parse_args()

    result = find_highlights(
        video_path=args.video_path,
        transcript_path=args.transcript_path,
        scenes_path=args.scenes_path,
        laughter_path=args.laughter_path,
        sentiment_path=args.sentiment_path,
        num_clips=args.num_clips,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
