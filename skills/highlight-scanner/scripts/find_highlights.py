#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.video_utils import parse_srt


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


def _clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _generate_title_and_hook(text: str, rank: int) -> tuple:
    clean = _clean_text(text)
    if not clean:
        return f"Clip {rank}: Highlight utama", "Bagian paling menarik dari pembahasan ini"

    first_sentence = clean.split(". ")[0].strip(" .")
    if len(first_sentence) > 70:
        first_sentence = first_sentence[:67].rstrip() + "..."

    title = f"Clip {rank}: {first_sentence}" if first_sentence else f"Clip {rank}: Highlight utama"

    words = clean.split()
    hook = " ".join(words[:15]).strip(" .")
    if len(words) > 15:
        hook += "..."

    if not hook:
        hook = "Bagian paling menarik dari pembahasan ini"

    return title, hook


def _strip_code_fence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_payload(payload: str):
    cleaned = _strip_code_fence(payload)
    if not cleaned:
        raise ValueError("Empty AI response")

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        return json.loads(snippet)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        parsed = json.loads(snippet)
        if isinstance(parsed, dict) and isinstance(parsed.get("highlights"), list):
            return parsed["highlights"]
        raise ValueError("AI JSON object missing 'highlights' list")

    raise ValueError("Could not parse AI JSON response")


def _parse_time_value(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError("Invalid timestamp")

    raw = value.strip().replace(",", ".")
    parts = raw.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m = float(parts[0])
            s = float(parts[1])
            return m * 60 + s
        return float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc


def _build_ai_transcript_lines(transcript: list, max_chars: int = 60000) -> str:
    lines = []
    size = 0
    for item in transcript:
        line = f"[{item['start']:.2f}-{item['end']:.2f}] {item['text']}"
        next_size = size + len(line) + 1
        if next_size > max_chars:
            break
        lines.append(line)
        size = next_size
    return "\n".join(lines)


def _slice_transcript_text(transcript: list, start: float, end: float) -> str:
    parts = []
    for item in transcript:
        if item["end"] >= start and item["start"] <= end:
            parts.append(item["text"])
    return _clean_text(" ".join(parts))[:240]


def _call_openai_highlights(prompt: str, api_key: str, model: str) -> list:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    content = response.choices[0].message.content if response.choices else ""
    return _parse_json_payload(content)


def _call_gemini_highlights(prompt: str, api_key: str, model: str) -> list:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    requested = _clean_text(model) or "gemini-flash-lite-latest"
    if requested.startswith("models/"):
        requested = requested.split("/", 1)[1]

    candidates = [requested]
    prefixed = f"models/{requested}"
    if prefixed != requested:
        candidates.append(prefixed)

    last_error = None
    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4,
                ),
            )
            return _parse_json_payload(getattr(response, "text", ""))
        except Exception as exc:
            last_error = exc

    raise RuntimeError(str(last_error))


def _normalize_ai_highlights(
    ai_items: list, transcript: list, num_clips: int, min_duration: float, max_duration: float
) -> list:
    normalized = []

    for idx, item in enumerate(ai_items or [], 1):
        try:
            start_raw = item.get("start")
            end_raw = item.get("end")
            if start_raw is None:
                start_raw = item.get("start_time")
            if end_raw is None:
                end_raw = item.get("end_time")

            start = _parse_time_value(start_raw)
            end = _parse_time_value(end_raw)
            if end <= start:
                continue

            duration = end - start
            if duration < min_duration:
                continue
            if duration > max_duration:
                end = start + max_duration

            title = _clean_text(item.get("title"))
            hook_text = _clean_text(item.get("hook_text"))
            reasoning = _clean_text(item.get("reason") or item.get("reasoning"))
            text = _slice_transcript_text(transcript, start, end)

            if not title or not hook_text:
                fallback_title, fallback_hook = _generate_title_and_hook(text, idx)
                title = title or fallback_title
                hook_text = hook_text or fallback_hook

            normalized.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "transcript_score": 1.0,
                    "laughter_score": 0,
                    "sentiment_score": 0,
                    "scene_score": 0,
                    "keywords": ["ai_highlight"],
                    "text": text,
                    "title": title,
                    "hook_text": hook_text,
                    "reasoning": reasoning or "AI-selected highlight",
                    "virality_score": max(0.7, 1.0 - (idx - 1) * 0.03),
                }
            )
        except Exception:
            continue

    normalized = filter_by_duration(normalized, min_duration, max_duration)
    normalized = rank_and_select(normalized, num_clips)
    return normalized


def find_highlights_with_ai(
    transcript_path: str,
    num_clips: int,
    min_duration: float,
    max_duration: float,
    provider: str,
    ai_model: str,
    openai_api_key: str,
    gemini_api_key: str,
) -> list:
    transcript = parse_srt(transcript_path)
    if not transcript:
        return []

    transcript_lines = _build_ai_transcript_lines(transcript)
    prompt = f"""
Kamu adalah editor short-form video. Pilih {num_clips} highlight terbaik dari transcript.

Aturan wajib:
- Durasi setiap highlight harus {int(min_duration)}-{int(max_duration)} detik.
- Output harus JSON array valid.
- Setiap item wajib punya: start_time, end_time, title, hook_text, reason.
- hook_text maksimal 15 kata, bahasa Indonesia santai, tanpa emoji.
- title harus catchy dan relevan dengan isi clip.

Format output:
[
  {{
    "start_time": 12.3,
    "end_time": 66.8,
    "title": "...",
    "hook_text": "...",
    "reason": "..."
  }}
]

Transcript:
{transcript_lines}
"""

    provider = (provider or "").lower()
    if provider == "openai" and openai_api_key:
        items = _call_openai_highlights(prompt, openai_api_key, ai_model or "gpt-4o-mini")
    elif provider == "gemini" and gemini_api_key:
        items = _call_gemini_highlights(
            prompt, gemini_api_key, ai_model or "gemini-flash-lite-latest"
        )
    else:
        return []

    return _normalize_ai_highlights(items, transcript, num_clips, min_duration, max_duration)


def find_highlights(
    video_path: str,
    transcript_path: str = None,
    scenes_path: str = None,
    laughter_path: str = None,
    sentiment_path: str = None,
    num_clips: int = 5,
    min_duration: float = 40,
    max_duration: float = 120,
    weights: dict = None,
    highlight_model: str = "heuristic",
    highlight_ai_model: str = None,
    openai_api_key: str = None,
    gemini_api_key: str = None,
) -> dict:
    """Find viral-worthy highlight segments."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    if weights is None:
        weights = DEFAULT_WEIGHTS

    try:
        if highlight_model in ["openai", "gemini", "auto"] and transcript_path:
            provider = highlight_model
            if provider == "auto":
                provider = "gemini" if gemini_api_key else "openai"

            ai_highlights = find_highlights_with_ai(
                transcript_path=transcript_path,
                num_clips=num_clips,
                min_duration=min_duration,
                max_duration=max_duration,
                provider=provider,
                ai_model=highlight_ai_model,
                openai_api_key=openai_api_key,
                gemini_api_key=gemini_api_key,
            )

            if ai_highlights:
                highlights = format_highlights(ai_highlights)
                return {
                    "success": True,
                    "video_path": video_path,
                    "total_segments_analyzed": len(ai_highlights),
                    "num_clips_requested": num_clips,
                    "highlights": highlights,
                    "analysis_summary": {
                        "avg_virality_score": sum(s.get("virality_score", 0) for s in highlights)
                        / len(highlights)
                        if highlights
                        else 0,
                        "total_highlight_duration": sum(h["duration"] for h in highlights),
                        "best_segment_start": highlights[0]["start_time"] if highlights else 0,
                        "method": f"ai:{provider}",
                    },
                }

        segments = []

        if transcript_path:
            segments = analyze_transcript(transcript_path, segments)

        if scenes_path:
            segments = integrate_scenes(scenes_path, segments)

        if laughter_path:
            segments = integrate_laughter(laughter_path, segments)

        if sentiment_path:
            segments = integrate_sentiment(sentiment_path, segments)

        if not segments and transcript_path:
            segments = fallback_segments_from_transcript(
                transcript_path, min_duration, max_duration
            )

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


def fallback_segments_from_transcript(
    transcript_path: str, min_duration: float, max_duration: float
) -> list:
    """Fallback: build segments from transcript timing when no keywords match."""
    transcript = parse_srt(transcript_path)
    if not transcript:
        return []

    segments = []
    idx = 0
    total = len(transcript)

    while idx < total:
        start = transcript[idx]["start"]
        end = transcript[idx]["end"]
        text_parts = [transcript[idx]["text"]]
        idx += 1

        while idx < total and end - start < min_duration:
            end = transcript[idx]["end"]
            text_parts.append(transcript[idx]["text"])
            idx += 1

        if end - start < min_duration:
            break

        if end - start > max_duration:
            end = start + max_duration

        segments.append(
            {
                "start": start,
                "end": end,
                "transcript_score": 0,
                "laughter_score": 0,
                "sentiment_score": 0,
                "scene_score": 0,
                "keywords": [],
                "text": " ".join(text_parts)[:200],
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

        title = _clean_text(seg.get("title"))
        hook_text = _clean_text(seg.get("hook_text"))
        if not title or not hook_text:
            title, hook_text = _generate_title_and_hook(seg.get("text", ""), i)

        highlights.append(
            {
                "rank": i,
                "start_time": round(seg["start"], 2),
                "end_time": round(seg["end"], 2),
                "duration": round(duration, 2),
                "virality_score": seg["virality_score"],
                "title": title,
                "hook_text": hook_text,
                "scores": {
                    "transcript": round(seg.get("transcript_score", 0), 2),
                    "laughter": round(seg.get("laughter_score", 0), 2),
                    "sentiment": round(seg.get("sentiment_score", 0), 2),
                    "scenes": round(seg.get("scene_score", 0), 2),
                },
                "text": seg.get("text", "")[:200],
                "reasoning": _clean_text(seg.get("reasoning")) or f"Contains {reasoning}",
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
    parser.add_argument("--min-duration", type=float, default=40)
    parser.add_argument("--max-duration", type=float, default=120)
    parser.add_argument(
        "--highlight-model", choices=["heuristic", "auto", "openai", "gemini"], default="heuristic"
    )
    parser.add_argument("--highlight-ai-model", default=None)
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--gemini-api-key", default=os.getenv("GEMINI_API_KEY"))
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
        highlight_model=args.highlight_model,
        highlight_ai_model=args.highlight_ai_model,
        openai_api_key=args.openai_api_key,
        gemini_api_key=args.gemini_api_key,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
