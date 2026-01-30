#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from shared.video_utils import parse_srt


EMOTION_KEYWORDS = {
    "positive": [
        "amazing",
        "incredible",
        "fantastic",
        "awesome",
        "wonderful",
        "great",
        "excellent",
        "love",
        "happy",
        "joy",
        "excited",
        "thrilled",
        "delighted",
        "pleased",
        "perfect",
        "brilliant",
        "outstanding",
        "impressive",
        "remarkable",
        "beautiful",
        "best",
        "winning",
        "success",
        "victory",
        "achieve",
        "accomplished",
    ],
    "negative": [
        "terrible",
        "awful",
        "horrible",
        "bad",
        "disappointing",
        "hate",
        "angry",
        "frustrated",
        "annoyed",
        "upset",
        "sad",
        "depressed",
        "worst",
        "disgusting",
        "pathetic",
        "useless",
        "failed",
        "fail",
        "broken",
        "wrong",
        "mistake",
        "problem",
        "issue",
        "disaster",
    ],
    "surprise": [
        "wow",
        "oh my god",
        "what!",
        "unbelievable",
        "shocked",
        "stunned",
        "surprised",
        "amazed",
        "astonished",
        "can't believe",
        "no way",
        "crazy",
        "insane",
        "wild",
        "unreal",
        "mind blown",
        "incredible",
    ],
    "excitement": [
        "let's go",
        "come on",
        "yes!",
        "pumped",
        "fired up",
        "hyped",
        "thrilled",
        "ecstatic",
        "overjoyed",
        "enthusiastic",
        "energized",
        "go for it",
        "do it",
        "now!",
        "hurry",
        "fast",
        "quick",
    ],
}


def analyze_sentiment(
    video_path: str, method: str = "keywords", transcript_path: str = None
) -> dict:
    """Analyze sentiment in video content."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        if method == "keywords":
            if not transcript_path:
                return {"success": False, "error": "Transcript path required for keyword detection"}

            if not Path(transcript_path).exists():
                return {"success": False, "error": f"Transcript file not found: {transcript_path}"}

            result = analyze_from_keywords(transcript_path)

        elif method == "ai":
            result = analyze_with_ai(video_path)

        elif method == "audio":
            result = analyze_from_audio(video_path)

        else:
            return {"success": False, "error": f"Unknown method: {method}"}

        result["success"] = True
        result["video_path"] = video_path
        result["method"] = method

        return result

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def analyze_from_keywords(transcript_path: str) -> dict:
    """Analyze sentiment from transcript keywords."""
    transcript = parse_srt(transcript_path)

    sentiment_counts = {"positive": 0, "negative": 0, "surprise": 0, "excitement": 0, "neutral": 0}

    emotional_peaks = []
    sentiment_timeline = []

    for item in transcript:
        text = item["text"].lower()
        start = item["start"]
        end = item["end"]

        detected_emotions = []
        for emotion, keywords in EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    detected_emotions.append(emotion)
                    sentiment_counts[emotion] += 1
                    break

        if detected_emotions:
            intensity = len(detected_emotions) * 0.3 + 0.4
            intensity = min(intensity, 1.0)

            emotional_peaks.append(
                {
                    "timestamp": start,
                    "duration": end - start,
                    "emotion": detected_emotions[0],
                    "intensity": round(intensity, 2),
                    "text": item["text"],
                    "keywords": [kw for kw in detected_emotions],
                }
            )

    total_count = sum(sentiment_counts.values())
    if total_count > 0:
        overall_sentiment = {
            "positive": sentiment_counts["positive"] / total_count,
            "negative": sentiment_counts["negative"] / total_count,
            "surprise": sentiment_counts["surprise"] / total_count,
            "excitement": sentiment_counts["excitement"] / total_count,
            "neutral": sentiment_counts["neutral"] / total_count,
        }
    else:
        overall_sentiment = {
            "positive": 0,
            "negative": 0,
            "surprise": 0,
            "excitement": 0,
            "neutral": 1.0,
        }

    return {
        "overall_sentiment": overall_sentiment,
        "emotional_peaks": emotional_peaks,
        "sentiment_timeline": sentiment_timeline,
    }


def analyze_with_ai(video_path: str) -> dict:
    """Analyze sentiment using Gemini AI."""
    import os
    from models.gemini_transcriber import GeminiTranscriber

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "overall_sentiment": {},
            "emotional_peaks": [],
            "sentiment_timeline": [],
            "error": "GEMINI_API_KEY not set",
        }

    try:
        from shared.ffmpeg_wrapper import FFmpegWrapper
        from shared.audio_utils import extract_audio_from_video
        import tempfile

        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

            extract_audio_from_video(video_path, temp_audio_path, sample_rate=16000)

            transcriber = GeminiTranscriber(api_key=api_key, model="gemini-flash-lite-latest")
            analysis = transcriber.analyze_audio(temp_audio_path, "emotions")

            os.unlink(temp_audio_path)

            emotional_peaks = []
            if "analysis" in analysis:
                lines = analysis["analysis"].split("\n")
                for line in lines:
                    if "[" in line and "]" in line:
                        try:
                            time_str = line.split("]")[0].replace("[", "")
                            timestamp = (
                                float(time_str.split(":")[0]) * 3600
                                + float(time_str.split(":")[1]) * 60
                                + float(time_str.split(":")[2])
                            )

                            emotional_peaks.append(
                                {
                                    "timestamp": timestamp,
                                    "duration": 5.0,
                                    "emotion": "detected",
                                    "intensity": 0.75,
                                    "text": line.split(":", 1)[1].strip() if ":" in line else line,
                                }
                            )
                        except:
                            continue

            return {
                "overall_sentiment": {"detected": 1},
                "emotional_peaks": emotional_peaks,
                "sentiment_timeline": [],
                "ai_analysis": analysis.get("analysis", ""),
            }

    except Exception as e:
        return {
            "overall_sentiment": {},
            "emotional_peaks": [],
            "sentiment_timeline": [],
            "error": str(e),
        }


def analyze_from_audio(video_path: str) -> dict:
    """Analyze sentiment from audio features."""
    try:
        from shared.ffmpeg_wrapper import FFmpegWrapper
        from shared.audio_utils import load_audio, extract_audio_features, extract_audio_from_video
        import tempfile

        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

            extract_audio_from_video(video_path, temp_audio_path, sample_rate=16000)
            audio, sr = load_audio(temp_audio_path, sample_rate=16000)

            features = extract_audio_features(audio, sr)

            emotional_peaks = []
            for i in range(1, len(features) - 1):
                curr = features[i]
                prev = features[i - 1]
                next_f = features[i + 1]

                if (
                    curr["energy"] > prev["energy"] * 1.5
                    and curr["energy"] > next_f["energy"] * 1.5
                ):
                    emotional_peaks.append(
                        {
                            "timestamp": curr["timestamp"],
                            "duration": 3.0,
                            "emotion": "excitement",
                            "intensity": min(curr["energy"] * 2, 1.0),
                            "text": "[audio peak]",
                        }
                    )

            os.unlink(temp_audio_path)

            return {
                "overall_sentiment": {
                    "excitement": len(emotional_peaks) / len(features) if features else 0
                },
                "emotional_peaks": emotional_peaks[:10],
                "sentiment_timeline": [],
            }

    except Exception as e:
        return {
            "overall_sentiment": {},
            "emotional_peaks": [],
            "sentiment_timeline": [],
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment in video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--method", choices=["keywords", "ai", "audio"], default="keywords")
    parser.add_argument("--transcript-path", help="Path to transcript SRT/VTT file")
    parser.add_argument("-o", "--output", help="Output JSON path")

    args = parser.parse_args()

    result = analyze_sentiment(
        video_path=args.video_path, method=args.method, transcript_path=args.transcript_path
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
