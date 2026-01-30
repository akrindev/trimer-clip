#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


def analyze_speaker_dynamics(diarization_path: str) -> dict:
    """Analyze speaker interactions and dynamics."""

    if not Path(diarization_path).exists():
        return {"success": False, "error": f"Diarization file not found: {diarization_path}"}

    try:
        with open(diarization_path, "r") as f:
            diarization = json.load(f)

        speakers = diarization.get("speakers", {})
        segments = diarization.get("segments", [])
        overlapping = diarization.get("overlapping_segments", [])

        total_duration = diarization.get("duration", 0)

        dominant_speaker = max(speakers.items(), key=lambda x: x[1]["duration"])[0]

        speaker_durations = [s["duration"] for s in speakers.values()]
        avg_duration = sum(speaker_durations) / len(speaker_durations)
        max_deviation = max(abs(d - avg_duration) for d in speaker_durations)
        speaker_balance = 1.0 - (max_deviation / avg_duration if avg_duration > 0 else 0)

        interaction_moments = []

        for i in range(len(segments) - 1):
            curr = segments[i]
            next_seg = segments[i + 1]

            if curr["speaker"] != next_seg["speaker"]:
                gap = next_seg["start"] - curr["end"]

                if gap < 1.0:
                    interaction_type = "quick_exchange"
                elif gap < 3.0:
                    interaction_type = "conversation"
                else:
                    interaction_type = "turn_change"

                interaction_moments.append(
                    {
                        "type": interaction_type,
                        "start": curr["end"],
                        "end": next_seg["start"],
                        "from_speaker": curr["speaker"],
                        "to_speaker": next_seg["speaker"],
                        "gap": gap,
                    }
                )

        for overlap in overlapping:
            interaction_moments.append(
                {
                    "type": "overlapping_speech",
                    "start": overlap["start"],
                    "end": overlap["end"],
                    "speakers": overlap["speakers"],
                    "duration": overlap["end"] - overlap["start"],
                }
            )

        speaker_activity_timeline = []
        window_size = 30.0

        for window_start in range(0, int(total_duration), int(window_size)):
            window_end = window_start + window_size
            window_speakers = {}

            for seg in segments:
                if seg["start"] < window_end and seg["end"] > window_start:
                    overlap_start = max(seg["start"], window_start)
                    overlap_end = min(seg["end"], window_end)
                    duration = overlap_end - overlap_start

                    speaker = seg["speaker"]
                    window_speakers[speaker] = window_speakers.get(speaker, 0) + duration

            if window_speakers:
                most_active = max(window_speakers.items(), key=lambda x: x[1])[0]
                speaker_activity_timeline.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "most_active_speaker": most_active,
                        "activity_scores": window_speakers,
                    }
                )

        return {
            "success": True,
            "speaker_dynamics": {
                "total_speakers": len(speakers),
                "dominant_speaker": dominant_speaker,
                "speaker_balance": round(speaker_balance, 3),
                "total_interactions": len(interaction_moments),
                "overlapping_segments_count": len(overlapping),
                "interaction_moments": sorted(interaction_moments, key=lambda x: x["start"]),
                "speaker_activity_timeline": speaker_activity_timeline,
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Analyze speaker dynamics")
    parser.add_argument("diarization_path", help="Path to diarization JSON")

    args = parser.parse_args()

    result = analyze_speaker_dynamics(args.diarization_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
