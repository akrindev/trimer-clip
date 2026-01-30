#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper
from shared.audio_utils import extract_audio_from_video
import tempfile


def diarize_video(
    video_path: str,
    output_format: str = "json",
    min_speakers: int = None,
    max_speakers: int = None,
    num_speakers: int = None,
    device: str = "auto",
    huggingface_token: str = None,
) -> dict:
    """Perform speaker diarization on video."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        token = huggingface_token or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            return {
                "success": False,
                "error": "HuggingFace token required. Set HUGGINGFACE_TOKEN env var or use --huggingface-token",
            }

        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        extract_audio_from_video(video_path, temp_audio_path, sample_rate=16000)

        try:
            from pyannote.audio import Pipeline
            import torch

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1", use_auth_token=token
            )
            pipeline.to(torch.device(device))

            if num_speakers:
                diarization = pipeline(temp_audio_path, num_speakers=num_speakers)
            else:
                diarization = pipeline(temp_audio_path)

            segments = []
            speaker_stats = {}
            overlapping_segments = []

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start,
                }
                segments.append(segment)

                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {"duration": 0, "segments": 0}
                speaker_stats[speaker]["duration"] += segment["duration"]
                speaker_stats[speaker]["segments"] += 1

            for i in range(len(segments) - 1):
                curr = segments[i]
                next_seg = segments[i + 1]

                if curr["end"] > next_seg["start"] and curr["speaker"] != next_seg["speaker"]:
                    overlapping_segments.append(
                        {
                            "start": next_seg["start"],
                            "end": min(curr["end"], next_seg["end"]),
                            "speakers": [curr["speaker"], next_seg["speaker"]],
                        }
                    )

            result = {
                "success": True,
                "video_path": video_path,
                "num_speakers": len(speaker_stats),
                "duration": ffmpeg.get_duration(video_path),
                "speakers": speaker_stats,
                "segments": segments,
                "overlapping_segments": overlapping_segments,
                "device_used": device,
            }

            os.unlink(temp_audio_path)

            if output_format == "rttm":
                return convert_to_rttm(result)
            elif output_format == "srt":
                return convert_to_srt(result)
            return result

        except ImportError:
            os.unlink(temp_audio_path)
            return {
                "success": False,
                "error": "pyannote.audio not installed. Run: pip install pyannote.audio torch torchaudio",
            }

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def convert_to_rttm(result: dict) -> str:
    """Convert diarization result to RTTM format."""
    rttm_lines = []
    base_name = Path(result["video_path"]).stem

    for seg in result["segments"]:
        rttm_lines.append(
            f"SPEAKER {base_name} 1 {seg['start']:.3f} {seg['duration']:.3f} "
            f"<NA> <NA> {seg['speaker']} <NA> <NA>"
        )

    return "\n".join(rttm_lines)


def convert_to_srt(result: dict) -> str:
    """Convert diarization result to SRT format."""
    srt_lines = []

    for i, seg in enumerate(result["segments"], 1):
        start = format_time_srt(seg["start"])
        end = format_time_srt(seg["end"])
        speaker = seg["speaker"]

        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(f"[{speaker}]")
        srt_lines.append("")

    return "\n".join(srt_lines)


def format_time_srt(seconds: float) -> str:
    """Format time for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Speaker diarization with pyannote")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-o", "--output", choices=["json", "rttm", "srt"], default="json")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--huggingface-token", default=os.getenv("HUGGINGFACE_TOKEN"))

    args = parser.parse_args()

    result = diarize_video(
        video_path=args.video_path,
        output_format=args.output,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        num_speakers=args.num_speakers,
        device=args.device,
        huggingface_token=args.huggingface_token,
    )

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    sys.exit(0 if (isinstance(result, str) or result.get("success")) else 1)


if __name__ == "__main__":
    main()
