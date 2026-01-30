#!/usr/bin/env python3
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from shared.ffmpeg_wrapper import FFmpegWrapper
from shared.audio_utils import extract_audio_from_video
from models.whisper_transcriber import WhisperTranscriber
from models.gemini_transcriber import GeminiTranscriber
import tempfile


def transcribe_video(
    video_path: str,
    model: str = "auto",
    whisper_model: str = "medium",
    use_faster: bool = True,
    output_path: str = None,
    format: str = "srt",
    language: str = None,
    speaker_diarization: bool = False,
    emotion_detection: bool = False,
    device: str = "auto",
) -> dict:
    """Transcribe audio from video file."""

    if not Path(video_path).exists():
        return {"success": False, "error": f"Video file not found: {video_path}"}

    try:
        ffmpeg = FFmpegWrapper()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        extract_audio_from_video(video_path, temp_audio_path, sample_rate=16000)

        if model == "auto":
            model = _select_model(video_path, speaker_diarization, emotion_detection, language)

        if model == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return {"success": False, "error": "GEMINI_API_KEY environment variable not set"}

            transcriber = GeminiTranscriber(api_key=api_key, model="gemini-flash-lite-latest")

            result = transcriber.transcribe(
                temp_audio_path,
                speaker_diarization=speaker_diarization,
                emotion_detection=emotion_detection,
                return_timestamps=True,
                language=language,
            )

            output_data = _format_gemini_result(result, format)

        else:
            transcriber = WhisperTranscriber(
                model_size=whisper_model, use_faster=use_faster, language=language, device=device
            )

            segments = transcriber.transcribe(
                temp_audio_path, word_timestamps=True, return_format="segments"
            )

            output_data = _format_whisper_result(segments, format)

        os.unlink(temp_audio_path)

        if output_path is None:
            output_path = str(Path(video_path).with_suffix(f".{format}"))

        with open(output_path, "w", encoding="utf-8") as f:
            if format == "json":
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            else:
                f.write(output_data)

        return {"success": True, "output_path": output_path, "format": format, "model_used": model}

    except Exception as e:
        return {"success": False, "error": str(e), "video_path": video_path}


def _select_model(
    video_path: str, speaker_diarization: bool, emotion_detection: bool, language: str
) -> str:
    """Auto-select appropriate transcription model."""
    if speaker_diarization or emotion_detection:
        return "gemini"

    ffmpeg = FFmpegWrapper()
    duration = ffmpeg.get_duration(video_path)

    if duration > 3600:
        return "whisper"

    return "gemini"


def _format_gemini_result(result: list, format: str):
    """Format Gemini transcription result."""
    if format == "json":
        return result

    srt_content = "WEBVTT\n\n" if format == "vtt" else ""
    segments = []

    for i, item in enumerate(result, 1):
        start = item.get("timestamp", 0)
        end = item.get("end_time", start + 3)
        speaker = item.get("speaker", "")
        text = item.get("text", "")

        if format == "srt":
            srt_content += f"{i}\n"
            srt_content += f"{_format_time_srt(start)} --> {_format_time_srt(end)}\n"
            if speaker:
                srt_content += f"{speaker}: "
            srt_content += f"{text}\n\n"
        elif format == "vtt":
            segments.append(
                f"{_format_time_vtt(start)} --> {_format_time_vtt(end)}\n{speaker}: {text}"
            )

    if format == "vtt":
        srt_content += "\n".join(segments)

    return srt_content


def _format_whisper_result(result: list, format: str):
    """Format Whisper transcription result."""
    if format == "json":
        return result

    srt_content = ""

    for seg in result:
        if format == "srt":
            srt_content += f"{seg['index']}\n"
            srt_content += f"{_format_time_srt(seg['start'])} --> {_format_time_srt(seg['end'])}\n"
            srt_content += f"{seg['text']}\n\n"

    return srt_content


def _format_time_srt(seconds: float) -> str:
    """Format time for SRT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_time_vtt(seconds: float) -> str:
    """Format time for WebVTT."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("-m", "--model", choices=["auto", "whisper", "gemini"], default="auto")
    parser.add_argument(
        "--whisper-model", choices=["tiny", "base", "small", "medium", "large-v3"], default="medium"
    )
    parser.add_argument("--use-faster", action="store_true", default=True)
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["srt", "vtt", "json"], default="srt")
    parser.add_argument("--language", help="Language code (e.g., en, id)")
    parser.add_argument("--speaker-diarization", action="store_true")
    parser.add_argument("--emotion-detection", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()

    result = transcribe_video(
        video_path=args.video_path,
        model=args.model,
        whisper_model=args.whisper_model,
        use_faster=args.use_faster,
        output_path=args.output,
        format=args.format,
        language=args.language,
        speaker_diarization=args.speaker_diarization,
        emotion_detection=args.emotion_detection,
        device=args.device,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
