import whisper
try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    faster_whisper = None
    FASTER_WHISPER_AVAILABLE = False
from typing import List, Dict, Any, Optional
from pathlib import Path


class WhisperTranscriber:
    """Wrapper for Whisper audio transcription models."""

    def __init__(
        self,
        model_size: str = "medium",
        use_faster: bool = True,
        language: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_size = model_size
        # Fall back to original whisper if faster_whisper is not available
        self.use_faster = use_faster and FASTER_WHISPER_AVAILABLE
        self.language = language
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        if self.use_faster:
            return faster_whisper.WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
            )
        else:
            # Original whisper doesn't accept 'auto', determine device
            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            return whisper.load_model(self.model_size, device=device)

    def transcribe(
        self, audio_path: str, word_timestamps: bool = True, return_format: str = "segments"
    ) -> List[Dict[str, Any]]:
        """Transcribe audio file."""
        if self.use_faster:
            return self._transcribe_faster(audio_path, word_timestamps, return_format)
        else:
            return self._transcribe_original(audio_path, word_timestamps, return_format)

    def _transcribe_faster(
        self, audio_path: str, word_timestamps: bool, return_format: str
    ) -> List[Dict[str, Any]]:
        """Transcribe using faster-whisper."""
        segments, info = self.model.transcribe(
            audio_path, language=self.language, word_timestamps=word_timestamps, beam_size=5
        )

        result = []
        for i, segment in enumerate(segments):
            result.append(
                {
                    "index": i + 1,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": segment.words if word_timestamps else [],
                }
            )

        if return_format == "srt":
            return self._to_srt(result)
        elif return_format == "vtt":
            return self._to_vtt(result)
        return result

    def _transcribe_original(
        self, audio_path: str, word_timestamps: bool, return_format: str
    ) -> List[Dict[str, Any]]:
        """Transcribe using original Whisper."""
        result = self.model.transcribe(
            audio_path, language=self.language, word_timestamps=word_timestamps
        )

        segments = []
        for i, segment in enumerate(result["segments"]):
            segments.append(
                {
                    "index": i + 1,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "words": segment.get("words", []),
                }
            )

        if return_format == "srt":
            return self._to_srt(segments)
        elif return_format == "vtt":
            return self._to_vtt(segments)
        return segments

    def _to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to SRT format."""
        srt_content = ""
        for seg in segments:
            srt_content += f"{seg['index']}\n"
            srt_content += (
                f"{self._format_time(seg['start'])} --> {self._format_time(seg['end'])}\n"
            )
            srt_content += f"{seg['text']}\n\n"
        return srt_content

    def _to_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to WebVTT format."""
        vtt_content = "WEBVTT\n\n"
        for seg in segments:
            vtt_content += (
                f"{self._format_time_vtt(seg['start'])} --> {self._format_time_vtt(seg['end'])}\n"
            )
            vtt_content += f"{seg['text']}\n\n"
        return vtt_content

    def _format_time(self, seconds: float) -> str:
        """Format time for SRT."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def _format_time_vtt(self, seconds: float) -> str:
        """Format time for WebVTT."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def transcribe_to_file(self, audio_path: str, output_path: str, format: str = "srt") -> bool:
        """Transcribe and save to file."""
        segments = self.transcribe(audio_path, word_timestamps=True, return_format=format)

        with open(output_path, "w", encoding="utf-8") as f:
            if format in ["srt", "vtt"]:
                f.write(segments)
            else:
                import json

                json.dump(segments, f, indent=2, ensure_ascii=False)

        return True

    def get_available_languages(self) -> Dict[str, str]:
        """Get available languages."""
        return whisper.tokenizer.LANGUAGES

    def detect_language(self, audio_path: str) -> str:
        """Detect language from audio."""
        if self.use_faster:
            _, info = self.model.transcribe(audio_path, beam_size=5)
            return info.language
        else:
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio)
            _, probs = self.model.detect_language(mel)
            return max(probs, key=probs.get)
