import os
from typing import Any, Dict, List, Optional

try:
    from google.cloud import speech
except ImportError:
    speech = None


DEFAULT_LANGUAGE = "id-ID"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MODEL = "latest_long"


class GoogleSpeechTranscriber:
    """Wrapper for Google Speech-to-Text transcription."""

    def __init__(
        self,
        language: Optional[str] = None,
        model: Optional[str] = None,
        use_enhanced: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        if speech is None:
            raise ImportError("google-cloud-speech package is required for GoogleSpeechTranscriber")

        self.client = speech.SpeechClient()
        self.language = language or os.getenv("GOOGLE_SPEECH_LANGUAGE", DEFAULT_LANGUAGE)
        self.model = model or os.getenv("GOOGLE_SPEECH_MODEL", DEFAULT_MODEL)
        self.use_enhanced = use_enhanced
        self.sample_rate = sample_rate

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        diarization: bool = False,
        return_format: str = "segments",
    ) -> List[Dict[str, Any]]:
        """Transcribe audio file with Google Speech-to-Text."""
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code=language or self.language,
            enable_automatic_punctuation=True,
            enable_word_time_offsets=word_timestamps,
            model=self.model,
            use_enhanced=self.use_enhanced,
        )

        if diarization:
            config.diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=2,
                max_speaker_count=6,
            )

        audio = speech.RecognitionAudio(content=content)

        if len(content) > 10 * 1024 * 1024:
            operation = self.client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=3600)
        else:
            response = self.client.recognize(config=config, audio=audio)

        segments = _results_to_segments(response.results)

        if return_format == "segments":
            return segments

        return segments


def _results_to_segments(results: List[Any]) -> List[Dict[str, Any]]:
    segments = []
    for result in results or []:
        if not result.alternatives:
            continue
        alternative = result.alternatives[0]
        words = []
        for word in getattr(alternative, "words", []) or []:
            words.append(
                {
                    "word": str(word.word).strip(),
                    "start": _duration_seconds(word.start_time),
                    "end": _duration_seconds(word.end_time),
                }
            )

        text = str(alternative.transcript).strip()
        start_time = words[0]["start"] if words else 0.0
        end_time = words[-1]["end"] if words else _duration_seconds(result.result_end_time)

        segments.append(
            {
                "start": start_time,
                "end": end_time,
                "text": text,
                "words": words,
            }
        )

    return segments


def _duration_seconds(duration: Any) -> float:
    if not duration:
        return 0.0
    seconds = getattr(duration, "seconds", 0)
    nanos = getattr(duration, "nanos", 0)
    return float(seconds) + float(nanos) / 1e9
