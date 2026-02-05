import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIWhisperTranscriber:
    """Wrapper for OpenAI Whisper API transcription."""

    def __init__(self, api_key: Optional[str] = None, model: str = "whisper-1"):
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIWhisperTranscriber")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        return_format: str = "segments",
    ) -> List[Dict[str, Any]]:
        """Transcribe audio file with OpenAI Whisper API."""
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word"] if word_timestamps else ["segment"],
            )

        data = _coerce_response(response)
        segments = data.get("segments", []) or []
        words = _normalize_words(data.get("words", []) or [])

        if not segments and words:
            segments = _segments_from_words(words)

        if segments and words:
            _attach_words_to_segments(segments, words)

        result = []
        for i, segment in enumerate(segments, 1):
            result.append(
                {
                    "index": i,
                    "start": float(segment.get("start", 0)),
                    "end": float(segment.get("end", 0)),
                    "text": str(segment.get("text", "")).strip(),
                    "words": segment.get("words", []),
                }
            )

        if return_format == "segments":
            return result

        return result


def _coerce_response(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return {}


def _normalize_words(words: List[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("word", "")).strip()
        if not text:
            continue
        normalized.append(
            {
                "word": text,
                "start": float(word.get("start", 0)),
                "end": float(word.get("end", 0)),
            }
        )
    return normalized


def _segments_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return []

    start = words[0]["start"]
    end = words[-1]["end"]
    text = " ".join(word["word"] for word in words)
    return [{"start": start, "end": end, "text": text, "words": words}]


def _attach_words_to_segments(segments: List[Dict[str, Any]], words: List[Dict[str, Any]]) -> None:
    word_index = 0
    total_words = len(words)

    for segment in segments:
        seg_start = float(segment.get("start", 0))
        seg_end = float(segment.get("end", 0))

        seg_words = []
        while word_index < total_words and words[word_index]["end"] <= seg_start:
            word_index += 1

        scan_index = word_index
        while scan_index < total_words and words[scan_index]["start"] < seg_end:
            seg_words.append(words[scan_index])
            scan_index += 1

        segment["words"] = seg_words
        word_index = scan_index
