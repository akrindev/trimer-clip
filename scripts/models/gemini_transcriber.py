from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
import base64
import json


class GeminiTranscriber:
    """Wrapper for Gemini API audio transcription and analysis."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-lite-latest",
        project_id: Optional[str] = None,
        location: str = "us-central1",
    ):
        self.api_key = api_key
        self.model = model
        self.project_id = project_id
        self.location = location

        if project_id:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
        else:
            self.client = genai.Client(api_key=self.api_key)

    def transcribe(
        self,
        audio_path: str,
        speaker_diarization: bool = True,
        emotion_detection: bool = False,
        return_timestamps: bool = True,
        language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe audio file with optional speaker diarization and emotion detection."""
        try:
            # The new SDK handles file upload differently or we can pass the path if it supports it
            # But the most reliable way for audio is usually upload or direct bytes for small files
            # For 20 mins audio, definitely upload.

            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

            prompt_parts = ["Transcribe this audio"]

            if language:
                prompt_parts.append(f" in {language}")

            if speaker_diarization:
                prompt_parts.append(" with speaker labels (Speaker A, Speaker B, etc.)")

            if return_timestamps:
                prompt_parts.append(" with timestamps in [HH:MM:SS] format")

            if emotion_detection:
                prompt_parts.append(" and detect emotions for each speaker")

            prompt = " ".join(prompt_parts) + "."

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        ],
                    )
                ],
            )

            return self._parse_transcript(response.text, speaker_diarization, emotion_detection)

        except Exception as e:
            raise RuntimeError(f"Gemini transcription failed: {e}")

    def _parse_transcript(
        self, transcript_text: str, speaker_diarization: bool, emotion_detection: bool
    ) -> List[Dict[str, Any]]:
        """Parse Gemini response into structured format."""
        segments = []

        lines = transcript_text.strip().split("\n")
        current_speaker = None
        current_emotion = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if speaker_diarization and line.startswith("["):
                time_str = line.split("]")[0].replace("[", "")
                try:
                    timestamp = self._parse_timestamp(time_str)
                except:
                    continue

                speaker_part = line.split(":", 1)[1].strip() if ":" in line else ""
                emotion_part = ""

                if emotion_detection and "(" in speaker_part:
                    parts = speaker_part.split("(")
                    current_speaker = parts[0].strip()
                    current_emotion = parts[1].replace(")", "").strip()
                else:
                    current_speaker = speaker_part

                segments.append(
                    {
                        "timestamp": timestamp,
                        "speaker": current_speaker,
                        "emotion": current_emotion,
                        "text": line.split(":", 1)[1].strip() if ":" in line else line,
                    }
                )

        return segments

    def _parse_timestamp(self, time_str: str) -> float:
        """Parse timestamp string to seconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        return float(time_str)

    def analyze_audio(self, audio_path: str, analysis_type: str = "viral") -> Dict[str, Any]:
        """Analyze audio content for specific insights."""
        try:
            file = self.client.files.upload(path=audio_path)

            prompts = {
                "viral": (
                    "Analyze this audio and identify the most viral-worthy segments. "
                    "Return timestamps with brief descriptions of each segment's viral potential. "
                    "Focus on: hooks, emotional peaks, surprising moments, "
                    "humorous content, and key takeaways."
                ),
                "summary": (
                    "Summarize the main topics and key points from this audio. "
                    "Return a structured summary with main themes and important moments."
                ),
                "emotions": (
                    "Analyze the emotional journey throughout this audio. "
                    "Identify emotional peaks, tone changes, and overall sentiment."
                ),
                "questions": (
                    "Extract key questions asked and answers provided in this audio. "
                    "Return them as pairs with timestamps."
                ),
            }

            prompt = prompts.get(analysis_type, prompts["viral"])

            response = self.client.models.generate_content(
                model=self.model, contents=[prompt, file]
            )

            return {"type": analysis_type, "analysis": response.text, "raw_response": response}

        except Exception as e:
            raise RuntimeError(f"Gemini audio analysis failed: {e}")

    def transcribe_with_timestamps(self, audio_path: str, output_format: str = "srt") -> str:
        """Transcribe and return formatted text (SRT/VTT)."""
        segments = self.transcribe(audio_path, return_timestamps=True)

        if output_format == "srt":
            return self._to_srt(segments)
        elif output_format == "vtt":
            return self._to_vtt(segments)
        return json.dumps(segments, indent=2)

    def _to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to SRT format."""
        srt_content = ""
        for i, seg in enumerate(segments, 1):
            if "timestamp" in seg:
                srt_content += f"{i}\n"
                srt_content += f"{self._format_time(seg['timestamp'])} --> {self._format_time(seg.get('end_time', seg['timestamp'] + 3))}\n"
                speaker_label = f"{seg.get('speaker', '')}: " if seg.get("speaker") else ""
                srt_content += f"{speaker_label}{seg.get('text', '')}\n\n"
        return srt_content

    def _to_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert segments to WebVTT format."""
        vtt_content = "WEBVTT\n\n"
        for seg in segments:
            if "timestamp" in seg:
                vtt_content += f"{self._format_time_vtt(seg['timestamp'])} --> {self._format_time_vtt(seg.get('end_time', seg['timestamp'] + 3))}\n"
                speaker_label = f"{seg.get('speaker', '')}: " if seg.get("speaker") else ""
                vtt_content += f"{speaker_label}{seg.get('text', '')}\n\n"
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

    def detect_viral_segments(self, audio_path: str, num_segments: int = 5) -> List[Dict[str, Any]]:
        """Detect viral-worthy segments in audio."""
        analysis = self.analyze_audio(audio_path, "viral")

        segments = []
        lines = analysis["analysis"].split("\n")

        for line in lines:
            if "[" in line:
                try:
                    time_str = line.split("]")[0].replace("[", "")
                    timestamp = self._parse_timestamp(time_str)
                    description = line.split(":", 1)[1].strip() if ":" in line else line

                    segments.append(
                        {
                            "start": timestamp,
                            "end": timestamp + 30,
                            "description": description,
                            "score": 0.8,
                        }
                    )
                except:
                    continue

        return segments[:num_segments]


def select_transcription_model(
    use_gemini: bool = False, gemini_api_key: Optional[str] = None, whisper_model: str = "medium"
):
    """Select and return appropriate transcription model."""
    if use_gemini and gemini_api_key:
        return GeminiTranscriber(api_key=gemini_api_key)
    else:
        return WhisperTranscriber(model_size=whisper_model)
