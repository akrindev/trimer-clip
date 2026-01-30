from typing import List, Dict, Any, Tuple
import subprocess
import json
import soundfile as sf
import numpy as np
from pathlib import Path


def extract_audio_from_video(
    video_path: str, output_path: str, sample_rate: int = 16000, channels: int = 1
) -> bool:
    """Extract audio from video file using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        "-y",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def load_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and return audio data and sample rate."""
    data, sr = sf.read(audio_path)
    if sr != sample_rate:
        data = resample_audio(data, sr, sample_rate)
    return data, sample_rate


def save_audio(audio: np.ndarray, output_path: str, sample_rate: int = 16000) -> None:
    """Save audio to file."""
    sf.write(output_path, audio, sample_rate)


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    ratio = target_sr / original_sr
    new_length = int(len(audio) * ratio)
    indices = np.round(np.linspace(0, len(audio) - 1, new_length)).astype(int)
    return audio[indices]


def split_audio_by_timestamps(
    audio: np.ndarray, timestamps: List[float], sample_rate: int = 16000
) -> List[np.ndarray]:
    """Split audio into segments based on timestamps."""
    segments = []

    for i, timestamp in enumerate(timestamps):
        start_sample = int(timestamp * sample_rate)
        end_sample = int(timestamps[i + 1] * sample_rate) if i + 1 < len(timestamps) else len(audio)

        if start_sample < len(audio):
            segment = audio[start_sample:end_sample]
            if len(segment) > 0:
                segments.append(segment)

    return segments


def detect_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.5,
) -> List[Dict[str, float]]:
    """Detect silence segments in audio."""
    frame_size = int(sample_rate * 0.05)
    hop_size = int(sample_rate * 0.01)

    is_silent = np.abs(audio) < silence_threshold
    silence_segments = []

    in_silence = False
    silence_start = 0

    for i in range(0, len(is_silent), hop_size):
        frame_silent = np.all(is_silent[i : min(i + frame_size, len(is_silent))])

        if frame_silent and not in_silence:
            silence_start = i / sample_rate
            in_silence = True
        elif not frame_silent and in_silence:
            silence_end = i / sample_rate
            if silence_end - silence_start >= min_silence_duration:
                silence_segments.append({"start": silence_start, "end": silence_end})
            in_silence = False

    return silence_segments


def calculate_audio_level(audio: np.ndarray) -> float:
    """Calculate RMS audio level."""
    rms = np.sqrt(np.mean(np.square(audio)))
    return float(rms)


def normalize_audio(audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    current_level = calculate_audio_level(audio)
    if current_level == 0:
        return audio

    target_linear = 10 ** (target_level / 20)
    gain = target_linear / (current_level + 1e-10)
    return audio * gain


def remove_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    silence_threshold: float = 0.01,
    min_silence_duration: float = 0.3,
) -> np.ndarray:
    """Remove silence from audio."""
    silence_segments = detect_silence(audio, sample_rate, silence_threshold, min_silence_duration)

    mask = np.ones(len(audio), dtype=bool)
    for seg in silence_segments:
        start = int(seg["start"] * sample_rate)
        end = int(seg["end"] * sample_rate)
        mask[start:end] = False

    return audio[mask]


def fade_audio(
    audio: np.ndarray,
    fade_in_duration: float = 0.1,
    fade_out_duration: float = 0.1,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Apply fade in and fade out to audio."""
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)

    result = audio.copy()

    if len(result) > fade_in_samples:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        if len(result.shape) > 1:
            fade_in_curve = fade_in_curve[:, np.newaxis]
        result[:fade_in_samples] *= fade_in_curve

    if len(result) > fade_out_samples:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        if len(result.shape) > 1:
            fade_out_curve = fade_out_curve[:, np.newaxis]
        result[-fade_out_samples:] *= fade_out_curve

    return result


def calculate_segments_from_transcript(
    transcript: List[Dict[str, Any]], max_gap: float = 2.0
) -> List[Dict[str, float]]:
    """Calculate continuous segments from transcript timestamps."""
    if not transcript:
        return []

    segments = []
    current_start = transcript[0]["start"]
    current_end = transcript[0]["end"]

    for entry in transcript[1:]:
        if entry["start"] - current_end <= max_gap:
            current_end = entry["end"]
        else:
            segments.append({"start": current_start, "end": current_end})
            current_start = entry["start"]
            current_end = entry["end"]

    segments.append({"start": current_start, "end": current_end})
    return segments


def detect_laughter_patterns(
    transcript: List[Dict[str, Any]], laughter_keywords: List[str] = None
) -> List[Dict[str, float]]:
    """Detect laughter segments from transcript."""
    if laughter_keywords is None:
        laughter_keywords = ["laugh", "laughter", "ha ha", "haha", "lol", "lmao"]

    laughter_segments = []

    for entry in transcript:
        text_lower = entry["text"].lower()
        for keyword in laughter_keywords:
            if keyword in text_lower:
                laughter_segments.append(
                    {"start": entry["start"], "end": entry["end"], "text": entry["text"]}
                )
                break

    return laughter_segments


def extract_audio_features(
    audio: np.ndarray, sample_rate: int = 16000, frame_duration: float = 0.1
) -> List[Dict[str, float]]:
    """Extract audio features for each frame."""
    frame_size = int(frame_duration * sample_rate)
    features = []

    for i in range(0, len(audio), frame_size):
        frame = audio[i : min(i + frame_size, len(audio))]

        if len(frame) > 0:
            energy = np.mean(np.square(frame))
            zero_crossings = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)

            features.append(
                {
                    "timestamp": i / sample_rate,
                    "energy": float(energy),
                    "zero_crossings": float(zero_crossings),
                }
            )

    return features
