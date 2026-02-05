import json
from pathlib import Path
from typing import Any, Dict, List, Optional


COLOR_MAP = {
    "white": "#FFFFFF",
    "black": "#000000",
    "yellow": "#FFC400",
    "red": "#FF3B30",
    "blue": "#2E86FF",
    "green": "#2ECC71",
    "orange": "#FF8A00",
}


KARAOKE_PRESETS = {
    "tiktok": {
        "font": "Plus Jakarta Sans",
        "font_scale": 0.032,
        "outline_width": 3,
        "margin_v_scale": 0.14,
        "margin_x_scale": 0.06,
        "primary_color": "white",
        "highlight_color": "yellow",
        "outline_color": "black",
    },
    "shorts": {
        "font": "Plus Jakarta Sans",
        "font_scale": 0.03,
        "outline_width": 3,
        "margin_v_scale": 0.14,
        "margin_x_scale": 0.06,
        "primary_color": "white",
        "highlight_color": "yellow",
        "outline_color": "black",
    },
    "reels": {
        "font": "Plus Jakarta Sans",
        "font_scale": 0.03,
        "outline_width": 3,
        "margin_v_scale": 0.14,
        "margin_x_scale": 0.06,
        "primary_color": "white",
        "highlight_color": "yellow",
        "outline_color": "black",
    },
    "default": {
        "font": "Plus Jakarta Sans",
        "font_scale": 0.03,
        "outline_width": 3,
        "margin_v_scale": 0.14,
        "margin_x_scale": 0.06,
        "primary_color": "white",
        "highlight_color": "yellow",
        "outline_color": "black",
    },
}


NO_SPACE_PREFIXES = {
    ",",
    ".",
    "!",
    "?",
    ":",
    ";",
    "%",
    ")",
    "]",
    "}",
    '"',
    "'",
}


def load_word_timestamps(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _extract_words(data)


def slice_words(
    words: List[Dict[str, Any]], clip_start: float, clip_end: float
) -> List[Dict[str, Any]]:
    sliced = []
    for word in words:
        start = float(word.get("start", 0))
        end = float(word.get("end", 0))
        if end <= clip_start or start >= clip_end:
            continue
        adjusted_start = max(0.0, start - clip_start)
        adjusted_end = min(clip_end - clip_start, end - clip_start)
        if adjusted_end <= adjusted_start:
            continue
        sliced.append(
            {
                "word": str(word.get("word", "")).strip(),
                "start": adjusted_start,
                "end": adjusted_end,
            }
        )
    return sliced


def build_karaoke_ass(
    words: List[Dict[str, Any]],
    output_path: str,
    video_width: int,
    video_height: int,
    style: str = "default",
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    primary_color: Optional[str] = None,
    highlight_color: Optional[str] = None,
    outline_color: Optional[str] = None,
    outline_width: Optional[int] = None,
    margin_v: Optional[int] = None,
    max_words: int = 6,
    max_chars: int = 32,
    max_gap: float = 0.8,
    max_duration: float = 4.0,
) -> Optional[str]:
    if not words:
        return None

    lines = _group_words_into_lines(words, max_words, max_chars, max_gap, max_duration)
    if not lines:
        return None

    style_config = _resolve_style(
        style,
        video_width,
        video_height,
        font=font,
        font_size=font_size,
        primary_color=primary_color,
        highlight_color=highlight_color,
        outline_color=outline_color,
        outline_width=outline_width,
        margin_v=margin_v,
    )

    ass_content = _render_ass(lines, style_config, video_width, video_height)
    Path(output_path).write_text(ass_content, encoding="utf-8")
    return output_path


def _extract_words(data: Any) -> List[Dict[str, Any]]:
    words = []

    if isinstance(data, dict):
        data = data.get("segments", [])

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "words" in item:
                for word in item.get("words", []) or []:
                    normalized = _normalize_word_entry(word)
                    if normalized:
                        words.append(normalized)
            elif isinstance(item, dict):
                normalized = _normalize_word_entry(item)
                if normalized:
                    words.append(normalized)

    words.sort(key=lambda w: w.get("start", 0))
    return words


def _normalize_word_entry(word: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(word, dict):
        return None

    if "word" not in word or "start" not in word or "end" not in word:
        return None

    text = str(word.get("word", "")).strip()
    if not text:
        return None

    return {
        "word": text,
        "start": float(word.get("start", 0)),
        "end": float(word.get("end", 0)),
    }


def _resolve_style(
    style: str,
    video_width: int,
    video_height: int,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
    primary_color: Optional[str] = None,
    highlight_color: Optional[str] = None,
    outline_color: Optional[str] = None,
    outline_width: Optional[int] = None,
    margin_v: Optional[int] = None,
) -> Dict[str, Any]:
    preset = KARAOKE_PRESETS.get(style, KARAOKE_PRESETS["default"])

    resolved_font = font or preset["font"]
    resolved_font_size = font_size or max(24, int(video_height * preset["font_scale"]))
    resolved_outline = outline_width or preset["outline_width"]
    resolved_margin_v = margin_v or max(40, int(video_height * preset["margin_v_scale"]))
    resolved_margin_x = max(40, int(video_width * preset["margin_x_scale"]))

    return {
        "font": resolved_font,
        "font_size": resolved_font_size,
        "primary_color": _color_to_ass(primary_color or preset["primary_color"]),
        "highlight_color": _color_to_ass(highlight_color or preset["highlight_color"]),
        "outline_color": _color_to_ass(outline_color or preset["outline_color"]),
        "outline_width": resolved_outline,
        "margin_v": resolved_margin_v,
        "margin_l": resolved_margin_x,
        "margin_r": resolved_margin_x,
    }


def _group_words_into_lines(
    words: List[Dict[str, Any]],
    max_words: int,
    max_chars: int,
    max_gap: float,
    max_duration: float,
) -> List[Dict[str, Any]]:
    lines = []
    current_words = []
    current_chars = 0
    line_start = None
    last_end = None

    for word in words:
        text = str(word.get("word", "")).strip()
        if not text:
            continue

        start = float(word.get("start", 0))
        end = float(word.get("end", 0))

        if line_start is None:
            line_start = start

        gap = start - (last_end if last_end is not None else start)
        if current_words and gap > max_gap:
            lines.append({"start": line_start, "end": last_end, "words": current_words})
            current_words = []
            current_chars = 0
            line_start = start

        needs_space = _needs_space(text) if current_words else False
        projected_chars = current_chars + len(text) + (1 if needs_space else 0)
        projected_duration = end - line_start

        if current_words and (
            len(current_words) >= max_words
            or projected_chars > max_chars
            or projected_duration > max_duration
        ):
            lines.append({"start": line_start, "end": last_end, "words": current_words})
            current_words = []
            current_chars = 0
            line_start = start

        if current_words and _needs_space(text):
            current_chars += 1

        current_words.append({"word": text, "start": start, "end": end})
        current_chars += len(text)
        last_end = end

    if current_words:
        lines.append({"start": line_start, "end": last_end, "words": current_words})

    return lines


def _needs_space(word: str) -> bool:
    return word[0] not in NO_SPACE_PREFIXES


def _render_ass(lines: List[Dict[str, Any]], style: Dict[str, Any], width: int, height: int) -> str:
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Karaoke,{font},{font_size},{primary_color},{highlight_color},{outline_color},"
            "&H00000000,0,0,0,0,100,100,0,0,1,{outline_width},0,2,{margin_l},"
            "{margin_r},{margin_v},1"
        ).format(**style),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events = []
    for line in lines:
        start = _format_ass_time(float(line["start"]))
        end = _format_ass_time(float(line["end"]))
        text = _build_karaoke_text(line["words"], float(line["end"]))
        if not text:
            continue
        events.append(f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}")

    return "\n".join(header + events) + "\n"


def _build_karaoke_text(words: List[Dict[str, Any]], line_end: float) -> str:
    parts = []
    for index, word in enumerate(words):
        token = str(word.get("word", "")).strip()
        if not token:
            continue

        next_start = line_end
        if index + 1 < len(words):
            next_start = float(words[index + 1].get("start", line_end))

        duration = max(0.01, next_start - float(word.get("start", 0)))
        duration_cs = max(1, int(round(duration * 100)))

        if parts and _needs_space(token):
            token = " " + token

        parts.append(f"{{\\kf{duration_cs}}}{token}")

    return "".join(parts)


def _format_ass_time(seconds: float) -> str:
    total_cs = int(round(seconds * 100))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centiseconds = total_cs % 100
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _color_to_ass(color: str) -> str:
    if not color:
        return "&H00FFFFFF"
    if color.startswith("&H"):
        return color

    value = color.lower()
    value = COLOR_MAP.get(value, value)
    if value.startswith("#"):
        value = value[1:]

    if len(value) != 6:
        return "&H00FFFFFF"

    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"&H00{blue:02X}{green:02X}{red:02X}"
