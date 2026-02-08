import subprocess
import json
from typing import Optional, List, Dict, Any
from pathlib import Path


class FFmpegWrapper:
    """Wrapper for FFmpeg operations with error handling and logging."""

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe."""
        cmd = [
            self.ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get video info: {result.stderr}")
        return json.loads(result.stdout)

    def get_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        info = self.get_video_info(video_path)
        return float(info["format"]["duration"])

    def get_resolution(self, video_path: str) -> tuple[int, int]:
        """Get video resolution (width, height)."""
        info = self.get_video_info(video_path)
        video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        if not video_stream:
            raise ValueError("No video stream found")
        return int(video_stream["width"]), int(video_stream["height"])

    def _get_audio_codec(self, video_path: str) -> Optional[str]:
        """Get audio codec name if available."""
        try:
            info = self.get_video_info(video_path)
            audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
            if not audio_stream:
                return None
            return audio_stream.get("codec_name")
        except Exception:
            return None

    def _audio_args(self, video_path: str) -> List[str]:
        """Choose audio encoding args for compatibility."""
        codec = self._get_audio_codec(video_path)
        if codec == "aac":
            return ["-c:a", "copy"]
        if codec:
            return ["-c:a", "aac", "-b:a", "128k"]
        return ["-c:a", "aac", "-b:a", "128k"]

    def extract_audio(self, video_path: str, output_path: str, sample_rate: int = 16000) -> bool:
        """Extract audio from video."""
        cmd = [
            self.ffmpeg_path,
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-y",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def trim_video(
        self,
        input_path: str,
        output_path: str,
        start_time: float,
        end_time: float,
        reencode: bool = False,
        vcodec: Optional[str] = None,
        acodec: Optional[str] = None,
    ) -> bool:
        """Trim video to specified time range."""
        duration = end_time - start_time

        cmd = [
            self.ffmpeg_path,
            "-i",
            input_path,
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-y",
        ]

        if reencode:
            cmd.extend(["-c:v", vcodec or "libx264", "-preset", "fast", "-crf", "23"])
            cmd.extend(["-c:a", acodec or "aac"])
        else:
            cmd.extend(["-c", "copy"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        return True

    def resize_video(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        crop_mode: str = "center",
        fill_color: str = "black",
    ) -> bool:
        """Resize video to target dimensions."""
        input_width, input_height = self.get_resolution(input_path)
        input_aspect = input_width / input_height
        target_aspect = width / height

        cmd = [self.ffmpeg_path, "-i", input_path]

        if input_aspect > target_aspect:
            new_width = width
            new_height = int(width / input_aspect)
            crop_filter = f"crop={new_width}:{new_height}:0:{(height - new_height) // 2}"
        else:
            new_width = int(height * input_aspect)
            new_height = height
            crop_filter = f"crop={new_width}:{new_height}:{(width - new_width) // 2}:0"

        cmd.extend(
            [
                "-vf",
                f"scale={new_width}:{new_height},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:{fill_color}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                *self._audio_args(input_path),
                "-y",
                output_path,
            ]
        )

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def concat_videos(
        self,
        input_paths: List[str],
        output_path: str,
        reencode: bool = False,
    ) -> bool:
        """Concatenate multiple videos."""
        concat_list = Path("concat_list.txt")
        try:
            with open(concat_list, "w") as f:
                for path in input_paths:
                    f.write(f"file '{path}'\n")

            cmd = [
                self.ffmpeg_path,
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy" if not reencode else "libx264",
                "-y",
                output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        finally:
            if concat_list.exists():
                concat_list.unlink()

    def add_subtitle(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        font_name: str = "Plus Jakarta Sans",
        font_size: int = 24,
        font_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 1.5,
        position: str = "bottom",
        force_style: bool = True,
    ) -> bool:
        """Add subtitle overlay to video."""
        # Get video dimensions to calculate appropriate font size
        try:
            width, height = self.get_resolution(video_path)
            is_portrait = height > width
        except:
            is_portrait = False
            height = 1920

        # Adjust font size for portrait videos (smaller relative to screen)
        # Portrait videos need smaller fonts since width is limited
        if is_portrait:
            # Scale font size based on width (1080p portrait = use smaller fonts)
            adjusted_font_size = min(font_size, int(height * 0.04))
        else:
            adjusted_font_size = font_size

        # ASS color format is &HAABBGGRR (alpha, blue, green, red)
        color_map = {
            "white": "&H00FFFFFF",
            "black": "&H00000000",
            "yellow": "&H0000FFFF",
            "red": "&H000000FF",
            "blue": "&H00FF0000",
            "green": "&H0000FF00",
            "orange": "&H00004080",  # Dark orange/burnt orange in BGR format
        }

        primary_color = color_map.get(font_color.lower(), "&H00FFFFFF")
        outline_col = color_map.get(outline_color.lower(), "&H00000000")

        # MarginV controls vertical margin from bottom (or top if Alignment changes)
        # Alignment=2 means bottom-center
        # Increased MarginV to push subtitles further up from the very bottom
        margin_v = 80 if is_portrait else 40

        # Escape the subtitle path for FFmpeg
        escaped_subtitle_path = subtitle_path.replace("'", r"'\''").replace(":", r"\:")

        # Get absolute path to fonts directory
        fonts_dir = str(Path(__file__).parent.parent.parent / "fonts")

        if force_style:
            subtitle_filter = (
                f"subtitles='{escaped_subtitle_path}':fontsdir='{fonts_dir}':force_style='"
                f"Fontname={font_name},"
                f"FontSize={adjusted_font_size},"
                f"PrimaryColour={primary_color},"
                f"OutlineColour={outline_col},"
                f"BorderStyle=1,"
                f"Outline={outline_width},"
                f"Shadow=0,"
                f"Alignment=2,"
                f"MarginV={margin_v}'"
            )
        else:
            subtitle_filter = f"subtitles='{escaped_subtitle_path}':fontsdir='{fonts_dir}'"

        cmd = [
            self.ffmpeg_path,
            "-i",
            video_path,
            "-vf",
            subtitle_filter,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            *self._audio_args(video_path),
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def detect_scenes(self, video_path: str, threshold: float = 0.3) -> List[Dict[str, float]]:
        """Detect scene changes in video using FFmpeg."""
        cmd = [
            self.ffmpeg_path,
            "-i",
            video_path,
            "-vf",
            f"select='gt(scene,{threshold}),showinfo'",
            "-f",
            "null",
            "-",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
        scenes = []

        for line in result.stdout.split("\n"):
            if "pts_time:" in line:
                time_str = line.split("pts_time:")[1].split()[0]
                scenes.append({"timestamp": float(time_str)})

        return scenes

    def crop_to_portrait(
        self,
        input_path: str,
        output_path: str,
        target_width: int = 1080,
        target_height: int = 1920,
        focus_point: Optional[tuple] = None,
    ) -> bool:
        """Crop video to portrait 9:16 format with smart focus."""
        input_width, input_height = self.get_resolution(input_path)

        if focus_point:
            fx, fy = focus_point
        else:
            fx, fy = input_width / 2, input_height / 2

        target_aspect = target_width / target_height  # 0.5625 for 9:16

        # Calculate crop dimensions to match target aspect ratio
        # We want to crop from the input to get the target aspect ratio
        crop_height = input_height
        crop_width = int(input_height * target_aspect)

        # If calculated crop width is larger than input width, adjust
        if crop_width > input_width:
            crop_width = input_width
            crop_height = int(input_width / target_aspect)

        # Calculate crop position centered on focus point
        crop_x = max(0, int(fx - crop_width / 2))
        crop_y = max(0, int(fy - crop_height / 2))

        # Ensure crop stays within bounds
        crop_x = min(crop_x, input_width - crop_width)
        crop_y = min(crop_y, input_height - crop_height)

        # Ensure non-negative values
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)

        cmd = [
            self.ffmpeg_path,
            "-i",
            input_path,
            "-vf",
            f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width}:{target_height}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            *self._audio_args(input_path),
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
