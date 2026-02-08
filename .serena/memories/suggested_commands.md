Setup:
- pip install -r requirements.txt
- ffmpeg -version

Core usage:
- python skills/autocut-shorts/scripts/autocut.py video.mp4 --num-clips 5 --platform tiktok
- python skills/video-transcriber/scripts/transcribe.py video.mp4 --model whisper --whisper-model medium
- python skills/subtitle-overlay/scripts/add_subtitles.py video.mp4 --subtitle video.srt
- python skills/portrait-resizer/scripts/resize_to_portrait.py video.mp4

Dev/test:
- ruff check .
- pytest