---
name: highres-clip-harvester
description: >-
  Sources high-resolution 1080p video footage via free local yt-dlp and PySceneDetect,
  inspects contact sheets to strip black bars and burned-in text, and stages clean muted
  scene_XX.mp4 clips into d:\Money\tiktok\<run_id>\. Prioritize real footage over static images.
---

# High-Res Clip Harvester (`$0` Local Tools Only)

## Tools Used
- `yt-dlp` (free CLI / Python module) for official trailers, press B-roll, and CC/editorial clips.
- `scenedetect` (`PySceneDetect` + OpenCV) and `FFmpeg` (`cropdetect`) for automatic scene boundary detection and aspect-ratio bar removal.

## Steps
1. Ask the user for confirmation before downloading large video batches if not already approved.
2. Prefer real high-resolution (`4K` / `1080p`) official trailers, keynotes, showcase reels, or clean footage over static AI images.
3. Download candidate source video(s) into `output/<run_id>/raw_trailers/` using `python -m yt_dlp -f "bestvideo[height<=2160][ext=mp4]/bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]/best"`.
4. Detect scene boundaries with PySceneDetect or sample frames every `1–2s` via `FFmpeg`:
   ```powershell
   scenedetect -i <raw_video.mp4> detect-content list-scenes
   ```
5. Compose a labeled contact sheet and inspect via `read_file` **before** cutting so zero title cards, foreign news chyrons, watermarks, or burned-in subtitles slip into any clip.
6. Run `ffmpeg -vf cropdetect=24:16:0` to crop out black letterbox/pillarbox bars, stage clean muted `scene_XX.mp4` (or `scene_XX_a.mp4` / `scene_XX_b.mp4` pairs for 24-cut alternating `wide_panel` + `fullscreen_portrait` framing) directly into `d:\Money\tiktok\<run_id>\` (with fallback JPG frames into `output/<run_id>/photos/`).
7. Clean up temporary `raw_trailers/` and `tmp_frames/` after verifying every `scene_XX.mp4` via `ffprobe`.
