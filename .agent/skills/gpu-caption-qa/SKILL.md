---
name: gpu-caption-qa
description: >-
  Generates bold high-contrast TikTok safe-zone captions from free edge-tts timestamps
  (with optional RTX 5060 faster-whisper alignment), inspects rendered review frames, and
  enforces zero-token ffprobe verification via scripts/verify_video_gate.py.
---

# GPU-Ready Captions & Visual QA

## Free Caption & Editorial HUD Generation
1. Primary (`$0` & zero GPU overhead): `MontageAgent` consumes `WordBoundary` timestamps from `edge-tts` (`src/providers/tts.py`), writes `word_times.json` and `scene_timings.json`, and builds pixel-measured `captions.srt` and `captions.ass` (`Arial Bold`, `57px`, `<= 790px` balanced 2-line wrap, mandatory break on sentence-ending punctuation `.` / `?` / `!`, Acid-Lime `&H0043FFCD` active-word highlight centered near `x=505, y=1300`).
2. Seamless Loop Audio & Back-to-Back QA: `generate_dark_ambient_bed(..., seamless_loop=True)` loops seamlessly without fade-to-silence, `prepare_audio_and_scene_timings` trims trailing dead silence (`loop_tail_pad_sec: 0.08`), and `build_loop_qa_preview` outputs `loop_qa_transition.mp4` for back-to-back visual/audio seam inspection.
3. Clean 3-Layer Editorial HUD (`src/providers/editorial_hud.py`): Renders `1080x1920` dark gradient overlays with `VIRAL STUDIO PRO` `#CDFF43` badge, compact top-right status pill (`OFFICIAL FOOTAGE`, `RELEASED FEATURE`, `ANNOUNCEMENT / DEMO`, `SYNTHETIC / AI-GENERATED`), 2-line headlines, subtle attribution, distinct-only Fact Card, and progress bar, while preserving `48 kHz` mastered audio with limiter + `measure_audio_metrics` (`LUFS`, `dBTP`, `clipped_samples`) and `cover.jpg`.
4. Optional local transcription (`RTX 5060`, `$0`): When aligning external voiceovers lacking `WordBoundary` metadata, use free local `faster-whisper` (`device="cuda"`, `compute_type="float16"`).

## Zero-Token Technical Gate (`ffprobe`)
Run immediately after any `final_9x16.mp4` render (also wired automatically via `.agents/hooks.json` `PostToolUse`):
```powershell
python scripts/verify_video_gate.py output/<run_id>/final_9x16.mp4
```
Confirm exit code `0`, `duration_sec > 61.0`, `1080x1920`, video + audio streams present.

## Multimodal Visual QA Gate (Mandatory Before `READY_FOR_REVIEW`)
1. Open `output/<run_id>/review_frames/frames_manifest.json` and inspect **every** frame (`scene_01.jpg`..`scene_10.jpg`) via `read_file`.
2. Confirm:
   - Full `1080px` width framing for photos/posters and letterbox-cropped B-roll (zero postage-stamp thumbnails, zero black bars).
   - Faces and titles uncropped; zero watermarks, social icons, or burned-in trailer subtitles.
   - Safe-zone captions readable without overlapping key subjects.
3. Verify `output/<run_id>/publish_kit.md` exists, then stop at `READY_FOR_REVIEW` and await explicit human approval.
