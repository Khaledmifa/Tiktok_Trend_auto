---
name: tiktok-studio-workflow
description: >-
  End-to-end TikTok 9:16 video production workflow in d:\Money using the unified
  tiktok/<run_id>/ media intake and export standard, chat-guided search terms,
  live B-roll prioritization, local edge-tts + FFmpeg rendering, and strict Visual QA.
  Use whenever creating, updating, reviewing, or exporting a TikTok video in d:\Money.
---

# TikTok Studio Unified Workflow (`tiktok/<run_id>/`)

## Overview
All user-supplied media (photos & video clips) and final approved TikTok export bundles live in the single unified folder `d:\Money\tiktok\<run_id>\`, paired with chat-guided asset requests and local `edge-tts` + `FFmpeg` assembly.

## Steps

### 1. Trend Discovery & 10-Scene Storyboard
1. Discover or receive the target trend:
   - Check the workspace's persistent multi-category curated queue at `d:\Money\data\trends_queue.json` (and `d:\Money\data\trends_queue.md`, covering `games`, `movies`, `series`, `anime`, `it`, `ai`, `technology`), and mark a completed trend `"status": "DONE"` with its `run_id` once approved/exported.
   - Automated discovery: `python run_pipeline.py --auto-top-trend --discover-trend-only --skip-external-llm`
   - Or analyze a user-provided trend screenshot/topic.
2. Create `tiktok/<run_id>/` and author `tiktok/<run_id>/script_plan.json` (`10` scenes, `155–185` English words ONLY, `visual_ai_prompt <= 15` words) and TikTok-only `tiktok/<run_id>/metadata.json`. Any non-English source articles/trends/snippets MUST be translated into English first.
3. Validate with `ScriptArchitectAgent.load_from_json(...)` (enforcing `enforce_english_only: true`).

### 2. Chat-Guided Media Intake (`tiktok/<run_id>/` + Chat)
1. Reply in Arabic with the numbered list of needed media (`scene_01`..`scene_10`) and **exact English search terms** (e.g. with `-getty -shutterstock -alamy` to avoid watermarks).
2. Instruct the user to drop files directly into `d:\Money\tiktok\<run_id>\` (`scene_XX.jpg/png/webp` for photos, `scene_XX.mp4` or `scene_XX_clip.mp4` for videos) or upload them in chat.

### 3. Live Video B-Roll & Smart Framing Preparation
- **Prefer live video clips over static photos** whenever real/trailer footage is available.
- Before trimming widescreen/trailer clips:
  1. Sample frames across the video via `FFmpeg` and inspect with `read_file`.
  2. Pick timestamps with **zero** foreign title cards or burned-in captions.
  3. Crop out letterbox black bars and corner watermarks while trimming muted (`-an`) clips.
- If replacing or re-fitting a scene photo inside `output/<run_id>/photos/`, delete any cached `output/<run_id>/photos/scene_XX_photo.jpg` so `PhotoGeneratorAgent._fit_to_vertical_9x16` re-generates it at full width (`1080px` edge-to-edge for vertical posters, wide uncropped framing above captions for landscape photos).

### 4. Local Render (`edge-tts` + Editorial HUD + `FFmpeg` 48 kHz Master)
Run:
```powershell
python run_pipeline.py --run-id <run_id> --script-json output/<run_id>/script_plan.json --metadata-json output/<run_id>/metadata.json --skip-external-llm
```
The pipeline pulls `scene_XX.*` / `scene_XX_a.mp4` + `scene_XX_b.mp4` clips from `d:\Money\tiktok\<run_id>\` first, composites the Acid-Lime `#CDFF43` Editorial HUD (`VIRAL STUDIO PRO` pill, 2-line headline, wide-panel `1080x608` / portrait alternating cuts, footage attribution, Fact Card, progress indicator), aligns scenes to `word_times.json`, burns pixel-measured 2-line captions, masters audio at `48 kHz` with limiter + LUFS/dBTP verification, and exports `cover.jpg` alongside `final_9x16.mp4`.

### 5. Visual QA & Human Approval Gate
1. Inspect **every** extracted frame in `output/<run_id>/review_frames/scene_01.jpg`..`scene_10.jpg` via `read_file` alongside `frames_manifest.json`, and inspect `loop_qa_transition.mp4` (back-to-back final-to-first seam) to confirm seamless visual/audio continuity and no end-card/fade interruption.
2. Confirm:
   - `1080x1920` dimensions, `duration > 61s`, `qa.passed: true`.
   - Displayed photos/posters are full-sized (never small/postage-stamped), faces/titles uncropped, zero watermarks/foreign text, legible captions in the safe zone.
3. Stop at `READY_FOR_REVIEW`, present the Arabic review package with clickable links, and wait for explicit confirmation.
4. Only after user approval, run:
```powershell
python run_pipeline.py --approve <run_id>
```
This exports `final_9x16.mp4` and `tiktok_post_details.txt` into the same `d:\Money\tiktok\<run_id>\` folder.
