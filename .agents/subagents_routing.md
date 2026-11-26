# Viral Studio Pro — Token-Efficient Subagent Routing

To protect subscription quota and enforce **$0 external cost**, every workflow stage routes to the lowest-cost tier capable of completing it accurately:

| Subagent / Executor | Model Tier (`invoke_subagent`) | Owned Tasks |
|---|---|---|
| `tiktok-trend-hunter` | `flash` / `flash_lite` | Browser-based scanning (2–3 public pages max via `browser` subagent on `flash`), DOM text extraction, deduplication via `scripts/trend_history_manager.py`, and drafting `output/latest_trends.md` in Arabic. |
| `trend-script-fast` | `flash` / `flash_lite` | Short 10–12 scene script drafting, cut-lists, TikTok captions/hashtags, `publish_kit.md` copy, and JSON formatting. |
| `local-media-runner` | `flash_lite` + Local OS Tools (`$0` tokens) | `yt-dlp` downloads, `scenedetect` scene splitting, `edge-tts` narration, `Kinocut`/`FFmpeg` rendering, `scripts/verify_video_gate.py`. |
| `visual-qa-auditor` | `inherit` (Vision / Heavier reasoning ONLY) | Final frame-by-frame visual/caption/audio-sync audit (`output/<run_id>/review_frames/`) and root-cause debugging. |

## Strict Rules
- Never invoke heavier reasoning models for JSON formatting, file moves, simple browser DOM extraction, or caption hashtag phrasing.
- Never solve or retry CAPTCHAs/login walls: skip immediately to the next public URL or `search_web`.
- Verification of MP4 duration (> 61s), `1080x1920` geometry, and stream presence happens via local `ffprobe` (`scripts/verify_video_gate.py`) with **zero AI tokens**.
