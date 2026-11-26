# Viral Studio Pro — TikTok Studio Workspace Rules (`d:\Money`)

## Communication & Language
- Respond strictly in clear Arabic for all explanations, progress updates, review packages, and questions.
- Keep code, file paths, terminal commands, config keys, and image/video search terms in English.
- **Strict English-Only Video Rule:** ALL produced videos—voiceover narration, burned-in captions/subtitles, on-screen HUD headlines/fact cards/badges, and TikTok publish-kit captions/descriptions/hashtags—MUST be 100% English (`en` only). Even when source materials, news articles, trend feeds, web snippets, or notes are in another language (German, Arabic, French, etc.), they MUST be translated into natural English before generating scripts, voiceovers, captions, HUD overlays, or TikTok publish kits. Never generate, speak, or burn non-English text into any video.

## Runtime Architecture & Future Server Portability
- **Current Runtime:** The entire pipeline and Telegram bot run 100% locally on the user's Windows laptop (HP OMEN, RTX 5060, 32GB RAM) at **$0 extra cost**. Do NOT move or split services onto a low-RAM server right now.
- **Future Server Migration Goal:** Copying the project folder + `.env` and running `pip install -r requirements.txt` on a future Linux or Windows dedicated server MUST work without rewriting code.
- **Mandatory Coding Invariant:** Always resolve `.env`, `config.yaml`, `data/`, `tiktok/`, `assets/`, and `logs/` via `pathlib.Path` relative to `PROJECT_ROOT` regardless of current working directory. Never hardcode `d:\Money`, `C:\`, or OS-specific paths inside Python modules or `config.yaml`. Keep all persistent channel state strictly inside `data/`, all adjustable knobs strictly in `config.yaml`, all secrets strictly in `.env`, all per-run trend files/media/videos/frames/publish kits strictly inside the unified `tiktok/<run_id>/` folder (never `output/`), and Windows-specific auto-start scripts (`*.ps1`, `*.bat`) isolated in `scripts/`.

## Clean-Replace Policy (No Legacy Conflicts)
- Whenever new settings, storage paths, categories, or workflows are agreed upon, immediately delete the superseded old settings, legacy fallbacks, and unused code paths so older configurations never conflict with the active workflow.

## Strict $0 Extra Cost & Hardware Guardrail
- Total external cost MUST stay **$0**: use ONLY free local open-source tools (`Kinocut`/`FFmpeg`, `ffprobe`, `yt-dlp`, `scenedetect`, `edge-tts`, optional RTX 5060 `faster-whisper`) and built-in Antigravity capabilities (subagents, built-in Browser, `search_web`).
- **STRICTLY FORBIDDEN:** Paid APIs, RapidAPI keys, external paid Gemini keys/billing, paid cloud video/image SaaS, paid stock sites, or automated TikTok/YouTube upload APIs.

## Platform, Format & Compliance (TikTok Only — Manual Upload)
- Target TikTok exclusively (**Viral Studio Pro**): vertical `9:16` **4K UHD** (`2160x3840`, `30fps`, with `1080x1920` compatibility), `65–85s` duration (`> 61s` Creator Rewards floor), general adult audience (`18–45`), `10–12` dynamic high-res video segments prioritizing real/trailer B-roll over static slideshows. Deliver Telegram previews uncompressed via `send_as_document: true` (`< 48 MB` copy while keeping full-bitrate 4K master locally).
- Zero plagiarism, zero watermarked/copyrighted clips, zero celebrity deepfakes, and keep AI/synthetic disclosure enabled (`tiktok_disclose_ai_content: true`).
- Every production run must generate `publish_kit.md` and stop cleanly at `READY_FOR_REVIEW`. Never auto-publish or run `--approve` without explicit human approval.

## Token-Efficient Subagent Routing & Zero-Token Hooks
- Route trend hunting, script/cut-list drafting, hashtags, and JSON formatting to fast/lightweight models (`flash`/`flash_lite`); run media/FFmpeg commands locally with zero AI tokens; reserve vision/heavier reasoning models for final visual/caption/audio-sync QA and complex debugging.
- Keep spoken scripts compact (`155–185` EN words, English-only across all videos) and visual prompts `<= 15` words; ask before heavy GPU batches or large video downloads.
- Zero-token validation via `scripts/verify_video_gate.py` runs automatically via `.agents/hooks.json` (`PostToolUse`), while `PreToolUse` blocks any automated public upload command.

## AI Escalation Queue (>3 Edit Rounds)
- After `> 3` `تعديل` edit rounds on the same video, `scripts/telegram_bot.py` stops auto-looping, tries local `$0` Ollama diagnosis if running, and otherwise logs an open ticket in `data/ai_escalation_queue.json` and `tiktok/<run_id>/AI_ESCALATION_TICKET.md` for Antigravity to inspect, fix, re-render, and deliver via `python scripts/telegram_bot.py --deliver-review-run <run_id>`.

## Visual QA Invariant
- Before presenting `READY_FOR_REVIEW`, inspect all rendered frames in `tiktok/<run_id>/review_frames/`.
- Verify every frame matches standard `1080x1920` dimensions, foreground photos/posters fill the frame width cleanly (never postage-stamped or undersized), faces/subjects are uncropped, captions sit cleanly in the safe zone, and no watermarks, social icons, or foreign title cards appear.
