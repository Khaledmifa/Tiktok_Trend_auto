# Viral Studio Pro — $0 Local TikTok Studio

Fully local, **$0 external API cost** TikTok vertical video studio (`9:16`, 4K UHD `2160x3840` / `1080x1920`, `30fps`, `> 61s` Creator Rewards floor) controlled by a resilient interactive Telegram Bot with unified `tiktok/<run_id>/` storage, automated quality gates, and future server portability.

## Core Architecture & Key Files
- `run_pipeline.py` — Unified CLI pipeline runner (`--topic`, `--run-id`, `--script-json`, `--revision-feedback`, `--approve <run_id>`).
- `config.yaml` — Centralized configuration for the 4 fixed pillars (`movies`, `series`, `games`, `ai`), Edge TTS monolingual English voice (`en-US-AndrewNeural`), 4K/60fps-ready video spec, Editorial HUD, kinetic subtitles, and Telegram bot behavior.
- `AGENTS.md` — Permanent workspace engineering rules (strict English-only video rule, Clean-Replace Policy, $0 local stack, `PROJECT_ROOT` relative paths, AI escalation rule).
- `src/config.py` — Portable `AppConfig` resolving all paths relative to `PROJECT_ROOT`.
- `src/agents/research_agent.py` — Free RSS, Google Trends, DuckDuckGo, and curated queue trend discovery & similarity deduplication.
- `src/agents/script_agent.py` — 10-scene English-only script & hook builder grounded in live web facts + local Ollama (when available).
- `src/agents/photo_agent.py` — Visual asset staging (`tiktok/<run_id>/`) and watermark-free web image fallback (`9:16` smart framing).
- `src/agents/video_agent.py` — Normalizes live `yt-dlp` B-roll clips or renders camera motion (`zoom_in`, `zoom_out`, `pan_left`, `pan_right`, `whip_zoom`).
- `src/agents/montage_agent.py` — Monolingual Edge TTS (`en-US-AndrewNeural`), pixel-measured kinetic ASS captions, Editorial HUD overlay, `-14 LUFS` 48kHz audio master, seamless loop QA clip, and safe monetization floor (`>= 62.5s` > `61s`).
- `src/agents/reviewer_agent.py` — Frame extraction (`review_frames/`) & automated visual/dimension QA checks.
- `src/agents/publisher_agent.py` — HITL review gate (`READY_FOR_REVIEW`), `publish_kit.md`, and manual TikTok export (`tiktok/<run_id>/`).
- `src/utils/revision_engine.py` — Natural-language `تعديل` parser (scene/time-range deletion & re-merge, voice/language lock, speed control, scene clip swap) + SHA-256 verification + `> 3` edit AI escalation queue (`data/ai_escalation_queue.json`).
- `scripts/auto_media_harvester.py` — Free `yt-dlp` + `ffmpeg cropdetect` 1080p B-roll harvester and DDGS fallback.
- `scripts/telegram_bot.py` & `scripts/send_to_telegram.py` — Resilient single-instance interactive Telegram bot powered by **Telethon MTProto** (`streamable_preview` during review, uncompressed 4K master up to `2 GB` on approval, 4-button dashboard, auto-wipe after `2/2` daily quota, published list in chat, unpublished video + copy text sender, `> 3` edit AI escalation).
- `scripts/verify_video_gate.py` — Zero-token `ffprobe` + English-only subtitle QA gate.
- `scripts/install_bot_autostart.ps1` / `scripts/bot_service_status.ps1` — Silent Windows Task Scheduler `pythonw.exe` service setup and status checks.
- `MIGRATION_GUIDE.md` — Zero-code-rewrite Linux/Windows dedicated server migration instructions.

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Fill TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID inside .env
```

### Verify Suite
```powershell
python -m unittest tests/test_mvp_smoke.py -v
```

### Run Pipeline Manually
```powershell
python run_pipeline.py --topic "Your Trending Topic" --run-id run_custom_001
python scripts/verify_video_gate.py tiktok/run_custom_001/final_9x16.mp4
python run_pipeline.py --approve run_custom_001
```

### Start Silent Background Bot (Windows)
```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_bot_autostart.ps1
powershell -ExecutionPolicy Bypass -File scripts/bot_service_status.ps1
```
