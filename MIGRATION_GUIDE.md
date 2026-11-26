# Viral Studio Pro — Server Migration Guide

This project is engineered for **zero-code-rewrite portability** between your local Windows laptop (HP OMEN RTX 5060) and any future Linux or Windows dedicated server.

## 1. Exact Files & Folders to Copy
Copy these items to your new server folder (for example `/opt/viral-studio-pro` on Linux or `D:\ViralStudioPro` on Windows):

| Item | Purpose |
|---|---|
| `run_pipeline.py`, `src/`, `scripts/` | Core video pipeline, Telegram bot, quota & QA gates |
| `config.yaml` | All channel knobs (7 fixed categories, daily quota, encoder prefs, TTS voices) |
| `requirements.txt` | Full Python dependency manifest |
| `.env` *(transfer securely)* | `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, optional keys |
| `data/` | **100% of persistent channel memory:** `trends_history.json`, `telegram_cards_state.json`, `daily_run_state.json`, `trends_queue.json`, `pipeline_memory.db`, `history_log.json`, `viral_studio_bot.session` |
| `assets/`, `.agent/`, `.agents/` | Fonts/music, custom skills, subagent routing, zero-token hooks |

*(Optional: you can skip copying old `output/` and `logs/` unless archiving past renders.)*

## 2. Recreate Environment in One Command
### Linux (Ubuntu / Debian)
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-venv python3-pip fonts-dejavu-core
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Windows Server / PC
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
*(Ensure `ffmpeg` and `ffprobe` are installed and on `PATH`, or set `FFMPEG_PATH` in `.env`.)*

## 3. Pre-Flight Verification
Run from inside or outside the project folder:
```bash
python scripts/check_daily_quota.py
python -c "import scripts.telegram_bot as tb; from src.config import load_config; cfg=load_config('config.yaml'); print(cfg.project['channel_name'], len(cfg.telegram['categories']))"
```

## 4. Running the Background Telegram Bot
- **Windows:** Run `powershell -ExecutionPolicy Bypass -File scripts/install_bot_autostart.ps1` (registers Task Scheduler `ViralStudioPro_TelegramBot` silently via `pythonw.exe`).
- **Linux (`systemd`):**
```ini
[Unit]
Description=Viral Studio Pro Telegram Bot
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/viral-studio-pro
ExecStart=/opt/viral-studio-pro/.venv/bin/python /opt/viral-studio-pro/scripts/telegram_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
The Python single-instance lock automatically uses Windows Named Mutex on Windows and `fcntl.flock` on Linux, and Telethon (MTProto) delivers uncompressed master videos up to 2 GB directly without 50 MB Bot API down-compression.
