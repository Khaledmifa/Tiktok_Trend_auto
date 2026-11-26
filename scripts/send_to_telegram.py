from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from PIL import Image
from telethon import TelegramClient
from telethon.sessions import MemorySession
from telethon.tl.types import DocumentAttributeVideo

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.verify_video_gate import verify_video
from src.config import PROJECT_ROOT, AppConfig, load_config

DEFAULT_BOT_SESSION_PATH = PROJECT_ROOT / "data" / "viral_studio_bot.session"
DEFAULT_SENDER_SESSION_PATH = PROJECT_ROOT / "data" / "viral_studio_sender.session"
MTPROTO_MAX_UPLOAD_MB = 1950.0


class MissingTelegramCredentialsError(RuntimeError):
    """Raised when required Telethon MTProto credentials are missing or invalid in .env."""


@dataclass(frozen=True)
class TelegramCredentials:
    api_id: int
    api_hash: str
    bot_token: str
    chat_id: int


def mask_secret(text: str, *secrets: str) -> str:
    """Never allow Telegram bot tokens or API hashes to leak into logs or error messages."""
    masked = text
    for idx, secret in enumerate(secrets):
        if not secret:
            continue
        placeholder = "***TELEGRAM_TOKEN***" if idx == 0 else "***TELEGRAM_SECRET***"
        masked = masked.replace(secret, placeholder)
    return masked


def is_cryptg_available() -> bool:
    """Check whether optional C-accelerated cryptg is installed for faster MTProto uploads."""
    try:
        import cryptg  # noqa: F401

        return True
    except Exception:
        return False


def load_telethon_credentials(env_file: Path | None = None) -> TelegramCredentials:
    """Load TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from .env."""
    target_env = env_file or (PROJECT_ROOT / ".env")
    if target_env.exists():
        load_dotenv(dotenv_path=target_env, override=False)

    api_id_raw = os.getenv("TELEGRAM_API_ID", "").strip()
    api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    missing: list[str] = []
    if not api_id_raw:
        missing.append("TELEGRAM_API_ID")
    if not api_hash:
        missing.append("TELEGRAM_API_HASH")
    if not bot_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not chat_id_raw:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        missing_str = "، ".join(missing)
        raise MissingTelegramCredentialsError(
            f"⚠️ تنبيه Telegram (Telethon MTProto): البيانات التالية ناقصة في ملف البيئة ({target_env}): {missing_str}.\n"
            "• احصل على TELEGRAM_API_ID و TELEGRAM_API_HASH مجاناً من https://my.telegram.org (قسم API development tools).\n"
            "• واحصل على TELEGRAM_BOT_TOKEN من @BotFather ثم أضفها إلى ملف .env وأعد تشغيل البوت."
        )

    try:
        api_id = int(api_id_raw)
    except ValueError as exc:
        raise MissingTelegramCredentialsError(
            f"⚠️ قيمة TELEGRAM_API_ID في ({target_env}) يجب أن تكون رقماً صحيحاً (عدد صحيح)."
        ) from exc

    try:
        chat_id = int(chat_id_raw)
    except ValueError as exc:
        raise MissingTelegramCredentialsError(
            f"⚠️ قيمة TELEGRAM_CHAT_ID في ({target_env}) يجب أن تكون رقماً صحيحاً."
        ) from exc

    return TelegramCredentials(
        api_id=api_id,
        api_hash=api_hash,
        bot_token=bot_token,
        chat_id=chat_id,
    )


def should_send_raw_on_review(tg_cfg: dict[str, Any]) -> bool:
    """Return True when review stage should immediately send the raw uncompressed file as document."""
    if bool(tg_cfg.get("send_raw_on_review", False)):
        return True
    mode = str(tg_cfg.get("review_mode", "streamable_preview")).strip().lower()
    return mode in ("raw", "raw_document", "document", "uncompressed")


def build_video_attributes(
    width: int,
    height: int,
    duration_sec: float,
    supports_streaming: bool = True,
) -> list[DocumentAttributeVideo]:
    """Build MTProto video attributes preserving exact 9:16 dimensions and mobile streaming."""
    return [
        DocumentAttributeVideo(
            duration=int(round(max(1.0, float(duration_sec)))),
            w=int(width),
            h=int(height),
            supports_streaming=bool(supports_streaming),
        )
    ]


def make_upload_progress_callback(
    file_path: Path,
    target_logger: logging.Logger | None = None,
    threshold_mb: float = 50.0,
    step_percent: int = 10,
) -> Callable[[int, int], None] | None:
    """Log periodic upload progress to telegram_bot.log for large (>50 MB) MTProto uploads."""
    if not file_path.exists():
        return None
    size_bytes = file_path.stat().st_size
    if size_bytes <= int(float(threshold_mb) * 1024 * 1024):
        return None

    log = target_logger or logging.getLogger("telegram_bot")
    started = time.monotonic()
    state = {"last_milestone": -1}
    step = max(1, min(50, int(step_percent)))
    file_name = file_path.name

    def _progress(sent_bytes: int, total_bytes: int) -> None:
        total = max(1, int(total_bytes or size_bytes))
        sent = max(0, int(sent_bytes))
        pct = min(100, int((sent * 100) / total))
        milestone = 100 if sent >= total else (pct // step) * step
        if milestone <= state["last_milestone"]:
            return
        state["last_milestone"] = milestone
        elapsed = max(0.05, time.monotonic() - started)
        sent_mb = sent / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        speed_mbps = sent_mb / elapsed
        log.info(
            "MTProto upload progress [%s]: %d%% (%.1f / %.1f MB, %.2f MB/s)",
            file_name,
            pct,
            sent_mb,
            total_mb,
            speed_mbps,
        )

    return _progress


def prepare_small_thumbnail(cover_path: Path | None) -> Path | None:
    """Create a small <=320px JPEG thumbnail for Telegram video/document previews."""
    if not cover_path or not cover_path.exists():
        return None
    try:
        thumb_path = cover_path.with_name(f"{cover_path.stem}_tg_thumb.jpg")
        if (
            thumb_path.exists()
            and thumb_path.stat().st_mtime >= cover_path.stat().st_mtime
            and thumb_path.stat().st_size > 500
        ):
            return thumb_path
        with Image.open(cover_path) as im:
            rgb = im.convert("RGB")
            rgb.thumbnail((320, 320))
            rgb.save(thumb_path, format="JPEG", quality=82, optimize=True)
        return thumb_path if thumb_path.exists() else None
    except Exception:
        return None


def create_telethon_client(
    creds: TelegramCredentials,
    session_path: Path | None = None,
    *,
    connection_retries: int = 5,
    retry_delay: int = 2,
    use_memory_fallback: bool = True,
) -> TelegramClient:
    """Instantiate a portable Telethon TelegramClient stored under data/ (with lock fallback)."""
    kwargs: dict[str, Any] = {
        "connection_retries": connection_retries,
        "retry_delay": retry_delay,
        "auto_reconnect": True,
        "flood_sleep_threshold": 60,
    }
    if session_path is None:
        return TelegramClient(MemorySession(), creds.api_id, creds.api_hash, **kwargs)

    session_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return TelegramClient(str(session_path), creds.api_id, creds.api_hash, **kwargs)
    except sqlite3.OperationalError:
        if use_memory_fallback:
            return TelegramClient(MemorySession(), creds.api_id, creds.api_hash, **kwargs)
        raise


def _build_copy_paste_message_html(
    topic: str,
    caption: str,
    hashtags: list[str],
    disclose_ai: bool,
    lang: str,
    duration_sec: float,
    resolution_label: str = "2160x3840 (4K)",
    master_note: str = "",
) -> tuple[str, str]:
    tags_clean = [f"#{str(h).lstrip('#')}" for h in hashtags[:5] if str(h).strip()]
    tags_str = " ".join(tags_clean) if tags_clean else "#learnontiktok #trending #explained"
    full_copy_text = f"{caption.strip()}\n\n{tags_str}".strip()
    ai_note = (
        "مفعل (ON) — يُرجى تفعيل خيار AI-generated content عند النشر على TikTok."
        if disclose_ai
        else "غير مفعل (OFF)."
    )

    cap_lines = [
        f"🎬 فيديو جاهز للمراجعة والنشر اليدوي ({resolution_label})",
        f"📌 الموضوع: {topic}",
        f"🌐 اللغة: {lang.upper()} | ⏱️ المدة: {duration_sec:.2f}s",
    ]
    if master_note:
        cap_lines.append(master_note)
    video_caption = "\n".join(cap_lines)[:1000]

    msg_html = "\n".join(
        [
            "📋 <b>بيانات النشر الجاهزة للنسخ (اضغط مطولاً للنسخ):</b>",
            f"<pre>{html.escape(full_copy_text)}</pre>",
            f"🤖 <b>تنبيه المحتوى بالذكاء الاصطناعي:</b> {html.escape(ai_note)}",
        ]
    )
    return video_caption, msg_html


def _ensure_publish_kit(run_dir: Path, review_data: dict[str, Any]) -> Path | None:
    for name in ("publish_kit.md", "publish_kit.txt", "tiktok_post_details.txt"):
        cand = run_dir / name
        if cand.exists():
            return cand
    if not review_data:
        return None
    meta = review_data.get("metadata", {})
    tags_str = " ".join(f"#{str(h).lstrip('#')}" for h in meta.get("tiktok_hashtags", [])[:5])
    kit_path = run_dir / "publish_kit.md"
    kit_path.write_text(
        "\n".join(
            [
                f"# Viral Studio Pro — TikTok Publish Kit ({review_data.get('run_id', run_dir.name)})",
                "",
                f"- **Title:** {meta.get('tiktok_title', review_data.get('topic', ''))}",
                f"- **Duration:** `{review_data.get('duration_sec', 0):.2f}s`",
                f"- **AI Disclosure Toggle:** `{'ON' if meta.get('tiktok_disclose_ai_content', True) else 'OFF'}`",
                "",
                "## Copy-Paste Hook Caption & Hashtags",
                "```text",
                f"{meta.get('tiktok_caption', '')}\n\n{tags_str}".strip(),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    return kit_path


async def _deliver_run_async(
    creds: TelegramCredentials,
    video_path: Path,
    video_caption: str,
    copy_msg_html: str,
    publish_kit_path: Path | None,
    *,
    force_document: bool,
    width: int,
    height: int,
    duration_sec: float,
    thumb_path: Path | None = None,
    progress_threshold_mb: float = 50.0,
    session_path: Path = DEFAULT_SENDER_SESSION_PATH,
) -> None:
    log = logging.getLogger("telegram_bot")
    client = create_telethon_client(
        creds,
        session_path=session_path,
        connection_retries=5,
        retry_delay=2,
        use_memory_fallback=True,
    )
    try:
        await client.start(bot_token=creds.bot_token)
        progress_cb = make_upload_progress_callback(
            video_path,
            target_logger=log,
            threshold_mb=progress_threshold_mb,
        )
        small_thumb = prepare_small_thumbnail(thumb_path)
        await client.send_file(
            creds.chat_id,
            str(video_path),
            caption=video_caption,
            force_document=force_document,
            supports_streaming=(not force_document),
            attributes=None if force_document else build_video_attributes(width, height, duration_sec, True),
            thumb=str(small_thumb) if small_thumb else None,
            progress_callback=progress_cb,
        )
        if publish_kit_path and publish_kit_path.exists():
            await client.send_file(
                creds.chat_id,
                str(publish_kit_path),
                caption=f"📎 ملف تفاصيل النشر: {publish_kit_path.name}",
                force_document=True,
            )
        await client.send_message(
            creds.chat_id,
            copy_msg_html,
            parse_mode="html",
            link_preview=False,
        )
    finally:
        await client.disconnect()


async def _deliver_trends_async(
    creds: TelegramCredentials,
    trends_path: Path,
    preview: str,
    session_path: Path = DEFAULT_SENDER_SESSION_PATH,
) -> None:
    client = create_telethon_client(
        creds,
        session_path=session_path,
        connection_retries=5,
        retry_delay=2,
        use_memory_fallback=True,
    )
    try:
        await client.start(bot_token=creds.bot_token)
        await client.send_message(creds.chat_id, preview, link_preview=False)
        await client.send_file(
            creds.chat_id,
            str(trends_path),
            caption="📈 تقرير الترندات الكامل (latest_trends.md)",
            force_document=True,
        )
    finally:
        await client.disconnect()


def deliver_run_to_telegram(
    out_dir: Path,
    lang: str | None = None,
    config: AppConfig | None = None,
    send_as_document_override: bool | None = None,
    dry_run: bool = False,
    graceful: bool = True,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Deliver raw uncompressed final_9x16.mp4 (up to 2 GB) via Telethon MTProto."""
    cfg = config or load_config(WORKSPACE_ROOT / "config.yaml")
    tg_cfg = cfg.telegram
    if not tg_cfg.get("enabled", True):
        return {"ok": False, "skipped": True, "reason": "telegram.enabled=false"}

    target_dir = out_dir / lang.strip().lower() if lang and (out_dir / lang.strip().lower()).is_dir() else out_dir
    video_path = target_dir / "final_9x16.mp4"
    min_sec = float(cfg.video.get("hard_monetization_floor_sec", 61))
    gate = verify_video(video_path, min_sec=min_sec)
    if not gate["passed"]:
        return {
            "ok": False,
            "skipped": True,
            "reason": f"Video gate failed: {'; '.join(gate.get('issues', []))}",
            "gate": gate,
        }

    file_size_bytes = video_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    max_upload_mb = float(tg_cfg.get("max_upload_mb", MTPROTO_MAX_UPLOAD_MB))
    if file_size_bytes > int(max_upload_mb * 1024 * 1024):
        return {
            "ok": False,
            "skipped": True,
            "reason": f"Video size {file_size_mb:.1f} MB exceeds MTProto max_upload_mb ({max_upload_mb:.0f} MB)",
            "gate": gate,
        }

    review_json_path = target_dir / "review_package.json"
    review_data = json.loads(review_json_path.read_text(encoding="utf-8")) if review_json_path.exists() else {}
    meta = review_data.get("metadata", {})
    topic = str(review_data.get("topic") or target_dir.name)
    caption = str(meta.get("tiktok_caption") or topic)
    hashtags = list(meta.get("tiktok_hashtags") or ["learnontiktok", "trending", "explained"])
    disclose_ai = bool(meta.get("tiktok_disclose_ai_content", True))
    resolved_lang = lang or str(cfg.project.get("language", "en"))

    publish_kit_path = _ensure_publish_kit(target_dir, review_data)
    send_as_doc = (
        bool(send_as_document_override)
        if send_as_document_override is not None
        else bool(tg_cfg.get("send_as_document", True))
    )

    gw = int(gate.get("width", 1080))
    gh = int(gate.get("height", 1920))
    res_label = f"{gw}x{gh} (4K)" if gw >= 2160 else f"{gw}x{gh}"
    master_note = (
        f"📦 النسخة الأصلية الكاملة غير المضغوطة ({file_size_mb:.1f} MB — MTProto حتى 2GB)"
        if send_as_doc
        else f"▶️ نسخة قابلة للتشغيل المباشر ({file_size_mb:.1f} MB)"
    )
    video_caption, copy_msg_html = _build_copy_paste_message_html(
        topic=topic,
        caption=caption,
        hashtags=hashtags,
        disclose_ai=disclose_ai,
        lang=resolved_lang,
        duration_sec=float(gate["duration_sec"]),
        resolution_label=res_label,
        master_note=master_note,
    )

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "transport": "telethon_mtproto",
            "method": "sendDocument" if send_as_doc else "sendVideo",
            "force_document": send_as_doc,
            "delivery_video": str(video_path),
            "file_size_mb": file_size_mb,
            "publish_kit": str(publish_kit_path) if publish_kit_path else None,
            "video_caption": video_caption,
            "copy_message_html": copy_msg_html,
            "gate": gate,
        }

    target_env_path = env_file or (PROJECT_ROOT / ".env")
    try:
        creds = load_telethon_credentials(env_file=target_env_path)
    except MissingTelegramCredentialsError as exc:
        reminder = str(exc)
        print(reminder)
        if graceful:
            return {"ok": False, "skipped": True, "reason": "missing_credentials", "message_ar": reminder}
        raise

    progress_threshold_mb = float(tg_cfg.get("progress_log_threshold_mb", 50.0))
    asyncio.run(
        _deliver_run_async(
            creds=creds,
            video_path=video_path,
            video_caption=video_caption,
            copy_msg_html=copy_msg_html,
            publish_kit_path=publish_kit_path,
            force_document=send_as_doc,
            width=gw,
            height=gh,
            duration_sec=float(gate["duration_sec"]),
            thumb_path=target_dir / "cover.jpg",
            progress_threshold_mb=progress_threshold_mb,
        )
    )

    return {
        "ok": True,
        "transport": "telethon_mtproto",
        "method": "sendDocument" if send_as_doc else "sendVideo",
        "force_document": send_as_doc,
        "delivery_video": str(video_path),
        "file_size_mb": file_size_mb,
        "publish_kit": str(publish_kit_path) if publish_kit_path else None,
        "gate": gate,
    }


def deliver_trends_to_telegram(
    trends_path: Path = WORKSPACE_ROOT / "data" / "latest_trends.md",
    dry_run: bool = False,
    graceful: bool = True,
    env_file: Path | None = None,
) -> dict[str, Any]:
    if not trends_path.exists():
        return {"ok": False, "skipped": True, "reason": f"Trends file missing: {trends_path}"}

    content = trends_path.read_text(encoding="utf-8").strip()
    preview = content[:3800] + ("\n\n...(مرفق الملف الكامل أدناه)" if len(content) > 3800 else "")

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "transport": "telethon_mtproto",
            "trends_path": str(trends_path),
            "preview_len": len(preview),
            "preview_head": preview[:240],
        }

    target_env_path = env_file or (PROJECT_ROOT / ".env")
    try:
        creds = load_telethon_credentials(env_file=target_env_path)
    except MissingTelegramCredentialsError as exc:
        reminder = str(exc)
        print(reminder)
        if graceful:
            return {"ok": False, "skipped": True, "reason": "missing_credentials", "message_ar": reminder}
        raise

    asyncio.run(_deliver_trends_async(creds=creds, trends_path=trends_path, preview=preview))
    return {"ok": True, "transport": "telethon_mtproto", "trends_path": str(trends_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Free Telethon (MTProto) 2GB Telegram delivery for Viral Studio Pro.")
    parser.add_argument("run_dir", nargs="?", default=None, help="Run output directory (e.g. tiktok/run_xxx)")
    parser.add_argument("--lang", default=None, help="Optional localized subfolder/language code (en)")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--trends", action="store_true", help="Send data/latest_trends.md summary to Telegram")
    parser.add_argument("--trends-file", default=str(WORKSPACE_ROOT / "data" / "latest_trends.md"))
    parser.add_argument("--as-document", action="store_true", help="Override config and send raw uncompressed document")
    parser.add_argument("--as-video", action="store_true", help="Override config and send streamable video")
    parser.add_argument("--dry-run", action="store_true", help="Validate & build payloads without calling Telegram")
    parser.add_argument("--graceful", action="store_true", help="Exit 0 with Arabic reminder if credentials are missing")
    args = parser.parse_args()

    if args.trends:
        res = deliver_trends_to_telegram(
            trends_path=Path(args.trends_file),
            dry_run=args.dry_run,
            graceful=args.graceful,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if (res.get("ok") or args.graceful) else 1

    if not args.run_dir:
        parser.error("run_dir is required unless --trends is set")

    doc_override: bool | None = True if args.as_document else (False if args.as_video else None)
    cfg = load_config(args.config)
    res = deliver_run_to_telegram(
        out_dir=Path(args.run_dir),
        lang=args.lang,
        config=cfg,
        send_as_document_override=doc_override,
        dry_run=args.dry_run,
        graceful=args.graceful,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if (res.get("ok") or args.graceful) else 1


if __name__ == "__main__":
    sys.exit(main())
