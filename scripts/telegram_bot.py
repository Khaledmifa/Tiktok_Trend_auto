from __future__ import annotations

import argparse
import asyncio
import atexit
import ctypes
import html
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from telethon import Button, TelegramClient, errors as tg_errors, events
from telethon.tl.types import ReplyInlineMarkup

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.send_to_telegram import (
    DEFAULT_BOT_SESSION_PATH,
    DEFAULT_SENDER_SESSION_PATH,
    MissingTelegramCredentialsError,
    TelegramCredentials,
    _ensure_publish_kit,
    build_video_attributes,
    create_telethon_client,
    is_cryptg_available,
    load_telethon_credentials,
    make_upload_progress_callback,
    mask_secret,
    prepare_small_thumbnail,
    should_send_raw_on_review,
)
from scripts.trend_cards_pool import build_category_trend_pools, resolve_trend_card_image
from scripts.trend_history_manager import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_PIPELINE_LOG_PATH,
    mark_trend_published,
)
from scripts.verify_video_gate import verify_video
from src.config import AppConfig, load_config
from src.utils.revision_engine import (
    compute_file_sha256,
    create_ai_escalation_ticket,
    resolve_ai_escalation_ticket,
    try_local_ai_revision_diagnosis,
)

DEFAULT_STATE_PATH = WORKSPACE_ROOT / "data" / "telegram_cards_state.json"
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.yaml"
DEFAULT_PID_PATH = WORKSPACE_ROOT / "data" / "telegram_bot.pid"
DEFAULT_LOG_PATH = WORKSPACE_ROOT / "logs" / "telegram_bot.log"
DEFAULT_MUTEX_NAME = "Local\\ViralStudioPro_TelegramBot_Mutex"


def configure_logging(
    log_path: Path = DEFAULT_LOG_PATH,
    logger_name: str = "telegram_bot",
) -> logging.Logger:
    """Configure rotating file logs (~5 MB x 3 backups) safe for windowless pythonw.exe."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target_logger = logging.getLogger(logger_name)
    target_logger.setLevel(logging.INFO)
    target_logger.propagate = False

    for h in list(target_logger.handlers):
        target_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    rot_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    rot_handler.setFormatter(fmt)
    target_logger.addHandler(rot_handler)

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        target_logger.addHandler(stream_handler)

    logging.getLogger("telethon").setLevel(logging.WARNING)
    return target_logger


logger = logging.getLogger("telegram_bot")


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if k32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return True
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class SingleInstanceLock:
    """Cross-OS single-instance guard (Windows Named Mutex + Linux fcntl.flock + PID file)."""

    def __init__(
        self,
        pid_path: Path = DEFAULT_PID_PATH,
        mutex_name: str = DEFAULT_MUTEX_NAME,
    ) -> None:
        self.pid_path = pid_path
        self.mutex_name = mutex_name
        self._mutex_handle: Any = None
        self._flock_fd: Any = None
        self._flock_path = pid_path.with_suffix(".lock")
        self._acquired = False

    def acquire(self) -> bool:
        if self._acquired:
            return True
        if os.name == "nt":
            ERROR_ALREADY_EXISTS = 183
            k32 = ctypes.windll.kernel32
            k32.SetLastError(0)
            handle = k32.CreateMutexW(None, False, self.mutex_name)
            last_err = k32.GetLastError()
            if not handle:
                return False
            if last_err == ERROR_ALREADY_EXISTS:
                k32.CloseHandle(handle)
                return False
            self._mutex_handle = handle
        else:
            try:
                import fcntl

                self._flock_path.parent.mkdir(parents=True, exist_ok=True)
                fd = self._flock_path.open("a+", encoding="utf-8")
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._flock_fd = fd
                except OSError:
                    fd.close()
                    return False
            except ImportError:
                pass

        if self.pid_path.exists():
            try:
                old_pid = int(self.pid_path.read_text(encoding="utf-8").strip())
                if old_pid != os.getpid() and _is_pid_alive(old_pid):
                    self._release_mutex()
                    return False
            except Exception:
                pass

        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        self._acquired = True
        return True

    def _release_mutex(self) -> None:
        if os.name == "nt" and self._mutex_handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self._mutex_handle)
            except Exception:
                pass
            self._mutex_handle = None
        if self._flock_fd is not None:
            try:
                import fcntl

                fcntl.flock(self._flock_fd.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                self._flock_fd.close()
            except Exception:
                pass
            self._flock_fd = None

    def release(self) -> None:
        if self._acquired:
            try:
                if self.pid_path.exists() and self.pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    self.pid_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._acquired = False
        self._release_mutex()


def next_backoff_sec(current_sec: float, uptime_sec: float, base_sec: float = 5.0, max_sec: float = 120.0) -> float:
    """Compute exponential reconnect delay after sleep/wake or network drops."""
    if uptime_sec >= 60.0:
        return base_sec
    return min(max_sec, max(base_sec, current_sec * 2.0))


def _resolve_console_python() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cand = exe.with_name("python.exe")
        if cand.exists():
            return str(cand)
    return str(exe)


def load_bot_credentials(env_path: Path | None = None) -> TelegramCredentials:
    """Load Telethon MTProto credentials strictly from .env relative to PROJECT_ROOT."""
    return load_telethon_credentials(env_file=env_path or (WORKSPACE_ROOT / ".env"))


def is_authorized(chat_id: int | None, user_id: int | None, allowed_chat_id: int) -> bool:
    """Hard whitelist check: only exact TELEGRAM_CHAT_ID is permitted."""
    if chat_id is None or user_id is None:
        return False
    return int(chat_id) == int(allowed_chat_id) and int(user_id) == int(allowed_chat_id)


def _msg_id(msg: Any) -> int:
    return int(getattr(msg, "id", getattr(msg, "message_id", 0)) or 0)


def empty_inline_keyboard() -> ReplyInlineMarkup:
    """Empty MTProto ReplyInlineMarkup to clear inline buttons cleanly on edit_message."""
    return ReplyInlineMarkup(rows=[])


def format_trend_card_caption(item: dict[str, Any]) -> str:
    """Exact 3-line Arabic card format required under each category photo."""
    cat_ar = str(item.get("category_ar") or item.get("category_key") or "ترند").strip()
    title = str(item.get("title") or "").strip()
    summary_ar = str(item.get("short_summary_ar") or "").strip()
    caption = f"{cat_ar}\n{title}\n{summary_ar}"
    return caption[:1020]


def format_dashboard_text(used: int, quota: int, note: str | None = None) -> str:
    lines = [
        "🎬 Viral Studio Pro — لوحة التحكم",
        "━━━━━━━━━━━━━━━━━━━━",
        f"📊 حصة اليوم: {used} / {quota} ترند",
    ]
    if note:
        lines.append(f"✨ {note}")
    lines.append("")
    lines.append("اختر الإجراء المطلوب من الأزرار بالأسفل:")
    return "\n".join(lines)


def main_menu_keyboard() -> list[list[Any]]:
    return [
        [Button.inline("🔥 طلب قائمة الترندات الجديدة", b"menu:trends")],
        [Button.inline("⚙️ إظهار العمليات الحالية", b"menu:status")],
        [Button.inline("✅ الترندات المنتجة والمنشورة", b"menu:published")],
        [Button.inline("📦 الترندات غير المنشورة (فيديو + نص)", b"menu:unpublished")],
    ]


def back_to_menu_keyboard() -> list[list[Any]]:
    return [[Button.inline("🔙 العودة إلى لوحة التحكم", b"menu:home")]]


def card_inline_keyboard(category_key: str) -> list[list[Any]]:
    return [
        [Button.inline("جهّز هذا الترند", f"prod:{category_key}".encode("utf-8"))],
        [Button.inline("الترند التالي من نفس الفئة", f"next:{category_key}".encode("utf-8"))],
    ]


def review_inline_keyboard(job_id: str) -> list[list[Any]]:
    return [
        [
            Button.inline("✅ موافقة", f"appr:{job_id}".encode("utf-8")),
            Button.inline("✏️ تعديل", f"edit:{job_id}".encode("utf-8")),
        ]
    ]


def publish_confirm_keyboard(job_id: str) -> list[list[Any]]:
    return [[Button.inline("✅ تم النشر", f"pub:{job_id}".encode("utf-8"))]]


def publish_run_confirm_keyboard(run_id: str) -> list[list[Any]]:
    return [[Button.inline("✅ تم النشر", f"pubrun:{run_id}".encode("utf-8"))]]


def extract_pipeline_failure_reason(run_dir: Path, gate: dict[str, Any], out_log: str) -> str:
    """Read real QA/Reviewer error diagnostics instead of CLI footer banner."""
    issues: list[str] = []
    review_json = run_dir / "review_package.json"
    if review_json.exists():
        try:
            data = json.loads(review_json.read_text(encoding="utf-8"))
            qa_issues = data.get("qa", {}).get("issues") or []
            issues.extend(str(i).strip() for i in qa_issues if str(i).strip())
            rv = data.get("reviewer_verdict") or {}
            diag = str(rv.get("error_diagnosis") or "").strip()
            if diag and diag not in issues:
                issues.append(diag)
        except Exception:
            pass
    for g_iss in gate.get("issues", []):
        s = str(g_iss).strip()
        if s and s not in issues:
            issues.append(s)
    if issues:
        return " | ".join(issues)[:600]

    filtered = [
        line.strip()
        for line in (out_log or "").splitlines()
        if line.strip()
        and not line.strip().startswith("===")
        and "HUMAN-IN-THE-LOOP" not in line
        and "--approve" not in line
    ]
    return "\n".join(filtered[-6:])[:500] or "خطأ غير معروف أثناء الإنتاج."


def collect_published_trends_list(
    state: dict[str, Any],
    history_path: Path = DEFAULT_HISTORY_PATH,
    pipeline_log_path: Path = DEFAULT_PIPELINE_LOG_PATH,
) -> list[dict[str, str]]:
    """Collect deduplicated list of produced & published trends for in-chat display (NOT a file)."""
    results: list[dict[str, str]] = []
    seen_keys: set[str] = set()

    def _add(title: str, run_id: str = "", category: str = "", timestamp: str = "") -> None:
        clean = title.strip()
        if not clean:
            return
        norm = f"{run_id.strip().lower()}|{clean.lower()}"
        topic_key = clean.lower()
        if topic_key in seen_keys or (run_id and norm in seen_keys):
            return
        seen_keys.add(topic_key)
        if run_id:
            seen_keys.add(norm)
        results.append(
            {
                "title": clean,
                "run_id": run_id,
                "category": category,
                "date": (timestamp or "")[:10],
            }
        )

    for job in state.get("jobs", {}).values():
        if job.get("status") == "published":
            _add(
                str(job.get("trend_title") or ""),
                str(job.get("run_id") or ""),
                str(job.get("category_ar") or job.get("category") or ""),
                str(job.get("published_at") or ""),
            )

    if history_path.exists():
        try:
            hist = json.loads(history_path.read_text(encoding="utf-8"))
            for entry in hist.get("entries", []):
                if str(entry.get("status") or "").lower() == "published":
                    _add(
                        str(entry.get("topic") or ""),
                        str(entry.get("run_id") or ""),
                        str(entry.get("category") or ""),
                        str(entry.get("timestamp") or ""),
                    )
        except Exception:
            pass

    if pipeline_log_path.exists():
        try:
            for r in json.loads(pipeline_log_path.read_text(encoding="utf-8")):
                if str(r.get("status") or "") == "APPROVED_AND_PACKAGED":
                    _add(
                        str(r.get("topic") or ""),
                        str(r.get("run_id") or ""),
                        "",
                        str(r.get("approved_at") or r.get("created_at") or ""),
                    )
        except Exception:
            pass

    return results


def _build_post_copy_message(run_dir: Path, topic: str, data: dict[str, Any] | None = None) -> str:
    pkg = data
    if pkg is None:
        review_path = run_dir / "review_package.json"
        pkg = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    meta = pkg.get("metadata", {})
    caption = str(meta.get("tiktok_caption") or pkg.get("topic") or topic).strip()
    tags = [f"#{str(h).lstrip('#')}" for h in meta.get("tiktok_hashtags", [])[:5]]
    tags_str = " ".join(tags) if tags else "#learnontiktok #trending #viral"
    disclose = bool(meta.get("tiktok_disclose_ai_content", True))
    ai_note = (
        "مفعل (ON) — فعّل علامة AI-generated content عند النشر على TikTok."
        if disclose
        else "غير مفعل (OFF)."
    )
    copy_block = f"{caption}\n\n{tags_str}".strip()
    return "\n".join(
        [
            f"✅ <b>جاهز للنشر اليدوي — {html.escape(topic)}</b>",
            f"<pre>{html.escape(copy_block)}</pre>",
            f"🤖 <b>تنبيه AI Content:</b> {html.escape(ai_note)}",
            "",
            "بعد الرفع على TikTok اضغط زر <b>✅ تم النشر</b> بالأسفل لأرشفته.",
        ]
    )


def collect_unpublished_runs(
    state: dict[str, Any],
    tiktok_dir: Path,
    history_path: Path = DEFAULT_HISTORY_PATH,
    pipeline_log_path: Path = DEFAULT_PIPELINE_LOG_PATH,
) -> list[dict[str, Any]]:
    """Find produced videos (tiktok/<run_id>/final_9x16.mp4) not yet marked published."""
    published_items = collect_published_trends_list(state, history_path, pipeline_log_path)
    pub_run_ids = {item["run_id"] for item in published_items if item["run_id"]}
    pub_topics = {item["title"].lower() for item in published_items if item["title"]}

    jobs_by_run: dict[str, dict[str, Any]] = {}
    for job in state.get("jobs", {}).values():
        rid = str(job.get("run_id") or "").strip()
        if rid:
            jobs_by_run[rid] = job

    unpublished: list[dict[str, Any]] = []
    if tiktok_dir.is_dir():
        for run_dir in sorted(tiktok_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name
            video_path = run_dir / "final_9x16.mp4"
            review_path = run_dir / "review_package.json"
            if not video_path.exists() or not review_path.exists():
                continue
            try:
                pkg = json.loads(review_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(pkg.get("status") or "") == "QA_FAILED":
                continue
            topic = str(pkg.get("topic") or run_id).strip()
            job = jobs_by_run.get(run_id)
            if job and job.get("status") == "published":
                continue
            if run_id in pub_run_ids or topic.lower() in pub_topics:
                continue
            unpublished.append(
                {
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "job_id": str(job["job_id"]) if job else None,
                    "topic": topic,
                    "video_path": video_path,
                    "duration_sec": float(pkg.get("duration_sec") or 65.0),
                    "copy_html": _build_post_copy_message(run_dir, topic, pkg),
                    "review_data": pkg,
                }
            )
    return unpublished


class BotStateStore:
    """Persistent state tracking category pools, messages, jobs, edits, and daily quota."""

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH) -> None:
        self.state_path = state_path
        self.state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("pools", {})
                    data.setdefault("cards", {})
                    data.setdefault("jobs", {})
                    data.setdefault("daily_counts", {})
                    data.setdefault("active_edit_job_id", None)
                    data.setdefault("tracked_message_ids", [])
                    data.setdefault("dashboard_message_id", None)
                    return data
            except Exception:
                pass
        return {
            "pools": {},
            "cards": {},
            "jobs": {},
            "daily_counts": {},
            "active_edit_job_id": None,
            "tracked_message_ids": [],
            "dashboard_message_id": None,
        }

    def reload(self) -> None:
        """Reload latest state from disk (syncs CLI flags with background bot)."""
        self.state = self._load()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def track_message(self, message_id: int | None) -> None:
        if not message_id:
            return
        mid = int(message_id)
        tracked: list[int] = self.state.setdefault("tracked_message_ids", [])
        if mid not in tracked:
            tracked.append(mid)
            self.save()

    def set_dashboard_message(self, message_id: int) -> None:
        self.state["dashboard_message_id"] = int(message_id)
        self.track_message(message_id)

    def collect_all_tracked_ids(self) -> list[int]:
        ids: set[int] = set(int(x) for x in self.state.get("tracked_message_ids", []) if x)
        if self.state.get("dashboard_message_id"):
            ids.add(int(self.state["dashboard_message_id"]))
        for c in self.state.get("cards", {}).values():
            if c.get("card_message_id"):
                ids.add(int(c["card_message_id"]))
        for j in self.state.get("jobs", {}).values():
            for field in ("card_message_id", "review_message_id", "publish_message_id"):
                if j.get(field):
                    ids.add(int(j[field]))
        return sorted(ids)

    def clear_tracked_messages(
        self,
        *,
        keep_ids: set[int] | None = None,
        preserve_edit_job: bool = False,
    ) -> None:
        if keep_ids is None:
            self.state["tracked_message_ids"] = []
            self.state["dashboard_message_id"] = None
            self.state["cards"] = {}
            if not preserve_edit_job:
                self.state["active_edit_job_id"] = None
        else:
            keep_set = {int(x) for x in keep_ids if x}
            self.state["tracked_message_ids"] = sorted(keep_set)
            dash_id = self.state.get("dashboard_message_id")
            if dash_id and int(dash_id) not in keep_set:
                self.state["dashboard_message_id"] = None
            self.state["cards"] = {
                cid: c
                for cid, c in (self.state.get("cards") or {}).items()
                if int(c.get("card_message_id") or 0) in keep_set
            }
            for j in (self.state.get("jobs") or {}).values():
                for field in ("card_message_id", "review_message_id", "publish_message_id"):
                    val = j.get(field)
                    if val and int(val) not in keep_set:
                        j[field] = None
            if not preserve_edit_job:
                self.state["active_edit_job_id"] = None
        self.save()

    @staticmethod
    def today_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_today_quota_used(self) -> int:
        return int(self.state["daily_counts"].get(self.today_key(), 0))

    def increment_today_quota(self) -> int:
        k = self.today_key()
        self.state["daily_counts"][k] = int(self.state["daily_counts"].get(k, 0)) + 1
        self.save()
        return self.state["daily_counts"][k]

    def set_pools(self, pools: dict[str, dict[str, Any]]) -> None:
        self.state["pools"] = pools
        self.save()

    def register_card(self, card_message_id: int, category_key: str, pool_index: int = 0) -> dict[str, Any]:
        pool = self.state["pools"].get(category_key, {})
        items = pool.get("items", [])
        item = items[pool_index] if items and 0 <= pool_index < len(items) else {}
        rec = {
            "card_message_id": int(card_message_id),
            "category": category_key,
            "category_ar": pool.get("category_ar", category_key),
            "pool_index": pool_index,
            "trend_id": item.get("id", ""),
            "status": "active",
        }
        self.state["cards"][str(card_message_id)] = rec
        self.track_message(card_message_id)
        self.save()
        return rec

    def get_card(self, card_message_id: int) -> dict[str, Any] | None:
        return self.state["cards"].get(str(card_message_id))

    def advance_card_trend(self, card_message_id: int, category_key: str) -> tuple[dict[str, Any] | None, bool]:
        pool = self.state["pools"].get(category_key, {})
        items = pool.get("items", [])
        if not items:
            return None, False

        card = self.state["cards"].get(str(card_message_id))
        if not card:
            card = self.register_card(card_message_id, category_key, pool_index=0)

        cur_idx = int(card.get("pool_index", 0))
        next_idx = (cur_idx + 1) % len(items)
        wrapped = next_idx == 0 and len(items) > 1
        item = items[next_idx]
        card["pool_index"] = next_idx
        card["trend_id"] = item.get("id", "")
        self.state["cards"][str(card_message_id)] = card
        self.save()
        return item, wrapped

    def start_job_for_card(
        self,
        card_message_id: int,
        category_key: str,
        daily_quota: int = 2,
    ) -> tuple[dict[str, Any] | None, str]:
        if self.get_today_quota_used() >= daily_quota:
            return None, "quota_exceeded"

        card = self.state["cards"].get(str(card_message_id))
        if not card:
            return None, "card_not_found"
        if card.get("status") in ("producing", "published"):
            return None, f"card_{card['status']}"

        items = self.state["pools"].get(category_key, {}).get("items", [])
        idx = int(card.get("pool_index", 0))
        if not items or idx >= len(items):
            return None, "empty_pool"
        item = items[idx]

        now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = f"job_{card_message_id}_{idx}"
        run_id = f"run_{now_tag}"
        job = {
            "job_id": job_id,
            "run_id": run_id,
            "card_message_id": int(card_message_id),
            "category": category_key,
            "category_ar": card.get("category_ar", category_key),
            "trend_id": item.get("id", ""),
            "trend_title": item.get("title", ""),
            "sourcing_keywords": item.get("sourcing_keywords", []),
            "status": "producing",
            "review_message_id": None,
            "review_sent_raw": False,
            "publish_message_id": None,
            "revision_round": 0,
            "edit_history": [],
        }
        card["status"] = "producing"
        self.state["cards"][str(card_message_id)] = card
        self.state["jobs"][job_id] = job
        self.save()
        return job, "ok"

    def mark_job_review_ready(
        self,
        job_id: str,
        review_message_id: int | None = None,
        review_sent_raw: bool | None = None,
    ) -> dict[str, Any]:
        job = self.state["jobs"][job_id]
        job["status"] = "awaiting_review"
        if review_message_id is not None:
            job["review_message_id"] = int(review_message_id)
            self.track_message(review_message_id)
        if review_sent_raw is not None:
            job["review_sent_raw"] = bool(review_sent_raw)
        if self.state.get("active_edit_job_id") == job_id:
            self.state["active_edit_job_id"] = None
        self.save()
        return job

    def mark_job_failed(self, job_id: str, reason: str) -> None:
        job = self.state["jobs"].get(job_id)
        if not job:
            return
        job["status"] = "failed"
        job["error"] = reason
        card = self.state["cards"].get(str(job["card_message_id"]))
        if card:
            card["status"] = "active"
        if self.state.get("active_edit_job_id") == job_id:
            self.state["active_edit_job_id"] = None
        self.save()

    def mark_job_escalated(self, job_id: str, ticket_id: str, reason: str) -> dict[str, Any] | None:
        job = self.state["jobs"].get(job_id)
        if not job:
            return None
        job["status"] = "escalated_to_ai"
        job["escalation_ticket_id"] = ticket_id
        job["escalation_reason"] = reason
        if self.state.get("active_edit_job_id") == job_id:
            self.state["active_edit_job_id"] = None
        self.save()
        return job

    def enter_edit_feedback(self, job_id: str) -> dict[str, Any] | None:
        job = self.state["jobs"].get(job_id)
        if not job:
            return None
        job["status"] = "awaiting_edit_feedback"
        self.state["active_edit_job_id"] = job_id
        self.save()
        return job

    def submit_edit_feedback(self, feedback_text: str) -> dict[str, Any] | None:
        job_id = self.state.get("active_edit_job_id")
        if not job_id or job_id not in self.state["jobs"]:
            return None
        job = self.state["jobs"][job_id]
        job["status"] = "revising"
        job["revision_round"] = int(job.get("revision_round", 0)) + 1
        job.setdefault("edit_history", []).append(feedback_text)
        self.state["active_edit_job_id"] = None
        self.save()
        return job

    def approve_job(self, job_id: str, publish_message_id: int | None = None) -> dict[str, Any] | None:
        job = self.state["jobs"].get(job_id)
        if not job:
            return None
        job["status"] = "approved_awaiting_publish"
        if publish_message_id is not None:
            job["publish_message_id"] = int(publish_message_id)
            self.track_message(publish_message_id)
        self.save()
        return job

    def mark_job_published(
        self,
        job_id: str,
        history_path: Path = DEFAULT_HISTORY_PATH,
        queue_path: Path = WORKSPACE_ROOT / "data" / "trends_queue.json",
    ) -> tuple[dict[str, Any] | None, bool]:
        job = self.state["jobs"].get(job_id)
        if not job:
            return None, False
        if job.get("status") == "published":
            return job, True

        mark_trend_published(
            trend_id=str(job.get("trend_id", "")),
            topic=str(job.get("trend_title", "")),
            category=str(job.get("category", "")),
            run_id=str(job.get("run_id", "")),
            history_path=history_path,
            queue_path=queue_path,
        )
        job["status"] = "published"
        job["published_at"] = datetime.now(timezone.utc).isoformat()
        card = self.state["cards"].get(str(job.get("card_message_id")))
        if card:
            card["status"] = "published"
        self.increment_today_quota()
        if self.state_path == DEFAULT_STATE_PATH:
            try:
                from scripts.check_daily_quota import record_daily_video

                record_daily_video(str(job.get("run_id", "")), topic=str(job.get("trend_title", "")))
            except Exception:
                pass
        self.save()
        return job, False

    def mark_run_published_direct(
        self,
        run_id: str,
        topic: str,
        category: str = "",
        history_path: Path = DEFAULT_HISTORY_PATH,
        queue_path: Path = WORKSPACE_ROOT / "data" / "trends_queue.json",
    ) -> bool:
        for job in self.state.get("jobs", {}).values():
            if job.get("run_id") == run_id:
                _, already = self.mark_job_published(
                    job["job_id"], history_path=history_path, queue_path=queue_path
                )
                return already

        mark_trend_published(
            trend_id="",
            topic=topic,
            category=category,
            run_id=run_id,
            history_path=history_path,
            queue_path=queue_path,
        )
        synth_id = f"job_run_{run_id}"
        if synth_id in self.state["jobs"] and self.state["jobs"][synth_id].get("status") == "published":
            return True
        self.state["jobs"][synth_id] = {
            "job_id": synth_id,
            "run_id": run_id,
            "card_message_id": 0,
            "category": category,
            "category_ar": category,
            "trend_id": "",
            "trend_title": topic,
            "status": "published",
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
        self.increment_today_quota()
        if self.state_path == DEFAULT_STATE_PATH:
            try:
                from scripts.check_daily_quota import record_daily_video

                record_daily_video(run_id, topic=topic)
            except Exception:
                pass
        self.save()
        return False


async def _run_subprocess_async(args: list[str]) -> tuple[int, str]:
    clean_args = list(args)
    if clean_args and clean_args[0] == sys.executable:
        clean_args[0] = _resolve_console_python()
    kwargs: dict[str, Any] = {
        "cwd": str(WORKSPACE_ROOT),
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    proc = await asyncio.create_subprocess_exec(*clean_args, **kwargs)
    out_bytes, _ = await proc.communicate()
    output = out_bytes.decode("utf-8", errors="replace") if out_bytes else ""
    return int(proc.returncode or 0), output


def _delete_was_ignored_by_telegram(result: Any) -> bool:
    """Detect when MTProto delete_messages returns AffectedMessages(pts_count=0) for >48h messages."""
    if result is None:
        return False
    items = result if isinstance(result, (list, tuple)) else [result]
    pts_vals = [getattr(item, "pts_count", None) for item in items]
    valid = [v for v in pts_vals if v is not None]
    return bool(valid) and all(int(v) == 0 for v in valid)


async def _delete_or_archive_card(client: Any, chat_id: int, card_message_id: int) -> str:
    """Delete ONLY the original card message; handle >48h Telegram limit gracefully."""
    if not card_message_id:
        return "unchanged"
    try:
        res = await client.delete_messages(chat_id, [int(card_message_id)], revoke=True)
        if not _delete_was_ignored_by_telegram(res):
            return "deleted"
        logger.info("Card %s delete returned pts_count=0 (likely >48h); archiving caption.", card_message_id)
    except (tg_errors.RPCError, Exception) as exc:
        logger.info("Card %s could not be deleted (likely >48h): %s; archiving caption.", card_message_id, exc)

    try:
        await client.edit_message(
            chat_id,
            int(card_message_id),
            text="✅ تم النشر والأرشفة",
            buttons=empty_inline_keyboard(),
        )
        return "archived"
    except (tg_errors.RPCError, Exception) as edit_exc:
        logger.warning("Failed to archive card caption %s: %s", card_message_id, edit_exc)
        return "unchanged"


class InteractiveTikTokBot:
    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG_PATH,
        state_path: Path = DEFAULT_STATE_PATH,
        *,
        creds: TelegramCredentials | None = None,
        allow_missing_credentials: bool = False,
    ) -> None:
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        self.store = BotStateStore(state_path)
        session_rel = str(self.config.telegram.get("session_file", "data/viral_studio_bot.session"))
        sp = Path(session_rel)
        self.session_path: Path = sp if sp.is_absolute() else (WORKSPACE_ROOT / sp)

        if creds is not None:
            self.creds: TelegramCredentials | None = creds
        else:
            try:
                self.creds = load_bot_credentials()
            except MissingTelegramCredentialsError:
                if not allow_missing_credentials:
                    raise
                self.creds = None

        self.token: str = self.creds.bot_token if self.creds else ""
        self.api_hash: str = self.creds.api_hash if self.creds else ""
        self.allowed_chat_id: int = int(self.creds.chat_id) if self.creds else 0

    def _authorized(self, event: Any) -> bool:
        chat_id = getattr(event, "chat_id", None)
        sender_id = getattr(event, "sender_id", None)
        return is_authorized(chat_id, sender_id, self.allowed_chat_id)

    def _categories_label_list(self) -> str:
        cats = self.config.telegram.get("categories") or []
        labels = [str(c.get("label_ar") or c.get("key") or "").strip() for c in cats if isinstance(c, dict)]
        labels = [x for x in labels if x]
        return "، ".join(labels) if labels else "أفلام، مسلسلات، ألعاب، ذكاء اصطناعي"

    async def _send_tracked_text(
        self,
        client: Any,
        text: str,
        buttons: Any = None,
        parse_mode: str | None = None,
    ) -> Any:
        msg = await client.send_message(
            self.allowed_chat_id,
            text,
            buttons=buttons,
            parse_mode=parse_mode,
            link_preview=False,
        )
        self.store.track_message(_msg_id(msg))
        return msg

    async def _delete_message_safe(self, client: Any, mid: int) -> None:
        try:
            await client.delete_messages(self.allowed_chat_id, [int(mid)], revoke=True)
        except Exception:
            pass

    async def _send_video_file(
        self,
        client: Any,
        video_path: Path,
        caption: str,
        *,
        force_document: bool,
        width: int,
        height: int,
        duration_sec: float,
        buttons: Any = None,
        thumb_path: Path | None = None,
    ) -> Any:
        size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0.0
        threshold_mb = float(self.config.telegram.get("progress_log_threshold_mb", 50.0))
        mode_desc = "force_document=True (uncompressed master)" if force_document else "supports_streaming=True (preview)"
        logger.info("Uploading %s (%.1f MB) via Telethon MTProto [%s]", video_path.name, size_mb, mode_desc)

        progress_cb = make_upload_progress_callback(
            video_path,
            target_logger=logger,
            threshold_mb=threshold_mb,
        )
        small_thumb = prepare_small_thumbnail(thumb_path)
        msg = await client.send_file(
            self.allowed_chat_id,
            str(video_path),
            caption=caption,
            force_document=force_document,
            supports_streaming=(not force_document),
            attributes=None if force_document else build_video_attributes(width, height, duration_sec, True),
            thumb=str(small_thumb) if small_thumb else None,
            buttons=buttons,
            progress_callback=progress_cb,
        )
        self.store.track_message(_msg_id(msg))
        return msg

    async def send_dashboard_menu(self, client: Any, note: str | None = None) -> int:
        self.config = load_config(self.config_path)
        self.store.reload()
        quota = int(self.config.telegram.get("daily_video_quota", 2))
        used = self.store.get_today_quota_used()
        msg = await client.send_message(
            self.allowed_chat_id,
            format_dashboard_text(used, quota, note=note),
            buttons=main_menu_keyboard(),
            link_preview=False,
        )
        mid = _msg_id(msg)
        self.store.set_dashboard_message(mid)
        return mid

    async def ensure_dashboard_first(self, client: Any, note: str | None = None) -> int:
        """Keep the current Dashboard pinned as the first tracked message without duplicating it."""
        self.config = load_config(self.config_path)
        self.store.reload()
        quota = int(self.config.telegram.get("daily_video_quota", 2))
        used = self.store.get_today_quota_used()
        dash_id = self.store.state.get("dashboard_message_id")
        text = format_dashboard_text(used, quota, note=note)
        if dash_id:
            try:
                await client.edit_message(
                    self.allowed_chat_id,
                    int(dash_id),
                    text=text,
                    buttons=main_menu_keyboard(),
                    link_preview=False,
                )
                return int(dash_id)
            except tg_errors.MessageNotModifiedError:
                return int(dash_id)
            except Exception as exc:
                if "not modified" in str(exc).lower():
                    return int(dash_id)
        return await self.send_dashboard_menu(client, note=note)

    async def clear_non_dashboard_messages(
        self,
        client: Any,
        *,
        keep_ids: set[int] | None = None,
        preserve_edit_job: bool = False,
        note: str | None = None,
    ) -> int:
        """Delete all tracked messages except the first Dashboard (and optional keep_ids)."""
        dash_id = await self.ensure_dashboard_first(client, note=note)
        protected = {dash_id} | {int(x) for x in (keep_ids or set()) if x}
        msg_ids = self.store.collect_all_tracked_ids()
        for mid in reversed(msg_ids):
            if mid in protected:
                continue
            await self._delete_message_safe(client, mid)
        self.store.clear_tracked_messages(
            keep_ids=protected,
            preserve_edit_job=preserve_edit_job,
        )
        return dash_id

    async def clear_chat_and_show_menu(self, client: Any, note: str | None = None) -> int:
        """Delete all tracked bot/chat messages cleanly, then send a fresh 4-button dashboard."""
        msg_ids = self.store.collect_all_tracked_ids()
        for mid in reversed(msg_ids):
            await self._delete_message_safe(client, mid)
        self.store.clear_tracked_messages()
        return await self.send_dashboard_menu(client, note=note)

    async def send_category_cards(
        self, client: Any, loading_msg_id: int | None = None
    ) -> list[int]:
        self.config = load_config(self.config_path)
        pools = await asyncio.to_thread(
            build_category_trend_pools,
            self.config,
            fetch_remote_thumbs=True,
            refresh_live=True,
        )
        if loading_msg_id:
            await self._delete_message_safe(client, int(loading_msg_id))
        self.store.reload()
        self.store.set_pools(pools)
        delay_sec = float(self.config.telegram.get("card_send_delay_sec", 1.0))
        sent_ids: list[int] = []
        items_list = list(pools.items())
        for idx, (cat_key, cat_data) in enumerate(items_list):
            items = cat_data.get("items", [])
            if not items:
                await self._send_tracked_text(
                    client,
                    f"⚠️ لا توجد ترندات غير مستخدمة حالياً في فئة: {cat_data.get('category_ar', cat_key)}",
                )
            else:
                first = items[0]
                img_path = resolve_trend_card_image(first, fetch_remote=False)
                caption = format_trend_card_caption(first)
                msg = await client.send_file(
                    self.allowed_chat_id,
                    str(img_path),
                    caption=caption,
                    force_document=False,
                    buttons=card_inline_keyboard(cat_key),
                )
                mid = _msg_id(msg)
                self.store.register_card(mid, cat_key, pool_index=0)
                sent_ids.append(mid)
            if idx < len(items_list) - 1 and delay_sec > 0:
                await asyncio.sleep(delay_sec)
        await self._send_tracked_text(
            client,
            "اختر بطاقة أعلاه لبدء التجهيز، أو ارجع للوحة التحكم:",
            buttons=back_to_menu_keyboard(),
        )
        return sent_ids

    async def send_status_summary(self, client: Any) -> None:
        self.store.reload()
        quota = int(self.config.telegram.get("daily_video_quota", 2))
        used = self.store.get_today_quota_used()
        jobs = list(self.store.state.get("jobs", {}).values())
        active = [j for j in jobs if j.get("status") != "published"][-6:]
        status_ar = {
            "producing": "قيد الإنتاج",
            "revising": "قيد التعديل",
            "awaiting_review": "بانتظار مراجعتك",
            "awaiting_edit_feedback": "بانتظار ملاحظاتك",
            "approved_awaiting_publish": "معتمد وبانتظار النشر",
            "escalated_to_ai": "🤖 مُصعَّد لمراجعة الذكاء الاصطناعي (Antigravity)",
            "failed": "فشل",
        }
        lines = [
            f"⚙️ العمليات الحالية وحصة اليوم ({self.store.today_key()}): {used} / {quota}",
            "━━━━━━━━━━━━━━━━━━━━",
        ]
        if not active:
            lines.append("• لا توجد عمليات جارية حالياً.")
        else:
            for j in active:
                st = status_ar.get(str(j.get("status")), str(j.get("status")))
                lines.append(
                    f"• {j['run_id']} | {j.get('category_ar', '')} | {st}\n  📌 {j['trend_title']}"
                )
        await self._send_tracked_text(client, "\n".join(lines), buttons=back_to_menu_keyboard())

    async def send_published_list(self, client: Any) -> None:
        self.store.reload()
        items = collect_published_trends_list(self.store.state)
        if not items:
            await self._send_tracked_text(
                client,
                "✅ الترندات المنتجة والمنشورة:\nلا توجد ترندات مسجلة كمنشورة حتى الآن.",
                buttons=back_to_menu_keyboard(),
            )
            return

        header = f"✅ قائمة الترندات المنتجة والمنشورة ({len(items)}):\n━━━━━━━━━━━━━━━━━━━━"
        chunks: list[str] = []
        cur = header
        for idx, it in enumerate(items, start=1):
            meta_parts = [p for p in (it.get("run_id"), it.get("date")) if p]
            meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
            entry = f"\n{idx}. {it['title']}{meta_str}"
            if len(cur) + len(entry) > 3700:
                chunks.append(cur)
                cur = entry.lstrip("\n")
            else:
                cur += entry
        chunks.append(cur)

        for i, chunk in enumerate(chunks):
            kb = back_to_menu_keyboard() if i == len(chunks) - 1 else None
            await self._send_tracked_text(client, chunk, buttons=kb)

    async def send_unpublished_items(self, client: Any) -> None:
        self.config = load_config(self.config_path)
        self.store.reload()
        pending = collect_unpublished_runs(self.store.state, self.config.tiktok_dir)
        if not pending:
            await self._send_tracked_text(
                client,
                "📦 لا توجد ترندات منتجة بانتظار النشر حالياً.",
                buttons=back_to_menu_keyboard(),
            )
            return

        send_as_doc = bool(self.config.telegram.get("send_as_document", True))
        for item in pending:
            vpath: Path = item["video_path"]
            run_dir: Path = item["run_dir"]
            size_mb = vpath.stat().st_size / (1024 * 1024) if vpath.exists() else 0.0
            vid_cap = (
                f"📦 فيديو غير منشور بعد ({size_mb:.1f} MB — بدون ضغط)\n"
                f"📌 {item['topic']}\n"
                f"🆔 {item['run_id']} | ⏱️ {item['duration_sec']:.1f}s"
            )[:1000]
            await self._send_video_file(
                client,
                vpath,
                vid_cap,
                force_document=send_as_doc,
                width=int(self.config.video.get("width", 2160)),
                height=int(self.config.video.get("height", 3840)),
                duration_sec=float(item["duration_sec"]),
                thumb_path=run_dir / "cover.jpg",
            )

            kit_path = _ensure_publish_kit(run_dir, item.get("review_data") or {})
            if kit_path and kit_path.exists():
                kmsg = await client.send_file(
                    self.allowed_chat_id,
                    str(kit_path),
                    caption=f"📎 حزمة النشر: {kit_path.name}",
                    force_document=True,
                )
                self.store.track_message(_msg_id(kmsg))

            kb = (
                publish_confirm_keyboard(item["job_id"])
                if item.get("job_id")
                else publish_run_confirm_keyboard(item["run_id"])
            )
            await self._send_tracked_text(
                client,
                item["copy_html"],
                buttons=kb,
                parse_mode="html",
            )
        await self._send_tracked_text(
            client,
            f"✨ تم إرسال {len(pending)} ترند جاهز للنشر بجودته الأصلية مع نصوص النسخ.",
            buttons=back_to_menu_keyboard(),
        )

    async def cmd_start(self, event: Any) -> None:
        if not self._authorized(event):
            return
        self.store.track_message(_msg_id(getattr(event, "message", event)))
        await self.clear_chat_and_show_menu(event.client)

    async def cmd_trends(self, event: Any) -> None:
        if not self._authorized(event):
            return
        self.store.track_message(_msg_id(getattr(event, "message", event)))
        await self.clear_non_dashboard_messages(event.client)
        loading = await self._send_tracked_text(
            event.client,
            f"🔎 جارٍ فحص مصادر الترندات الحية وتحضير البطاقات ({self._categories_label_list()})...",
        )
        await self.send_category_cards(event.client, loading_msg_id=_msg_id(loading))

    async def cmd_status(self, event: Any) -> None:
        if not self._authorized(event):
            return
        self.store.track_message(_msg_id(getattr(event, "message", event)))
        await self.clear_non_dashboard_messages(event.client)
        await self.send_status_summary(event.client)

    async def _run_pipeline_and_deliver_review(
        self,
        client: Any,
        job: dict[str, Any],
        revision_feedback: str | None = None,
        ai_escalated_note: str | None = None,
        abort_if_unchanged: bool = False,
    ) -> bool:
        job_id = job["job_id"]
        run_id = job["run_id"]
        run_dir = self.config.tiktok_dir / run_id
        video_path = run_dir / "final_9x16.mp4"
        hash_before = compute_file_sha256(video_path)

        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "run_pipeline.py"),
            "--run-id",
            run_id,
            "--no-telegram-notify",
        ]
        if revision_feedback:
            cmd.extend(["--revision-feedback", revision_feedback])
        elif ai_escalated_note:
            script_p = run_dir / "script_plan.json"
            if script_p.exists():
                cmd.extend(["--script-json", str(script_p)])
        else:
            cmd.extend(["--topic", job["trend_title"]])
            if job.get("category"):
                cmd.extend(["--niche", str(job["category"])])
            kws = [str(k).strip() for k in (job.get("sourcing_keywords") or []) if str(k).strip()]
            if kws:
                cmd.extend(["--keywords", ",".join(kws[:4])])

        rc, out_log = await _run_subprocess_async(cmd)
        min_sec = float(self.config.video.get("hard_monetization_floor_sec", 61))
        gate = verify_video(video_path, min_sec=min_sec)
        hash_after = compute_file_sha256(video_path)

        if rc != 0 or not gate.get("passed"):
            reason = extract_pipeline_failure_reason(run_dir, gate, out_log)
            if not abort_if_unchanged:
                self.store.mark_job_failed(job_id, reason)
                await self.clear_non_dashboard_messages(client)
                await self._send_tracked_text(
                    client,
                    f"❌ تعذر إتمام إنتاج {run_id}:\n{reason}",
                    buttons=back_to_menu_keyboard(),
                )
            return False

        is_unchanged = bool((revision_feedback or ai_escalated_note) and hash_before and hash_before == hash_after)
        if abort_if_unchanged and is_unchanged:
            return False

        self.config = load_config(self.config_path)
        send_raw = should_send_raw_on_review(self.config.telegram)
        size_mb = video_path.stat().st_size / (1024 * 1024)

        lang = str(self.config.project.get("language", "en")).upper()
        rev_applied_info: dict[str, Any] = {}
        rev_applied_path = run_dir / "revision_applied.json"
        if revision_feedback and rev_applied_path.exists():
            try:
                rev_applied_info = json.loads(rev_applied_path.read_text(encoding="utf-8"))
            except Exception:
                rev_applied_info = {}

        applied_actions = [str(a).strip() for a in (rev_applied_info.get("applied_actions") or []) if str(a).strip()]
        rev_warnings = [str(w).strip() for w in (rev_applied_info.get("warnings") or []) if str(w).strip()]

        mode_label = (
            "معالجة ذكاء اصطناعي"
            if ai_escalated_note
            else ("نسخة معدلة" if revision_feedback else "جاهز للمراجعة")
        )
        gw = int(gate.get("width", 1080))
        gh = int(gate.get("height", 1920))
        res_label = f"{gw}x{gh} (4K)" if gw >= 2160 else f"{gw}x{gh}"
        cap_lines = [
            f"🎬 معاينة الفيديو ({mode_label}) — {res_label} ({size_mb:.1f} MB)",
            f"📌 {job['trend_title']}",
            f"🏷️ الفئة: {job.get('category_ar', job['category'])} | 🌐 {lang} | ⏱️ {gate['duration_sec']:.2f}s",
            f"📁 {run_id}",
        ]
        if send_raw:
            cap_lines.append("📦 النسخة الأصلية الكاملة غير المضغوطة (Document — بدون ضغط)")
        else:
            cap_lines.append("▶️ معاينة قابلة للتشغيل المباشر — عند الضغط على (موافقة) سيُرفع الملف الأصلي غير المضغوط بالكامل.")
        if applied_actions:
            cap_lines.append("🛠️ التعديلات المطبّقة: " + "؛ ".join(applied_actions))
        elif ai_escalated_note:
            cap_lines.append(f"🤖 تشخيص AI: {ai_escalated_note[:180]}")
        caption = "\n".join(cap_lines)[:1020]

        await self.clear_non_dashboard_messages(client)
        msg = await self._send_video_file(
            client,
            video_path,
            caption,
            force_document=send_raw,
            width=gw,
            height=gh,
            duration_sec=float(gate.get("duration_sec", 65)),
            buttons=review_inline_keyboard(job_id),
            thumb_path=run_dir / "cover.jpg",
        )
        self.store.mark_job_review_ready(job_id, review_message_id=_msg_id(msg), review_sent_raw=send_raw)
        if is_unchanged:
            await self._send_tracked_text(
                client,
                "⚠️ تنبيه: لم يتغير ملف الفيديو بعد إعادة التصيير — إن تكررت المشكلة سيتم تصعيدها تلقائياً لنموذج الذكاء الاصطناعي بعد 3 محاولات.",
            )
        elif revision_feedback and not applied_actions:
            await self._send_tracked_text(
                client,
                "⚠️ تم إعادة التصيير، لكن ملاحظتك لم تطابق أمراً تلقائياً مدعوماً.\n"
                "الأوامر المدعومة: حذف ثوانٍ/مشاهد مثل 'احذف من 15 إلى 21' أو 'Delete scenes 15s to 21s'، تغيير السرعة (أسرع/أبطأ)، تثبيت الإنجليزية، أو استبدال لقطة/مشهد محدد.",
            )
        elif rev_warnings:
            await self._send_tracked_text(client, "ℹ️ " + " | ".join(rev_warnings))
        return True

    async def _escalate_edit_job_to_ai(self, client: Any, job: dict[str, Any]) -> None:
        """After >3 edit rounds, try local $0 AI diagnosis or escalate to Antigravity without repeating the same video."""
        run_id = job["run_id"]
        job_id = job["job_id"]
        run_dir = self.config.tiktok_dir / run_id
        diag = await asyncio.to_thread(try_local_ai_revision_diagnosis, job, run_dir, self.config)
        diag_ar = str((diag or {}).get("diagnosis_ar") or "")

        if diag and diag.get("fixed"):
            ok = await self._run_pipeline_and_deliver_review(
                client,
                job,
                revision_feedback=None,
                ai_escalated_note=diag_ar or "إصلاح ذكي محلي",
                abort_if_unchanged=True,
            )
            if ok:
                return

        queue_rel = str(self.config.telegram.get("ai_escalation_queue_file", "data/ai_escalation_queue.json"))
        queue_path = WORKSPACE_ROOT / queue_rel
        ticket = create_ai_escalation_ticket(job, run_dir, queue_path, diagnosis_ar=diag_ar)
        reason = diag_ar or f"تجاوز الحد التلقائي ({job.get('revision_round', 4)} تعديلات) دون حل نهائي"
        self.store.mark_job_escalated(job_id, ticket["ticket_id"], reason=reason)
        await self.clear_non_dashboard_messages(client)
        await self._send_tracked_text(
            client,
            "\n".join(
                [
                    f"🤖 تم إيقاف التكرار الآلي وتصعيد المشروع ({job['trend_title']}) إلى مراجعة الذكاء الاصطناعي:",
                    f"🎫 رقم التذكرة: {ticket['ticket_id']}",
                    f"📁 التشغيل: tiktok/{run_id}",
                    f"🔁 عدد محاولات التعديل: {job.get('revision_round', 4)}",
                    f"📝 ملف التذكرة: tiktok/{run_id}/AI_ESCALATION_TICKET.md",
                    "",
                    f"لن يعيد البوت إرسال نفس الفيديو. افتح محادثة Antigravity واكتب: راجع تذكرة التصعيد {run_id}",
                ]
            ),
            buttons=back_to_menu_keyboard(),
        )

    async def _finish_publish_and_maybe_cleanup(
        self,
        client: Any,
        event: Any,
        trend_title: str,
        card_message_id: int,
    ) -> None:
        quota = int(self.config.telegram.get("daily_video_quota", 2))
        used = self.store.get_today_quota_used()
        should_wipe = used >= quota and bool(self.config.telegram.get("cleanup_chat_on_quota_complete", True))
        if should_wipe:
            await event.answer(f"تم نشر {quota}/{quota}! جارٍ تنظيف المحادثة...")
            await _delete_or_archive_card(client, self.allowed_chat_id, card_message_id)
            await self.clear_non_dashboard_messages(
                client,
                note=f"تم إتمام ونشر {used} / {quota} ترند اليوم وتنظيف الرسائل السابقة.",
            )
            return

        await _delete_or_archive_card(client, self.allowed_chat_id, card_message_id)
        confirm_text = (
            f"✅ تم تسجيل النشر في الأرشيف ({trend_title})\n"
            f"📊 حصة اليوم: {used} / {quota}"
        ).strip()
        await self.clear_non_dashboard_messages(client)
        await self._send_tracked_text(client, confirm_text, buttons=back_to_menu_keyboard())
        await event.answer("تم النشر والأرشفة بنجاح ✅")

    async def on_callback(self, event: Any) -> None:
        if not self._authorized(event):
            return
        self.store.reload()
        client = event.client
        raw_data = getattr(event, "data", b"")
        data = (
            raw_data.decode("utf-8", errors="replace")
            if isinstance(raw_data, (bytes, bytearray))
            else str(raw_data or "")
        )
        msg_id = int(getattr(event, "message_id", 0) or 0)

        if data == "menu:home":
            await event.answer()
            await self.clear_non_dashboard_messages(client)
            return

        if data == "menu:trends":
            await event.answer("جارٍ جلب بطاقات الترندات...")
            await self.clear_non_dashboard_messages(client)
            loading = await self._send_tracked_text(
                client,
                f"🔎 جارٍ فحص مصادر الترندات الحية وتحضير البطاقات ({self._categories_label_list()})...",
            )
            await self.send_category_cards(client, loading_msg_id=_msg_id(loading))
            return

        if data == "menu:status":
            await event.answer()
            await self.clear_non_dashboard_messages(client)
            await self.send_status_summary(client)
            return

        if data == "menu:published":
            await event.answer()
            await self.clear_non_dashboard_messages(client)
            await self.send_published_list(client)
            return

        if data == "menu:unpublished":
            await event.answer("جارٍ فحص الترندات غير المنشورة...")
            await self.clear_non_dashboard_messages(client)
            await self.send_unpublished_items(client)
            return

        if data.startswith("next:"):
            cat_key = data.split(":", 1)[1]
            item, wrapped = self.store.advance_card_trend(msg_id, cat_key)
            if not item:
                await event.answer("لا توجد ترندات إضافية في هذه الفئة حالياً.", alert=True)
                return
            img_path = resolve_trend_card_image(item, fetch_remote=False)
            caption = format_trend_card_caption(item)
            await client.edit_message(
                self.allowed_chat_id,
                msg_id,
                text=caption,
                file=str(img_path),
                buttons=card_inline_keyboard(cat_key),
            )
            await event.answer("تم الرجوع لأول ترند في القائمة." if wrapped else "تم عرض الترند التالي.")
            return

        if data.startswith("prod:"):
            cat_key = data.split(":", 1)[1]
            quota = int(self.config.telegram.get("daily_video_quota", 2))
            job, status = self.store.start_job_for_card(msg_id, cat_key, daily_quota=quota)
            if not job:
                if status == "quota_exceeded":
                    await event.answer(f"تم بلوغ الحد اليومي ({quota} فيديو). حاول غداً.", alert=True)
                elif status == "card_producing":
                    await event.answer("هذا الترند قيد التجهيز بالفعل.", alert=True)
                else:
                    await event.answer("تعذر بدء الإنتاج لهذه البطاقة.", alert=True)
                return
            await event.answer("بدأ تجهيز الفيديو في الخلفية 🎬")
            await self.clear_non_dashboard_messages(client)
            await self._send_tracked_text(
                client,
                (
                    f"⏳ بدأ الإنتاج الآلي للترند ({job['category_ar']}):\n"
                    f"📌 {job['trend_title']}\n"
                    f"📁 المجلد: tiktok/{job['run_id']}\n"
                    "سيتم جلب اللقطات وكتابة السيناريو وإرسال الفيديو للمراجعة هنا."
                ),
            )
            asyncio.create_task(self._run_pipeline_and_deliver_review(client, job))
            return

        if data.startswith("edit:"):
            job_id = data.split(":", 1)[1]
            job = self.store.enter_edit_feedback(job_id)
            if not job:
                await event.answer("لم يتم العثور على هذا المشروع.", alert=True)
                return
            await event.answer("أرسل التعديلات في رسالة نصية")
            await self.clear_non_dashboard_messages(client, preserve_edit_job=True)
            await self._send_tracked_text(
                client,
                (
                    f"✏️ اكتب التعديلات المطلوبة للفيديو ({job['trend_title']}) في رسالة عادية الآن:\n"
                    "(مثال: اجعل التعليق أسرع، أو استبدل الصور بلقطات أحدث)"
                ),
            )
            return

        if data.startswith("appr:"):
            job_id = data.split(":", 1)[1]
            job = self.store.state.get("jobs", {}).get(job_id)
            if not job:
                await event.answer("لم يتم العثور على هذا المشروع.", alert=True)
                return
            await event.answer("تمت الموافقة! جارٍ تجهيز النسخة الأصلية غير المضغوطة وحزمة النشر...")
            try:
                await client.edit_message(
                    self.allowed_chat_id,
                    msg_id,
                    buttons=empty_inline_keyboard(),
                )
            except Exception:
                pass

            rc, out_log = await _run_subprocess_async(
                [sys.executable, str(WORKSPACE_ROOT / "run_pipeline.py"), "--approve", job["run_id"]]
            )
            if rc != 0:
                await self._send_tracked_text(
                    client,
                    f"⚠️ حدث خطأ أثناء التصدير النهائي:\n{out_log[-300:]}",
                )
                return

            run_dir = self.config.tiktok_dir / job["run_id"]
            video_path = run_dir / "final_9x16.mp4"
            review_path = run_dir / "review_package.json"
            review_data = (
                json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
            )
            kit_path = _ensure_publish_kit(run_dir, review_data)

            if not bool(job.get("review_sent_raw", False)):
                await self.clear_non_dashboard_messages(client)
                if video_path.exists():
                    size_mb = video_path.stat().st_size / (1024 * 1024)
                    dur_sec = float(review_data.get("duration_sec") or 65.0)
                    gw = int(self.config.video.get("width", 2160))
                    gh = int(self.config.video.get("height", 3840))
                    res_label = f"{gw}x{gh} (4K)" if gw >= 2160 else f"{gw}x{gh}"
                    master_cap = (
                        f"📦 النسخة الأصلية الكاملة غير المضغوطة — {res_label} ({size_mb:.1f} MB)\n"
                        f"📌 {job['trend_title']}\n"
                        f"📁 {job['run_id']}"
                    )[:1020]
                    await self._send_video_file(
                        client,
                        video_path,
                        master_cap,
                        force_document=True,
                        width=gw,
                        height=gh,
                        duration_sec=dur_sec,
                        thumb_path=run_dir / "cover.jpg",
                    )
            else:
                await self.clear_non_dashboard_messages(client, keep_ids={msg_id})

            if kit_path and kit_path.exists():
                kmsg = await client.send_file(
                    self.allowed_chat_id,
                    str(kit_path),
                    caption=f"📎 ملف حزمة النشر: {kit_path.name}",
                    force_document=True,
                )
                self.store.track_message(_msg_id(kmsg))

            msg_html = _build_post_copy_message(run_dir, job["trend_title"], review_data)
            pub_msg = await self._send_tracked_text(
                client,
                msg_html,
                buttons=publish_confirm_keyboard(job_id),
                parse_mode="html",
            )
            self.store.approve_job(job_id, publish_message_id=_msg_id(pub_msg))
            return

        if data.startswith("pub:"):
            job_id = data.split(":", 1)[1]
            job, already_done = self.store.mark_job_published(job_id)
            if not job:
                await event.answer("المشروع غير موجود.", alert=True)
                return
            if already_done:
                await event.answer("تم تأكيد النشر مسبقاً ✅")
                return
            await self._finish_publish_and_maybe_cleanup(
                client,
                event,
                job["trend_title"],
                int(job.get("card_message_id") or 0),
            )
            return

        if data.startswith("pubrun:"):
            run_id = data.split(":", 1)[1]
            run_dir = self.config.tiktok_dir / run_id
            topic = run_id
            rev_file = run_dir / "review_package.json"
            if rev_file.exists():
                try:
                    topic = str(json.loads(rev_file.read_text(encoding="utf-8")).get("topic") or run_id)
                except Exception:
                    pass
            already_done = self.store.mark_run_published_direct(run_id, topic=topic)
            if already_done:
                await event.answer("تم تأكيد النشر مسبقاً ✅")
                return
            await self._finish_publish_and_maybe_cleanup(client, event, topic, 0)

    async def on_text_message(self, event: Any) -> None:
        if not self._authorized(event):
            return
        feedback = str(getattr(event, "raw_text", "") or "").strip()
        if not feedback:
            return
        self.store.reload()
        self.store.track_message(_msg_id(getattr(event, "message", event)))

        job = self.store.submit_edit_feedback(feedback)
        if not job:
            return
        client = event.client
        max_rounds = int(self.config.telegram.get("max_auto_edit_rounds", 3))
        await self.clear_non_dashboard_messages(client)
        if int(job.get("revision_round", 0)) > max_rounds:
            await self._send_tracked_text(
                client,
                f"🤖 تجاوزت التعديلات {max_rounds} محاولات ({job['revision_round']}) للترند ({job['trend_title']}).\n"
                "جارٍ فحص المشكلة وتصعيدها إلى نموذج الذكاء الاصطناعي...",
            )
            asyncio.create_task(self._escalate_edit_job_to_ai(client, job))
            return
        await self._send_tracked_text(
            client,
            f"⏳ تم استلام التعديلات للترند ({job['trend_title']})، وجارٍ إعادة التصيير في الخلفية...",
        )
        asyncio.create_task(
            self._run_pipeline_and_deliver_review(client, job, revision_feedback=feedback)
        )

    async def on_new_message(self, event: Any) -> None:
        if not self._authorized(event):
            return
        raw_text = str(getattr(event, "raw_text", "") or "").strip()
        if not raw_text:
            return
        cmd_match = re.match(r"^/(start|menu|trends|status)(?:@\w+)?(?:\s|$)", raw_text, flags=re.IGNORECASE)
        if cmd_match:
            cmd = cmd_match.group(1).lower()
            if cmd in ("start", "menu"):
                await self.cmd_start(event)
            elif cmd == "trends":
                await self.cmd_trends(event)
            elif cmd == "status":
                await self.cmd_status(event)
            return
        if raw_text.startswith("/"):
            return
        await self.on_text_message(event)

    def register_handlers(self, client: TelegramClient) -> None:
        client.add_event_handler(self.on_new_message, events.NewMessage(incoming=True))
        client.add_event_handler(self.on_callback, events.CallbackQuery())


async def _run_telethon_bot_once(bot_runner: InteractiveTikTokBot) -> None:
    assert bot_runner.creds is not None
    client = create_telethon_client(
        bot_runner.creds,
        session_path=bot_runner.session_path,
        connection_retries=-1,
        retry_delay=5,
        use_memory_fallback=False,
    )
    bot_runner.register_handlers(client)
    try:
        await client.start(bot_token=bot_runner.creds.bot_token)
        me = await client.get_me()
        username = getattr(me, "username", None) or str(getattr(me, "id", "bot"))
        logger.info(
            "Telethon MTProto bot connected as @%s (session=%s, cryptg=%s, whitelist=%s)",
            username,
            bot_runner.session_path,
            is_cryptg_available(),
            bot_runner.allowed_chat_id,
        )
        await client.run_until_disconnected()
    finally:
        await client.disconnect()


def run_polling_forever(bot_runner: InteractiveTikTokBot) -> None:
    """Keep Telethon MTProto connection resilient across laptop sleep/wake cycles and Wi-Fi drops."""
    backoff_sec = 5.0
    while True:
        started_at = time.monotonic()
        try:
            asyncio.run(_run_telethon_bot_once(bot_runner))
            logger.info("Telethon MTProto client exited cleanly.")
            break
        except KeyboardInterrupt:
            logger.info("Telegram bot stopped by KeyboardInterrupt.")
            break
        except Exception as exc:
            uptime = time.monotonic() - started_at
            safe_err = mask_secret(str(exc), bot_runner.token, bot_runner.api_hash)
            logger.warning(
                "Telethon connection interrupted after %.1fs (%s). Reconnecting in %.0fs...",
                uptime,
                safe_err,
                backoff_sec,
            )
            time.sleep(backoff_sec)
            backoff_sec = next_backoff_sec(backoff_sec, uptime_sec=uptime)


async def _send_cards_once_async(config_path: Path, state_path: Path) -> list[int]:
    bot_runner = InteractiveTikTokBot(config_path=config_path, state_path=state_path)
    assert bot_runner.creds is not None
    client = create_telethon_client(
        bot_runner.creds,
        session_path=DEFAULT_SENDER_SESSION_PATH,
        connection_retries=5,
        retry_delay=2,
        use_memory_fallback=True,
    )
    try:
        await client.start(bot_token=bot_runner.creds.bot_token)
        await bot_runner.clear_non_dashboard_messages(client)
        return await bot_runner.send_category_cards(client)
    finally:
        await client.disconnect()


async def _send_menu_once_async(config_path: Path, state_path: Path) -> int:
    bot_runner = InteractiveTikTokBot(config_path=config_path, state_path=state_path)
    assert bot_runner.creds is not None
    client = create_telethon_client(
        bot_runner.creds,
        session_path=DEFAULT_SENDER_SESSION_PATH,
        connection_retries=5,
        retry_delay=2,
        use_memory_fallback=True,
    )
    try:
        await client.start(bot_token=bot_runner.creds.bot_token)
        return await bot_runner.send_dashboard_menu(client)
    finally:
        await client.disconnect()


async def _deliver_review_run_once_async(config_path: Path, state_path: Path, run_id: str) -> dict[str, Any]:
    bot_runner = InteractiveTikTokBot(config_path=config_path, state_path=state_path)
    assert bot_runner.creds is not None
    bot_runner.store.reload()
    job = next(
        (j for j in bot_runner.store.state.get("jobs", {}).values() if j.get("run_id") == run_id),
        None,
    )
    if not job:
        raise RuntimeError(f"Job for run_id '{run_id}' not found in {state_path}")
    run_dir = bot_runner.config.tiktok_dir / run_id
    video_path = run_dir / "final_9x16.mp4"
    min_sec = float(bot_runner.config.video.get("hard_monetization_floor_sec", 61))
    gate = verify_video(video_path, min_sec=min_sec)
    if not gate.get("passed"):
        raise RuntimeError(f"Video gate failed for {run_id}: {gate.get('issues')}")

    send_raw = should_send_raw_on_review(bot_runner.config.telegram)
    size_mb = video_path.stat().st_size / (1024 * 1024)
    gw = int(gate.get("width", 1080))
    gh = int(gate.get("height", 1920))
    res_label = f"{gw}x{gh} (4K)" if gw >= 2160 else f"{gw}x{gh}"
    mode_tag = "نسخة معدلة" if int(job.get("revision_round", 0)) > 0 else "جاهز للمراجعة"
    cap = (
        f"🎬 معاينة الفيديو ({mode_tag} — صوت إنجليزي صافي 100%) — {res_label} ({size_mb:.1f} MB)\n"
        f"📌 {job['trend_title']}\n"
        f"🏷️ الفئة: {job.get('category_ar', job['category'])} | 🌐 EN | ⏱️ {gate['duration_sec']:.2f}s\n"
        f"📁 {run_id}"
    )[:1020]

    client = create_telethon_client(
        bot_runner.creds,
        session_path=DEFAULT_SENDER_SESSION_PATH,
        connection_retries=5,
        retry_delay=2,
        use_memory_fallback=True,
    )
    try:
        await client.start(bot_token=bot_runner.creds.bot_token)
        await bot_runner.clear_non_dashboard_messages(client)
        msg = await bot_runner._send_video_file(
            client,
            video_path,
            cap,
            force_document=send_raw,
            width=gw,
            height=gh,
            duration_sec=float(gate.get("duration_sec", 65)),
            buttons=review_inline_keyboard(job["job_id"]),
            thumb_path=run_dir / "cover.jpg",
        )
        mid = _msg_id(msg)
        bot_runner.store.mark_job_review_ready(job["job_id"], review_message_id=mid, review_sent_raw=send_raw)
        queue_rel = str(bot_runner.config.telegram.get("ai_escalation_queue_file", "data/ai_escalation_queue.json"))
        resolve_ai_escalation_ticket(run_id, WORKSPACE_ROOT / queue_rel)
        return {
            "ok": True,
            "transport": "telethon_mtproto",
            "run_id": run_id,
            "job_id": job["job_id"],
            "review_message_id": mid,
            "force_document": send_raw,
        }
    finally:
        await client.disconnect()


def main() -> int:
    os.chdir(str(WORKSPACE_ROOT))
    configure_logging(DEFAULT_LOG_PATH)

    parser = argparse.ArgumentParser(description="Viral Studio Pro 100% Free Telethon (MTProto) Telegram Bot.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config.yaml")
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH), help="Path to telegram_cards_state.json")
    parser.add_argument(
        "--send-cards-once",
        action="store_true",
        help="Push fresh category trend photo cards once and exit cleanly",
    )
    parser.add_argument(
        "--send-menu-once",
        action="store_true",
        help="Push the handsome 4-button Control Menu dashboard once and exit cleanly",
    )
    parser.add_argument(
        "--deliver-review-run",
        default=None,
        help="Deliver an already-rendered run_id to Telegram for review and update its job state",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = WORKSPACE_ROOT / cfg_path
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = WORKSPACE_ROOT / state_path

    try:
        if args.send_cards_once:
            sent_ids = asyncio.run(_send_cards_once_async(cfg_path, state_path))
            print(json.dumps({"ok": True, "sent_card_message_ids": sent_ids}))
            return 0

        if args.send_menu_once:
            dash_id = asyncio.run(_send_menu_once_async(cfg_path, state_path))
            print(json.dumps({"ok": True, "dashboard_message_id": dash_id}))
            return 0

        if args.deliver_review_run:
            res = asyncio.run(_deliver_review_run_once_async(cfg_path, state_path, args.deliver_review_run))
            print(json.dumps(res))
            return 0
    except MissingTelegramCredentialsError as exc:
        logger.error("%s", exc)
        if sys.stderr is not None:
            print(str(exc), file=sys.stderr)
        return 1

    lock = SingleInstanceLock(pid_path=DEFAULT_PID_PATH, mutex_name=DEFAULT_MUTEX_NAME)
    if not lock.acquire():
        logger.warning("Duplicate telegram_bot.py launch detected; exiting cleanly without conflict.")
        return 0
    atexit.register(lock.release)

    try:
        bot_runner = InteractiveTikTokBot(config_path=cfg_path, state_path=state_path)
        logger.info(
            "Starting Viral Studio Pro Telethon (MTProto) bot (cwd=%s, whitelist=%s, session=%s, cryptg=%s, python=%s)",
            os.getcwd(),
            bot_runner.allowed_chat_id,
            bot_runner.session_path,
            is_cryptg_available(),
            sys.executable,
        )
        run_polling_forever(bot_runner)
    except MissingTelegramCredentialsError as exc:
        logger.error("%s", exc)
        if sys.stderr is not None:
            print(str(exc), file=sys.stderr)
        return 1
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
