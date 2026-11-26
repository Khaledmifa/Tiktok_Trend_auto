from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9\u0600-\u06FF]{3,}", text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute lexical Jaccard similarity between two texts (0.0 to 1.0)."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


class MemoryStore:
    """Lightweight SQLite + JSON memory log to prevent duplicate topics, hooks, and scripts."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "pipeline_memory.db"
        self.json_log_path = self.data_dir / "history_log.json"
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    niche TEXT NOT NULL,
                    topic TEXT,
                    hook TEXT,
                    script_text TEXT,
                    video_path TEXT,
                    duration_sec REAL,
                    status TEXT NOT NULL,
                    approved_at TEXT,
                    publish_info_json TEXT
                )
                """
            )
            conn.commit()

    def record_run_start(self, run_id: str, niche: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs (run_id, created_at, niche, status)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, now, niche, "STARTED"),
            )
            conn.commit()
        self._sync_json_log()

    def update_run_script(
        self, run_id: str, topic: str, hook: str, script_text: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET topic = ?, hook = ?, script_text = ?, status = ?
                WHERE run_id = ?
                """,
                (topic, hook, script_text, "SCRIPTED", run_id),
            )
            conn.commit()
        self._sync_json_log()

    def update_run_review_ready(
        self, run_id: str, video_path: str, duration_sec: float, status: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET video_path = ?, duration_sec = ?, status = ?
                WHERE run_id = ?
                """,
                (video_path, duration_sec, status, run_id),
            )
            conn.commit()
        self._sync_json_log()

    def mark_approved(self, run_id: str, publish_info: dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, approved_at = ?, publish_info_json = ?
                WHERE run_id = ?
                """,
                ("APPROVED_AND_PACKAGED", now, json.dumps(publish_info, ensure_ascii=False), run_id),
            )
            conn.commit()
        self._sync_json_log()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def get_past_topics_and_scripts(
        self,
        exclude_run_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> list[dict[str, str]]:
        with self._connect() as conn:
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                rows = conn.execute(
                    f"SELECT run_id, topic, hook, script_text FROM runs WHERE topic IS NOT NULL AND status IN ({placeholders})",
                    tuple(statuses),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, topic, hook, script_text FROM runs WHERE topic IS NOT NULL AND status != 'QA_FAILED'"
                ).fetchall()
        results: list[dict[str, str]] = []
        for row in rows:
            if exclude_run_id and row["run_id"] == exclude_run_id:
                continue
            results.append(
                {
                    "run_id": row["run_id"],
                    "topic": row["topic"] or "",
                    "hook": row["hook"] or "",
                    "script_text": row["script_text"] or "",
                }
            )
        return results

    def max_similarity_against_history(
        self,
        candidate_text: str,
        exclude_run_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> float:
        past = self.get_past_topics_and_scripts(
            exclude_run_id=exclude_run_id, statuses=statuses
        )
        max_sim = 0.0
        for item in past:
            combined = f"{item['topic']} {item['hook']} {item['script_text']}"
            sim = jaccard_similarity(candidate_text, combined)
            if sim > max_sim:
                max_sim = sim
        return round(max_sim, 4)

    def max_topic_similarity_against_history(
        self,
        topic_text: str,
        exclude_run_id: str | None = None,
        statuses: tuple[str, ...] | None = None,
    ) -> float:
        past = self.get_past_topics_and_scripts(
            exclude_run_id=exclude_run_id, statuses=statuses
        )
        max_sim = 0.0
        for item in past:
            sim = jaccard_similarity(topic_text, item["topic"])
            if sim > max_sim:
                max_sim = sim
        return round(max_sim, 4)

    def delete_runs(self, run_ids: list[str]) -> int:
        clean_ids = [str(r).strip() for r in run_ids if str(r).strip()]
        if not clean_ids:
            return 0
        placeholders = ",".join("?" for _ in clean_ids)
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM runs WHERE run_id IN ({placeholders})",
                tuple(clean_ids),
            )
            deleted = int(cur.rowcount or 0)
            conn.commit()
        self._sync_json_log()
        return deleted

    def _sync_json_log(self) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id, created_at, niche, topic, hook, video_path, duration_sec, status, approved_at FROM runs ORDER BY created_at DESC"
            ).fetchall()
        payload = [dict(r) for r in rows]
        self.json_log_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
