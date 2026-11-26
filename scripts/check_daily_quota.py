from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config

DEFAULT_DAILY_STATE_PATH = PROJECT_ROOT / "data" / "daily_run_state.json"


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_daily_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("days"), dict):
                return payload
        except Exception:
            pass
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "days": {}}


def _save_daily_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_daily_quota_status(
    state_path: Path = DEFAULT_DAILY_STATE_PATH,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config(PROJECT_ROOT / "config.yaml")
    limit = int(cfg.telegram.get("daily_video_quota", 2))
    state = _load_daily_state(state_path)
    today = _today_key()
    day_rec = state["days"].get(today, {})
    used = int(day_rec.get("count", 0))
    return {
        "date": today,
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
        "allowed": used < limit,
        "runs": list(day_rec.get("runs", [])),
        "state_path": str(state_path),
    }


def record_daily_video(
    run_id: str,
    topic: str = "",
    state_path: Path = DEFAULT_DAILY_STATE_PATH,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    cfg = config or load_config(PROJECT_ROOT / "config.yaml")
    limit = int(cfg.telegram.get("daily_video_quota", 2))
    state = _load_daily_state(state_path)
    today = _today_key()
    day_rec = state["days"].setdefault(today, {"count": 0, "runs": []})
    runs: list[dict[str, Any]] = day_rec.setdefault("runs", [])
    if not any(r.get("run_id") == run_id and run_id for r in runs):
        day_rec["count"] = int(day_rec.get("count", 0)) + 1
        runs.append(
            {
                "run_id": run_id,
                "topic": topic,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    _save_daily_state(state_path, state)
    used = int(day_rec["count"])
    return {
        "date": today,
        "used": used,
        "limit": limit,
        "remaining": max(0, limit - used),
        "allowed": used < limit,
        "state_path": str(state_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Portable zero-token daily quota tracker (data/daily_run_state.json).")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--state", default=str(DEFAULT_DAILY_STATE_PATH), help="Path to daily_run_state.json")
    parser.add_argument("--record-run", default=None, help="Optionally record a completed/published run_id")
    parser.add_argument("--topic", default="", help="Optional topic label when recording a run")
    parser.add_argument("--enforce", action="store_true", help="Exit 2 if daily quota is already reached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    state_path = Path(args.state)
    if args.record_run:
        res = record_daily_video(args.record_run, topic=args.topic, state_path=state_path, config=cfg)
    else:
        res = get_daily_quota_status(state_path=state_path, config=cfg)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.enforce and not res["allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
