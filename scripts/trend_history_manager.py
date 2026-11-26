from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = WORKSPACE_ROOT / "data" / "trends_history.json"
DEFAULT_PIPELINE_LOG_PATH = WORKSPACE_ROOT / "data" / "history_log.json"
DEFAULT_LATEST_MD_PATH = WORKSPACE_ROOT / "data" / "latest_trends.md"

KIDS_BLOCKLIST = (
    "kids",
    "for children",
    "toddler",
    "cocomelon",
    "nursery rhyme",
    "peppa pig",
    "baby shark",
    "cartoon for kids",
    "kids song",
)

DANCE_LIPSYNC_BLOCKLIST = (
    "dance challenge",
    "tiktok dance",
    "lip sync",
    "lipsync",
    "choreography challenge",
    "twerk",
    "dance trend",
    "sing along",
)

COPYRIGHT_TRAP_BLOCKLIST = (
    "full movie",
    "leaked clip",
    "music video",
    "official music video",
    "nfl highlights",
    "nba highlights",
    "ufc fight",
    "celebrity gossip",
    "deepfake",
    "live stream",
    "tv episode",
    "full episode",
    "reupload",
)

STORYTELLING_CUES = (
    "ai",
    "openai",
    "claude",
    "gemini",
    "tech",
    "technology",
    "software",
    "chip",
    "cyber",
    "agent",
    "game",
    "gaming",
    "zelda",
    "switch",
    "playstation",
    "xbox",
    "movie",
    "film",
    "box office",
    "series",
    "season",
    "netflix",
    "anime",
    "manga",
    "history",
    "psychology",
    "science",
    "space",
    "mystery",
    "explained",
    "economy",
    "why",
    "billion",
    "million",
    "record",
    "breakthrough",
    "launch",
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9\u0600-\u06FF]{3,}", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _load_history(history_path: Path, pipeline_log_path: Path) -> dict[str, Any]:
    if history_path.exists():
        data = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data

    entries: list[dict[str, Any]] = []
    if pipeline_log_path.exists():
        try:
            runs = json.loads(pipeline_log_path.read_text(encoding="utf-8"))
            for r in runs:
                topic = str(r.get("topic") or "").strip()
                if not topic:
                    continue
                status = "produced" if r.get("status") in ("APPROVED_AND_PACKAGED", "READY_FOR_REVIEW") else "discovered"
                entries.append(
                    {
                        "topic": topic,
                        "source": "pipeline_history_log",
                        "status": status,
                        "run_id": r.get("run_id"),
                        "reason": f"Seeded from history_log.json ({r.get('status')})",
                        "timestamp": r.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception:
            pass

    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    history_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _rejection_reason(
    candidate: dict[str, Any], past_topics: list[str], sim_threshold: float = 0.50
) -> str | None:
    title = str(candidate.get("topic") or candidate.get("title") or "").strip()
    summary = str(candidate.get("summary") or "").strip()
    category = str(candidate.get("category") or "").strip().lower()
    combined = f"{title} {summary} {category}".lower()

    if len(title) < 4:
        return "too_short"
    for kw in KIDS_BLOCKLIST:
        if kw in combined:
            return f"kids_content:{kw}"
    for kw in DANCE_LIPSYNC_BLOCKLIST:
        if kw in combined:
            return f"dance_or_lipsync:{kw}"
    for kw in COPYRIGHT_TRAP_BLOCKLIST:
        if kw in combined:
            return f"copyright_trap:{kw}"

    norm_title = title.lower()
    for past in past_topics:
        norm_past = past.lower()
        if not norm_past:
            continue
        if len(norm_past) >= 6 and (norm_title in norm_past or norm_past in norm_title):
            return f"duplicate_past:{past}"
        if _jaccard(title, past) >= sim_threshold:
            return f"duplicate_similarity:{past}"
    return None


def score_candidate(candidate: dict[str, Any]) -> float:
    title = str(candidate.get("topic") or candidate.get("title") or "").strip()
    summary = str(candidate.get("summary") or "").strip()
    sources = candidate.get("sources") or ([candidate.get("source")] if candidate.get("source") else [])
    combined = f"{title} {summary}".lower()

    cue_hits = sum(1 for c in STORYTELLING_CUES if c in combined)
    cue_bonus = min(0.24, cue_hits * 0.05)

    source_bonus = 0.08 if len(set(sources)) >= 2 else 0.03
    if any("tiktok" in str(s).lower() for s in sources):
        source_bonus += 0.06
    if any("youtube" in str(s).lower() for s in sources):
        source_bonus += 0.04

    has_video = bool(candidate.get("video_footage_available", True))
    video_bonus = 0.06 if has_video else 0.0

    traffic = int(candidate.get("approx_traffic") or 0)
    traffic_bonus = min(0.14, math.log10(max(traffic, 1000)) * 0.025) if traffic > 0 else 0.02

    base = 0.50
    score = base + cue_bonus + source_bonus + video_bonus + traffic_bonus
    return round(max(0.10, min(0.99, score)), 4)


def process_candidates(
    candidates: list[dict[str, Any]],
    history_path: Path = DEFAULT_HISTORY_PATH,
    pipeline_log_path: Path = DEFAULT_PIPELINE_LOG_PATH,
    latest_md_path: Path = DEFAULT_LATEST_MD_PATH,
    region: str = "Global / US",
    sources_checked: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    history = _load_history(history_path, pipeline_log_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    past_topics = [
        str(e.get("topic", "")).strip()
        for e in history.get("entries", [])
        if e.get("status") in ("produced", "published", "rejected") and e.get("topic")
    ]
    if pipeline_log_path.exists():
        try:
            for r in json.loads(pipeline_log_path.read_text(encoding="utf-8")):
                if r.get("topic"):
                    past_topics.append(str(r["topic"]))
        except Exception:
            pass

    rejected: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen_now: list[str] = []

    for raw in candidates:
        title = str(raw.get("topic") or raw.get("title") or "").strip()
        if not title:
            continue
        reason = _rejection_reason(raw, past_topics + seen_now)
        if reason:
            rec = {
                "topic": title,
                "source": raw.get("source", "browser"),
                "status": "rejected",
                "reason": reason,
                "timestamp": now_iso,
            }
            rejected.append(rec)
            history["entries"].append(rec)
            continue

        seen_now.append(title)
        score = score_candidate(raw)
        item = {
            "topic": title,
            "category": raw.get("category", "general"),
            "source": raw.get("source", "browser"),
            "sources": raw.get("sources", [raw.get("source", "browser")]),
            "summary": str(raw.get("summary", "")).strip(),
            "suggested_angle_ar": str(raw.get("suggested_angle_ar", "")).strip(),
            "hook_idea_en": str(raw.get("hook_idea_en", "")).strip(),
            "clip_keywords": list(raw.get("clip_keywords", [])),
            "approx_traffic": int(raw.get("approx_traffic") or 0),
            "score": score,
        }
        eligible.append(item)
        history["entries"].append(
            {
                "topic": title,
                "category": item["category"],
                "source": item["source"],
                "status": "discovered",
                "score": score,
                "timestamp": now_iso,
            }
        )

    ranked = sorted(eligible, key=lambda x: (x["score"], x["approx_traffic"]), reverse=True)
    top_pick = ranked[0] if ranked else None
    backups = ranked[1:3]

    if top_pick:
        for entry in reversed(history["entries"]):
            if entry.get("topic") == top_pick["topic"] and entry.get("timestamp") == now_iso:
                entry["status"] = "selected"
                break

    history["updated_at"] = now_iso
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_latest_trends_md(
        latest_md_path,
        region=region,
        sources_checked=sources_checked or [],
        top_pick=top_pick,
        backups=backups,
        rejected=rejected,
        total_scanned=len(candidates),
    )

    return {
        "total_scanned": len(candidates),
        "eligible_count": len(ranked),
        "rejected": rejected,
        "top_pick": top_pick,
        "backups": backups,
        "history_path": str(history_path),
        "latest_md_path": str(latest_md_path),
    }


def mark_trend_published(
    trend_id: str,
    topic: str,
    category: str,
    run_id: str = "",
    history_path: Path = DEFAULT_HISTORY_PATH,
    pipeline_log_path: Path = DEFAULT_PIPELINE_LOG_PATH,
    queue_path: Path = WORKSPACE_ROOT / "data" / "trends_queue.json",
) -> dict[str, Any]:
    """Record a trend as permanently published in trends_history.json and mark DONE in queue."""
    now_iso = datetime.now(timezone.utc).isoformat()
    history = _load_history(history_path, pipeline_log_path)
    entries: list[dict[str, Any]] = history.setdefault("entries", [])
    already = any(
        e.get("status") == "published"
        and ((trend_id and e.get("trend_id") == trend_id) or e.get("topic", "").lower() == topic.lower())
        for e in entries
    )
    if not already:
        entries.append(
            {
                "trend_id": trend_id,
                "topic": topic,
                "category": category,
                "run_id": run_id,
                "source": "telegram_publish_confirm",
                "status": "published",
                "timestamp": now_iso,
            }
        )
    history["updated_at"] = now_iso
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    if queue_path.exists():
        try:
            q_data = json.loads(queue_path.read_text(encoding="utf-8"))
            changed = False
            for item in q_data.get("trends", []):
                if (trend_id and item.get("id") == trend_id) or (
                    topic and str(item.get("topic", "")).strip().lower() == topic.strip().lower()
                ):
                    item["status"] = "DONE"
                    if run_id:
                        item["run_id"] = run_id
                    changed = True
            if changed:
                q_data["updated_at"] = now_iso[:10]
                queue_path.write_text(json.dumps(q_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return {"ok": True, "trend_id": trend_id, "topic": topic, "status": "published", "already_logged": already}


def _write_latest_trends_md(
    path: Path,
    region: str,
    sources_checked: list[dict[str, str]],
    top_pick: dict[str, Any] | None,
    backups: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    total_scanned: int,
) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Viral Studio Pro — أحدث ترندات TikTok المكتشفة عبر المتصفح",
        "",
        f"- **تاريخ الفحص:** `{now_str}`",
        f"- **النطاق الجغرافي:** `{region}`",
        f"- **إجمالي المرشحين المفحوصين:** `{total_scanned}` | **المستبعدون (تكرار/أطفال/رقص/حقوق نشر):** `{len(rejected)}`",
        "",
    ]
    if sources_checked:
        lines.append("## حالة المصادر العامة المفحوصة ($0 Free)")
        for src in sources_checked:
            lines.append(f"- **{src.get('name', 'Source')}** (`{src.get('origin', '')}`): `{src.get('status', '')}` — {src.get('note', '')}")
        lines.append("")

    if top_pick:
        kw_str = ", ".join(f"`{k}`" for k in (top_pick.get("clip_keywords") or [top_pick["topic"]]))
        angle_ar = top_pick.get("suggested_angle_ar") or top_pick.get("summary") or "تحليل قصصي/اقتصادي مدته 65–85 ثانية مع مقارنة بصرية."
        hook_en = top_pick.get("hook_idea_en") or f"Everyone is talking about {top_pick['topic']}, but almost nobody saw this hidden detail."
        lines.extend(
            [
                "## الترند #1 المختار للإنتاج (بانتظار موافقتك)",
                f"- **الموضوع:** `{top_pick['topic']}`",
                f"- **الفئة:** `{top_pick['category']}` | **التقييم:** `{top_pick['score']:.2f}` | **المصادر:** `{', '.join(top_pick.get('sources', []))}`",
                f"- **الملخص السريع:** {top_pick.get('summary', '')}",
                f"- **الزاوية المقترحة (`65–85s`):** {angle_ar}",
                f"- **فكرة خطاف الـ 3 ثوانٍ (`Hook`):** `{hook_en}`",
                f"- **كلمات البحث عن مقاطع الفيديو (`yt-dlp` / B-roll):** {kw_str}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## لا يوجد ترند مؤهل حالياً",
                "- استبعدت الفلاتر جميع المرشحين الحاليين (تكرار أو مخاطر حقوق أو محتوى غير قصصي).",
                "",
            ]
        )

    if backups:
        lines.append("## خياران احتياطيان مؤهلان")
        for idx, b in enumerate(backups, start=2):
            lines.extend(
                [
                    f"### #{idx} — `{b['topic']}` (`{b['category']}` | Score: `{b['score']:.2f}`)",
                    f"- **السبب والزاوية:** {b.get('suggested_angle_ar') or b.get('summary') or 'مناسب لسرد تحليلي 65–85 ثانية.'}",
                    f"- **كلمات البحث:** {', '.join(f'`{k}`' for k in (b.get('clip_keywords') or [b['topic']]))}",
                    "",
                ]
            )

    if rejected:
        lines.append("## موضوعات مستبعدة تلقائياً")
        for r in rejected[:10]:
            lines.append(f"- `{r['topic']}` -> `{r['reason']}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate, filter, score, and log discovered TikTok trends.")
    parser.add_argument("--init-only", action="store_true", help="Initialize data/trends_history.json from history_log.json")
    parser.add_argument("--input", default=None, help="Path to JSON file containing discovered trend candidates")
    parser.add_argument("--region", default="Global / US", help="Target region label for report")
    args = parser.parse_args()

    if args.init_only:
        hist = _load_history(DEFAULT_HISTORY_PATH, DEFAULT_PIPELINE_LOG_PATH)
        print(json.dumps({"initialized": True, "entries": len(hist["entries"])}))
        return 0

    if not args.input:
        parser.error("--input JSON file is required unless --init-only is specified")

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    candidates = raw.get("candidates", raw) if isinstance(raw, dict) else raw
    sources_checked = raw.get("sources_checked", []) if isinstance(raw, dict) else []
    result = process_candidates(
        candidates=candidates,
        region=args.region,
        sources_checked=sources_checked,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
