from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.trend_history_manager import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_PIPELINE_LOG_PATH,
    _load_history,
    _rejection_reason,
)
from src.utils.language_guard import ensure_english_text, is_english_text

DEFAULT_SOURCES_PATH = WORKSPACE_ROOT / "data" / "trend_sources.json"
DEFAULT_QUEUE_PATH = WORKSPACE_ROOT / "data" / "trends_queue.json"
DEFAULT_LATEST_MD_PATH = WORKSPACE_ROOT / "data" / "latest_trends.md"

PILLAR_ORDER: tuple[str, ...] = ("movies", "series", "games", "ai")

DEFAULT_EXCLUDED_KEYWORDS: tuple[str, ...] = (
    "hardware review",
    "unboxing",
    "shopping guide",
    "buying guide",
    "hands-on review",
    "smartphone review",
    "laptop review",
    "pc build",
    "monitor review",
    "earbuds",
    "car review",
    "ev battery",
    "kids",
    "toddler",
    "nursery rhyme",
    "dance challenge",
    "lip sync",
    "full movie",
    "leaked clip",
    "music video",
    "nfl highlights",
    "nba highlights",
    "ufc fight",
    "celebrity gossip",
    "deepfake",
    "reupload",
)

PRIMARY_AI_DOMAINS = ("openai.com", "deepmind.google", "anthropic.com", "huggingface.co", "arxiv.org")
PRIMARY_ENTERTAINMENT_DOMAINS = (
    "variety.com", "deadline.com", "hollywoodreporter.com", "ign.com",
    "eurogamer.net", "blog.playstation.com", "steampowered.com", "pcgamer.com",
)


def load_trend_sources(sources_path: Path | None = None) -> dict[str, Any]:
    path = sources_path or DEFAULT_SOURCES_PATH
    if not path.exists():
        return {"pillars": {}, "global_signals": [], "excluded_keywords": list(DEFAULT_EXCLUDED_KEYWORDS)}
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_html_text(raw: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", html.unescape(raw or ""))
    return re.sub(r"\s+", " ", no_tags).strip()


def parse_feed_xml(
    xml_bytes: bytes,
    source_name: str,
    category: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Parse both RSS (<item>) and Atom (<entry>) XML feeds cleanly."""
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items: list[dict[str, Any]] = []
    rss_items = root.findall(".//item")
    if rss_items:
        for node in rss_items[:limit]:
            title = _clean_html_text(node.findtext("title") or "")
            desc = _clean_html_text(node.findtext("description") or "")[:240]
            link = (node.findtext("link") or "").strip()
            pub = (node.findtext("pubDate") or "").strip()
            if title:
                items.append(
                    {
                        "title": title,
                        "summary": desc,
                        "url": link,
                        "published": pub,
                        "source": source_name,
                        "category": category,
                    }
                )
        return items

    # Atom (<entry>) handling (namespace-agnostic)
    entries = [el for el in root.iter() if el.tag.lower().endswith("entry")]
    for entry in entries[:limit]:
        title = ""
        summary = ""
        link = ""
        pub = ""
        for child in entry:
            tag = child.tag.lower().split("}")[-1]
            if tag == "title" and child.text:
                title = _clean_html_text(child.text)
            elif tag in ("summary", "content") and child.text and not summary:
                summary = _clean_html_text(child.text)[:240]
            elif tag == "link":
                href = child.attrib.get("href") or child.text or ""
                if href and not link:
                    link = href.strip()
            elif tag in ("updated", "published") and child.text and not pub:
                pub = child.text.strip()
        if title:
            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": link,
                    "published": pub,
                    "source": source_name,
                    "category": category,
                }
            )
    return items


def is_excluded_topic(text: str, excluded_keywords: list[str] | tuple[str, ...]) -> str | None:
    low = (text or "").lower()
    for kw in excluded_keywords:
        clean_kw = str(kw).strip().lower()
        if clean_kw and clean_kw in low:
            return clean_kw
    return None


def classify_ai_claim_status(text: str) -> str:
    """Distinguish released features from announcements, demos, synthetic visuals, and rumors."""
    low = (text or "").lower()
    if any(w in low for w in ("rumor", "leak", "alleged", "unconfirmed", "reportedly", "speculation")):
        return "RUMOR / SPECULATION"
    if any(w in low for w in ("ai-generated", "synthetic", "deepfake", "reconstruct", "gaussian splat", "sora", "veo", "kling", "pre-vis")):
        return "SYNTHETIC / AI-GENERATED"
    if any(w in low for w in ("demo", "preview", "announced", "unveiled", "paper", "research", "upcoming", "prototype")):
        return "ANNOUNCEMENT / DEMO"
    return "RELEASED FEATURE"


def score_candidate_editorial(item: dict[str, Any]) -> int:
    """7-factor editorial rubric (max 35) matching Viral Studio Pro rules."""
    title = str(item.get("title") or "").strip()
    summary = str(item.get("summary") or "").strip()
    url = str(item.get("url") or "").lower()
    blob = f"{title} {summary}".lower()

    freshness = 5 if item.get("published") or item.get("source") else 3
    relevance = 5 if any(w in blob for w in ("trailer", "season", "episode", "game", "update", "ai", "model", "film", "movie", "series")) else 4
    discussion = 5 if any(w in blob for w in ("why", "secret", "hidden", "vs", "detail", "reveal", "change", "record", "debate", "breakdown")) else 3
    angle = 5 if any(w in blob for w in ("behind", "detail", "theory", "how", "compare", "mistake", "actually", "format", "engine")) else 3
    footage = 5 if any(w in blob for w in ("trailer", "gameplay", "demo", "teaser", "clip", "preview", "video", "showcase")) else 4
    is_primary = any(dom in url for dom in PRIMARY_AI_DOMAINS + PRIMARY_ENTERTAINMENT_DOMAINS)
    reliability = 5 if is_primary else 4
    effort = 5 if len(title) >= 16 else 3

    return freshness + relevance + discussion + angle + footage + reliability + effort


def _fetch_single_feed(feed_spec: dict[str, str], timeout_sec: int = 8) -> list[dict[str, Any]]:
    url = str(feed_spec.get("url") or "").strip()
    name = str(feed_spec.get("name") or url).strip()
    category = str(feed_spec.get("category") or "general").strip().lower()
    if not url.startswith("http"):
        return []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ViralStudioProTrendBot/2.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw_bytes = resp.read()
        return parse_feed_xml(raw_bytes, source_name=name, category=category, limit=6)
    except Exception:
        return []


def _infer_pillar_from_text(text: str, default: str = "movies") -> str:
    low = (text or "").lower()
    if any(w in low for w in ("ai ", "llm", "openai", "claude", "anthropic", "deepmind", "diffusion", "world model", "neural", "gaussian")):
        return "ai"
    if any(w in low for w in ("game", "gaming", "gta", "zelda", "nintendo", "playstation", "xbox", "steam", "warcraft", "fortnite", "patch")):
        return "games"
    if any(w in low for w in ("series", "season", "episode", "netflix", "prime video", "apple tv", "hbo", "showrunner", "finale")):
        return "series"
    return default


def fetch_live_trends(
    sources_path: Path | None = None,
    history_path: Path | None = None,
    pipeline_log_path: Path | None = None,
    queue_path: Path | None = None,
    latest_md_path: Path | None = None,
    per_category: int = 5,
    max_workers: int = 8,
    offline_items: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Scan all sources in data/trend_sources.json, filter, score, and refresh queue + latest_trends.md."""
    registry = load_trend_sources(sources_path)
    excluded_kw = list(registry.get("excluded_keywords") or DEFAULT_EXCLUDED_KEYWORDS)
    pillars_cfg = registry.get("pillars") or {}

    hist_p = history_path or DEFAULT_HISTORY_PATH
    log_p = pipeline_log_path or DEFAULT_PIPELINE_LOG_PATH
    q_p = queue_path or DEFAULT_QUEUE_PATH
    md_p = latest_md_path or DEFAULT_LATEST_MD_PATH

    history = _load_history(hist_p, log_p)
    past_topics = [
        str(e.get("topic") or "").strip()
        for e in history.get("entries", [])
        if e.get("status") in ("produced", "published", "rejected") and e.get("topic")
    ]

    raw_items: list[dict[str, Any]] = []
    if offline_items is not None:
        raw_items.extend(offline_items)
    else:
        feed_jobs: list[dict[str, str]] = []
        for cat_key in PILLAR_ORDER:
            for f in (pillars_cfg.get(cat_key, {}).get("rss_feeds") or []):
                feed_jobs.append({"name": str(f.get("name", "")), "url": str(f.get("url", "")), "category": cat_key})
        for gs in (registry.get("global_signals") or []):
            if str(gs.get("type", "")).lower() in ("rss", "atom"):
                feed_jobs.append({
                    "name": str(gs.get("name", "")),
                    "url": str(gs.get("url", "")),
                    "category": str(gs.get("category") or "general"),
                })
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(feed_jobs) or 1))) as pool:
            futs = [pool.submit(_fetch_single_feed, spec) for spec in feed_jobs]
            for fut in as_completed(futs):
                raw_items.extend(fut.result() or [])

    by_pillar: dict[str, list[dict[str, Any]]] = {k: [] for k in PILLAR_ORDER}
    rejected: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    for item in raw_items:
        title = str(item.get("title") or "").strip()
        if not is_english_text(title):
            title = ensure_english_text(title, fallback="", allow_translate=True)
        if len(title) < 10:
            continue
        t_low = title.lower()
        if t_low in seen_titles:
            continue
        seen_titles.add(t_low)

        summary = str(item.get("summary") or "").strip()
        if summary and not is_english_text(summary):
            summary = ensure_english_text(summary, fallback="", allow_translate=True)
        combined = f"{title} {summary}"
        excl = is_excluded_topic(combined, excluded_kw)
        if excl:
            rejected.append({"title": title, "reason": f"excluded:{excl}"})
            continue
        dup_reason = _rejection_reason({"title": title}, past_topics, sim_threshold=0.50)
        if dup_reason:
            rejected.append({"title": title, "reason": dup_reason})
            continue

        cat = str(item.get("category") or "").strip().lower()
        if cat not in PILLAR_ORDER:
            cat = _infer_pillar_from_text(combined, default="movies")

        score = score_candidate_editorial(item)
        ai_status = classify_ai_claim_status(combined) if cat == "ai" else ""
        by_pillar[cat].append(
            {
                "id": "live_" + hashlib.md5(f"{cat}:{t_low}".encode("utf-8")).hexdigest()[:10],
                "category": cat,
                "topic": title,
                "status": "PENDING",
                "priority": min(99, 60 + score),
                "editorial_score": score,
                "ai_claim_status": ai_status,
                "summary": summary or title,
                "source": str(item.get("source") or "live_feed"),
                "url": str(item.get("url") or ""),
            }
        )

    selected_by_pillar: dict[str, list[dict[str, Any]]] = {}
    for cat in PILLAR_ORDER:
        ranked = sorted(by_pillar[cat], key=lambda x: x.get("editorial_score", 0), reverse=True)
        selected_by_pillar[cat] = ranked[:per_category]

    if not dry_run:
        existing_done: list[dict[str, Any]] = []
        existing_pending: list[dict[str, Any]] = []
        if q_p.exists():
            try:
                old_q = json.loads(q_p.read_text(encoding="utf-8"))
                for row in old_q.get("trends", []):
                    if not isinstance(row, dict):
                        continue
                    r_cat = str(row.get("category", "")).lower()
                    r_stat = str(row.get("status", "PENDING")).upper()
                    if r_stat == "DONE":
                        existing_done.append(row)
                    elif r_cat in PILLAR_ORDER and not str(row.get("id", "")).startswith("live_"):
                        existing_pending.append(row)
            except Exception:
                pass

        live_rows = [row for cat in PILLAR_ORDER for row in selected_by_pillar[cat]]
        live_titles = {str(r["topic"]).lower() for r in live_rows}
        merged_pending = live_rows + [r for r in existing_pending if str(r.get("topic", "")).lower() not in live_titles]
        q_p.parent.mkdir(parents=True, exist_ok=True)
        q_p.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "categories": list(PILLAR_ORDER),
                    "trends": existing_done + merged_pending,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        md_lines = [
            f"# تقرير الترندات الحي من المصادر المعتمدة ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
            "",
            "- **الركائز الأربع:** `movies` | `series` | `games` | `ai`",
            f"- **إجمالي العناصر المفحوصة:** `{len(raw_items)}` | **المرفوضة (عتاد/تسوق/تكرار/حقوق):** `{len(rejected)}`",
            "",
        ]
        for cat in PILLAR_ORDER:
            label_ar = str(pillars_cfg.get(cat, {}).get("label_ar") or cat)
            md_lines.append(f"## {label_ar} (`{cat}`)")
            rows = selected_by_pillar[cat]
            if not rows:
                md_lines.append("- لا توجد عناصر جديدة صالحة في هذه الجولة (سيتم الاعتماد على قائمة الانتظار الحالية).")
            for row in rows:
                badge = f" [{row['ai_claim_status']}]" if row.get("ai_claim_status") else ""
                url_part = f" — {row['url']}" if row.get("url") else ""
                md_lines.append(
                    f"- **{row['topic']}**{badge} (Score: `{row['editorial_score']}/35` | Source: `{row['source']}`){url_part}"
                )
            md_lines.append("")
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md_p.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "ok": True,
        "scanned_count": len(raw_items),
        "rejected_count": len(rejected),
        "selected_by_pillar": selected_by_pillar,
        "queue_path": str(q_p),
        "latest_md_path": str(md_p),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch live 4-pillar trends from data/trend_sources.json ($0).")
    parser.add_argument("--sources", default=str(DEFAULT_SOURCES_PATH))
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = fetch_live_trends(
        sources_path=Path(args.sources),
        per_category=args.per_category,
        dry_run=args.dry_run,
    )
    summary = {
        "ok": res["ok"],
        "scanned_count": res["scanned_count"],
        "rejected_count": res["rejected_count"],
        "pillars": {k: len(v) for k, v in res["selected_by_pillar"].items()},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
