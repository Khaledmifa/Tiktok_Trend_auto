from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.trend_history_manager import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_PIPELINE_LOG_PATH,
    _load_history,
    _rejection_reason,
)
from src.config import AppConfig, load_config

DEFAULT_QUEUE_PATH = WORKSPACE_ROOT / "data" / "trends_queue.json"
DEFAULT_THUMBS_DIR = WORKSPACE_ROOT / "data" / "card_thumbs"

DEFAULT_CATEGORIES: list[dict[str, str]] = [
    {"key": "movies", "label_ar": "أفلام"},
    {"key": "series", "label_ar": "مسلسلات"},
    {"key": "games", "label_ar": "ألعاب"},
    {"key": "ai", "label_ar": "ذكاء اصطناعي"},
]

CATEGORY_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "movies": ((48, 16, 28), (255, 118, 92)),
    "series": ((24, 34, 48), (112, 224, 198)),
    "games": ((18, 28, 56), (94, 168, 255)),
    "ai": ((14, 34, 48), (72, 232, 220)),
}

DISTINCTIVE_ENTITIES = {
    "openai", "chatgpt", "claude", "anthropic", "deepmind", "meta", "avengers",
    "doomsday", "infinity", "imax", "dune", "gta", "warcraft", "blizzard",
    "zelda", "nintendo", "switch", "godzilla", "spider", "bond", "pattinson",
    "altruists", "blade", "runner", "sauron", "rings", "yellowjackets",
    "gaussian", "diffusion", "lean", "npc",
}

ARABIC_SUMMARY_HINTS: dict[str, str] = {
    "trend_movies_pilot_01": "سر إطلاق صيغة Infinity Vision من ديزني لـ Avengers: Doomsday بعد استحواذ Dune 3 على IMAX.",
    "trend_movies_02": "3 حقائق موثقة غيّرها فيلم Netflix الدرامي The Altruists عن القصة الحقيقية.",
    "trend_prev_04": "ماذا يعني اختيار روبرت باتينسون لدور جيمس بوند الجديد في نبرة السلسلة القادمة.",
    "trend_series_01": "تفصيل خفي في إعلان Blade Runner 2099 يكشف خيطاً أساسياً حول شخصية ميشيل يوه.",
    "trend_series_02": "ما الذي تغيّره القفزة الزمنية المؤكدة في الموسم الثالث من The Rings of Power.",
    "trend_series_03": "سر اعتماد مسلسلات الإثارة الحديثة على 3 خطوط زمنية متقاطعة منذ الحلقة الأولى.",
    "trend_games_01": "إعادة إطلاق World of Warcraft بخط زمني بديل يتفرع من نسخة 2004 دون مسح التقدم.",
    "trend_games_02": "كيف غيّر إضافة زر القفز والكاميرا الحرة تصميم ألغاز Zelda: Ocarina of Time على Switch 2.",
    "trend_games_03": "كيف تحصد الألعاب المستقلة لاعبين أوفياء بالتزامن مع إطلاق GTA VI الضخم.",
    "trend_ai_01": "الفرق بين عروض نماذج العالم بالذكاء الاصطناعي وبناء محركات الألعاب الفعلية.",
    "trend_ai_02": "كيف تستخدم استوديوهات السينما أدوات الذكاء الاصطناعي في تصور المشاهد قبل التصوير.",
    "trend_ai_03": "الفرق بين الإثارة الإعلامية والتحقق الرياضي الرسمي عبر Lean 4 في أبحاث الذكاء الاصطناعي.",
}

EVERGREEN_FALLBACKS: dict[str, list[dict[str, Any]]] = {
    "movies": [
        {
            "id": "eg_movies_1",
            "title": "Why Premium Cinema Formats Keep Breaking Opening Records",
            "short_summary_ar": "كيف تتنافس صيغ العرض السينمائية العملاقة على أفلام نهاية العام الكبرى.",
            "sourcing_keywords": ["IMAX cinema theater audience", "film projector beam theater"],
        },
        {
            "id": "eg_movies_2",
            "title": "Inside Practical Effects Replacing Green-Screen CGI in Cinema",
            "short_summary_ar": "عودة التصوير الحقيقي والمجسمات المصغرة في كبرى أفلام هوليوود الحديثة.",
            "sourcing_keywords": ["movie stunt practical effects behind the scenes", "film set lighting"],
        },
        {
            "id": "eg_movies_3",
            "title": "Hidden Clues Trailers Use Without Spoiling Act Three",
            "short_summary_ar": "كيف تزرع الاستوديوهات تفاصيل دقيقة في الإعلانات التشويقية دون حرق النهاية.",
            "sourcing_keywords": ["cinema trailer editing suite", "film director monitor"],
        },
    ],
    "series": [
        {
            "id": "eg_series_1",
            "title": "Why Non-Linear Thriller Series Hook Viewers Past Episode One",
            "short_summary_ar": "سر استخدام الخطوط الزمنية المتعددة في مسلسلات البث لرفع نسبة إكمال المواسم.",
            "sourcing_keywords": ["cinematic crime thriller scene rain", "TV production set camera"],
        },
        {
            "id": "eg_series_2",
            "title": "The Billion-Dollar Streaming War Between Fantasy TV Epics",
            "short_summary_ar": "كيف تتنافس منصات البث بميزانيات سينمائية غير مسبوقة في مواسم الفانتازيا.",
            "sourcing_keywords": ["medieval castle aerial drone shot", "TV studio filming crane"],
        },
        {
            "id": "eg_series_3",
            "title": "Why 8-Episode Prestige Shows Replaced 22-Episode Seasons",
            "short_summary_ar": "التحول الدرامي والاقتصادي في صناعة المسلسلات نحو المواسم القصيرة المكثفة.",
            "sourcing_keywords": ["streaming television remote night living room", "film director monitor"],
        },
    ],
    "games": [
        {
            "id": "eg_games_1",
            "title": "How Indie Studios Win Players Alongside $200M AAA Launches",
            "short_summary_ar": "كيف تنجح الاستوديوهات المستقلة بأفكار مركزة أمام أضخم إصدارات الألعاب.",
            "sourcing_keywords": ["indie game development showcase", "pc gaming gameplay reel"],
        },
        {
            "id": "eg_games_2",
            "title": "Why Co-Op Extraction Games Took Over Multiplayer Charts",
            "short_summary_ar": "السر التصميمي وراء تصدّر ألعاب التعاون والاستخراج لقوائم اللعب والبث.",
            "sourcing_keywords": ["tactical multiplayer game 4K gameplay", "esports tournament crowd"],
        },
        {
            "id": "eg_games_3",
            "title": "How One Balance Patch Can Reshape an Entire Competitive Meta",
            "short_summary_ar": "تحليل كيف يغيّر تعديل رقمي صغير في التحديثات سلوك ملايين اللاعبين.",
            "sourcing_keywords": ["game engine 3D scene editor", "gaming tournament arena"],
        },
    ],
    "ai": [
        {
            "id": "eg_ai_1",
            "title": "AI NPCs in Studio Demos vs Real Shipped Games",
            "short_summary_ar": "الفارق الحقيقي بين عروض الشخصيات الذكية في المعارض والألعاب الصادرة فعلياً.",
            "sourcing_keywords": ["3D game character animation demo", "AI simulation visualization"],
        },
        {
            "id": "eg_ai_2",
            "title": "How Gaussian Splatting Turns Video Into Navigable 3D Scenes",
            "short_summary_ar": "شرح تقنية إعادة البناء ثلاثي الأبعاد وتطبيقاتها في المؤثرات البصرية والألعاب.",
            "sourcing_keywords": ["3D point cloud gaussian splatting demo", "VFX studio monitor"],
        },
        {
            "id": "eg_ai_3",
            "title": "How Directors Use AI Pre-Visualization Before Physical Filming",
            "short_summary_ar": "الفرق بين التصور المسبق بالذكاء الاصطناعي والمؤثرات النهائية في صناعة السينما.",
            "sourcing_keywords": ["film previsualization 3D storyboard", "VFX artist workstation"],
        },
    ],
}


def _short_ar_summary(item: dict[str, Any]) -> str:
    if item.get("short_summary_ar"):
        return str(item["short_summary_ar"]).strip()
    tid = str(item.get("id", "")).strip()
    if tid in ARABIC_SUMMARY_HINTS:
        return ARABIC_SUMMARY_HINTS[tid]
    summary_en = str(item.get("summary") or "").strip()
    cat_ar = str(item.get("category_ar") or "الترند").strip()
    if summary_en:
        clean = summary_en[:120].rstrip(" ,.;:")
        return f"ترند صاعد في فئة {cat_ar}: {clean}."
    return f"موضوع شائع حديثاً في فئة {cat_ar} ومناسب لفيديو تحليلي مدته 65–85 ثانية."


def _extract_entities(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-zA-Z0-9]{2,}", text.lower()))
    return tokens & DISTINCTIVE_ENTITIES


def _fetch_ytdlp_thumb(query: str) -> str:
    try:
        import yt_dlp

        ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        entries = (info or {}).get("entries") or []
        if entries:
            vid_id = str(entries[0].get("id") or "").strip()
            if vid_id:
                return f"https://i.ytimg.com/vi/{vid_id}/hqdefault.jpg"
    except Exception:
        pass
    return ""


def _download_image_if_valid(url: str, dest: Path) -> bool:
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
        if len(data) < 5000:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        with Image.open(dest) as im:
            im.verify()
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def _load_portable_font(size: int, bold: bool = True) -> Any:
    candidates = (
        [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "arialbd.ttf",
            "arial.ttf",
        ]
        if bold
        else [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "arial.ttf",
        ]
    )
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _generate_local_card_thumb(item: dict[str, Any], dest: Path) -> Path:
    """Create a crisp 1280x720 fallback poster card locally (0 AI tokens)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cat_key = str(item.get("category_key", "technology")).lower()
    bg_rgb, accent_rgb = CATEGORY_COLORS.get(cat_key, ((24, 30, 44), (96, 200, 255)))
    w, h = 1280, 720
    img = Image.new("RGB", (w, h), bg_rgb)
    draw = ImageDraw.Draw(img)

    for y in range(0, h, 4):
        shade = int(20 * (y / h))
        r = max(0, bg_rgb[0] - shade)
        g = max(0, bg_rgb[1] - shade)
        b = max(0, bg_rgb[2] - shade)
        draw.line([(0, y), (w, y)], fill=(r, g, b), width=4)

    draw.rectangle([48, 48, w - 48, h - 48], outline=accent_rgb, width=4)
    draw.rectangle([72, 76, 390, 128], fill=accent_rgb)

    font_badge = _load_portable_font(28, bold=True)
    font_title = _load_portable_font(52, bold=True)
    font_sub = _load_portable_font(28, bold=False)

    badge_text = f"VIRAL STUDIO | {cat_key.upper()}"
    draw.text((88, 86), badge_text, fill=(15, 18, 24), font=font_badge)

    wrapped = textwrap.wrap(str(item.get("title", "Trending Topic")), width=34)[:3]
    y_cursor = 210
    for line in wrapped:
        draw.text((78, y_cursor), line, fill=(255, 255, 255), font=font_title)
        y_cursor += 70

    draw.text((78, h - 115), "TikTok 9:16 Ready • 65-85s Analysis", fill=accent_rgb, font=font_sub)
    img.save(dest, format="JPEG", quality=92)
    return dest


def resolve_trend_card_image(
    item: dict[str, Any],
    thumbs_dir: Path = DEFAULT_THUMBS_DIR,
    fetch_remote: bool = False,
) -> str:
    """Return an existing local image path (fetching YouTube thumb if fetch_remote=True, else Pillow fallback)."""
    existing = str(item.get("image_url_or_path") or "").strip()
    if existing and Path(existing).exists():
        return existing

    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", str(item.get("id") or "trend"))[:48]
    local_dest = thumbs_dir / f"{slug}.jpg"
    if local_dest.exists() and local_dest.stat().st_size > 4000:
        return str(local_dest)

    if fetch_remote:
        keywords = list(item.get("sourcing_keywords") or [item.get("title", "")])
        thumb_url = _fetch_ytdlp_thumb(str(keywords[0])) if keywords else ""
        if thumb_url and _download_image_if_valid(thumb_url, local_dest):
            return str(local_dest)

    return str(_generate_local_card_thumb(item, local_dest))


def _configured_categories(config: AppConfig) -> list[dict[str, str]]:
    raw = config.telegram.get("categories")
    if isinstance(raw, list) and len(raw) >= 1:
        out: list[dict[str, str]] = []
        for c in raw:
            if isinstance(c, dict) and c.get("key"):
                out.append({"key": str(c["key"]).lower(), "label_ar": str(c.get("label_ar") or c["key"])})
        if out:
            return out
    return list(DEFAULT_CATEGORIES)


def count_cross_category_overlaps(pools: dict[str, dict[str, Any]]) -> int:
    """Count any duplicate title or cross-category entity overlap across the 7 category pools."""
    overlaps = 0
    seen_titles: set[str] = set()
    cat_entities: dict[str, set[str]] = {}
    for cat_key, cat_data in pools.items():
        ents: set[str] = set()
        for item in cat_data.get("items", []):
            t_norm = str(item.get("title", "")).strip().lower()
            if t_norm in seen_titles:
                overlaps += 1
            seen_titles.add(t_norm)
            ents |= _extract_entities(f"{item.get('title', '')} {item.get('short_summary_ar', '')}")
        for prev_cat, prev_ents in cat_entities.items():
            if ents & prev_ents:
                overlaps += len(ents & prev_ents)
        cat_entities[cat_key] = ents
    return overlaps


def build_category_trend_pools(
    config: AppConfig | None = None,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH,
    pipeline_log_path: Path = DEFAULT_PIPELINE_LOG_PATH,
    thumbs_dir: Path = DEFAULT_THUMBS_DIR,
    pool_size: int | None = None,
    fetch_remote_thumbs: bool = False,
    refresh_live: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build ranked 3-5 unused trend pools across the 4 connected pillars with strict cross-category dedup."""
    cfg = config or load_config(WORKSPACE_ROOT / "config.yaml")
    if refresh_live:
        try:
            from scripts.fetch_live_trends import fetch_live_trends

            fetch_live_trends(
                history_path=history_path,
                pipeline_log_path=pipeline_log_path,
                queue_path=queue_path,
                per_category=max(3, min(5, int(pool_size or cfg.telegram.get("pool_size_per_category", 5)))),
            )
        except Exception:
            pass
    target_per_cat = max(3, min(5, int(pool_size or cfg.telegram.get("pool_size_per_category", 5))))
    categories = _configured_categories(cfg)

    history = _load_history(history_path, pipeline_log_path)
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

    queue_items: list[dict[str, Any]] = []
    if queue_path.exists():
        q_payload = json.loads(queue_path.read_text(encoding="utf-8"))
        queue_items = [t for t in q_payload.get("trends", []) if isinstance(t, dict)]

    # Single consolidated pass tracking global titles + cross-category entities
    global_selected_titles: list[str] = []
    cross_category_entities: set[str] = set()

    pools: dict[str, dict[str, Any]] = {}
    for cat in categories:
        cat_key = cat["key"]
        cat_ar = cat["label_ar"]
        raw_candidates: list[dict[str, Any]] = []

        # Strict category separation: exact match only (no mixing IT into technology or AI)
        for q in queue_items:
            q_cat = str(q.get("category", "")).strip().lower()
            q_status = str(q.get("status", "PENDING")).strip().upper()
            if q_cat == cat_key and q_status == "PENDING":
                raw_candidates.append(
                    {
                        "id": str(q.get("id") or hashlib.md5(q["topic"].encode("utf-8")).hexdigest()[:10]),
                        "category_key": cat_key,
                        "category_ar": cat_ar,
                        "title": str(q["topic"]).strip(),
                        "summary": str(q.get("summary", "")).strip(),
                        "priority": int(q.get("priority") or 80),
                        "sourcing_keywords": [str(q["topic"]).split("—")[0].strip() + " trailer showcase"],
                    }
                )

        for fb in EVERGREEN_FALLBACKS.get(cat_key, []):
            raw_candidates.append(
                {
                    "id": fb["id"],
                    "category_key": cat_key,
                    "category_ar": cat_ar,
                    "title": fb["title"],
                    "short_summary_ar": fb["short_summary_ar"],
                    "priority": 72,
                    "sourcing_keywords": list(fb.get("sourcing_keywords", [])),
                }
            )

        selected_items: list[dict[str, Any]] = []
        cat_entities: set[str] = set()
        sorted_cands = sorted(raw_candidates, key=lambda x: x.get("priority", 0), reverse=True)
        for cand in sorted_cands:
            if len(selected_items) >= target_per_cat:
                break
            reason = _rejection_reason(cand, past_topics + global_selected_titles, sim_threshold=0.35)
            if reason:
                continue

            item_summary = _short_ar_summary(cand)
            cand_entities = _extract_entities(f"{cand['title']} {cand.get('summary', '')} {item_summary}")
            # Reject if ANY distinctive entity was already claimed by a different category card
            if cand_entities & cross_category_entities:
                continue

            global_selected_titles.append(cand["title"])
            cat_entities |= cand_entities
            item = {
                "id": cand["id"],
                "category_key": cat_key,
                "category_ar": cat_ar,
                "title": cand["title"],
                "short_summary_ar": item_summary,
                "sourcing_keywords": list(cand.get("sourcing_keywords") or [cand["title"]]),
                "image_url_or_path": "",
            }
            item["image_url_or_path"] = resolve_trend_card_image(
                item, thumbs_dir=thumbs_dir, fetch_remote=fetch_remote_thumbs
            )
            selected_items.append(item)

        cross_category_entities |= cat_entities
        pools[cat_key] = {
            "category_key": cat_key,
            "category_ar": cat_ar,
            "items": selected_items,
        }
    return pools


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 3-5 unused trend cards per 4 connected pillars for Telegram.")
    parser.add_argument("--dry-run", action="store_true", help="Build pools offline with local fallback thumbs")
    parser.add_argument("--fetch-thumbs", action="store_true", help="Attempt free yt-dlp thumbnail resolution")
    parser.add_argument("--refresh-live", action="store_true", help="Re-scan data/trend_sources.json feeds first")
    parser.add_argument("--config", default=str(WORKSPACE_ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    categories = _configured_categories(cfg)
    pools = build_category_trend_pools(
        config=cfg,
        fetch_remote_thumbs=bool(args.fetch_thumbs and not args.dry_run),
        refresh_live=bool(args.refresh_live and not args.dry_run),
    )
    overlaps = count_cross_category_overlaps(pools)
    summary = {
        "total_categories": len(pools),
        "cross_category_overlap_count": overlaps,
        "categories": {
            k: {"category_ar": v["category_ar"], "count": len(v["items"]), "titles": [i["title"] for i in v["items"]]}
            for k, v in pools.items()
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if (len(pools) == len(categories) and overlaps == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
