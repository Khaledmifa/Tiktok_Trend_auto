from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.models import ResearchReport, ScriptPlan, ScriptScene, ViralOpportunity
from src.providers.llm import build_llm_provider
from src.state.memory_store import MemoryStore
from src.utils.language_guard import ensure_english_text, is_english_text, translate_to_english

logger = logging.getLogger(__name__)

_is_english_text = is_english_text
_ENTITY_GENERIC_WORDS: set[str] = {
    "the", "and", "for", "with", "from", "prime", "video", "netflix", "hbo",
    "apple", "official", "trailer", "series", "movie", "film", "game", "hidden",
    "clue", "about", "what", "why", "how", "new", "2026", "2025",
}
_DANGLING_END_WORDS: set[str] = {
    "of", "the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "and", "or",
}


def _extract_core_entity_tokens(topic: str) -> set[str]:
    """Extract distinctive core entity tokens (e.g. {'blade', 'runner', '2099'}) from a topic title."""
    main_part = re.split(r"\s*[—–|:]\s*", (topic or "").strip(), maxsplit=1)[0].strip()
    main_part = re.sub(r"\([^)]*\)", " ", main_part).strip()
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", main_part.lower())
    filtered = {t for t in tokens if t not in _ENTITY_GENERIC_WORDS}
    return filtered or set(tokens)


def _is_topic_relevant_fact(snippet: str, topic: str) -> bool:
    """Ensure fetched snippet actually discusses the target entity rather than a partial namesake."""
    core_tokens = _extract_core_entity_tokens(topic)
    if not core_tokens:
        return True
    lower = (snippet or "").lower()
    matched = sum(1 for tok in core_tokens if re.search(rf"\b{re.escape(tok)}\b", lower))
    if len(core_tokens) >= 2:
        return matched >= 2
    return matched >= 1


def _clean_complete_sentence(snippet: str, max_words: int = 22) -> str | None:
    """Return a clean, self-contained sentence without date prefixes, URLs, ellipsis, or truncated endings."""
    clean = re.sub(r"https?://\S+", "", snippet or "")
    clean = re.sub(r"\s+", " ", clean).lstrip(" .-|:\u00b7").rstrip(" -|:\u00b7")
    clean = re.sub(
        r"^(?:[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})\s*[·•\-–|:]\s*",
        "",
        clean,
    ).strip()
    clean = re.sub(r"(?:\.\.\.|…)+$", "", clean).strip()
    if not clean:
        return None
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", clean) if p.strip()]
    for part in parts:
        if not part.endswith((".", "!", "?")):
            continue
        words = part.split()
        if not (7 <= len(words) <= max_words):
            continue
        last_word = re.sub(r"[^a-zA-Z]", "", words[-1]).lower()
        if last_word in _DANGLING_END_WORDS:
            continue
        return part
    return None


def _ddgs_text_with_region(ddgs: Any, query: str, region: str, max_results: int) -> list[dict[str, Any]]:
    try:
        return list(ddgs.text(query, region=region, max_results=max_results) or [])
    except TypeError:
        return list(ddgs.text(query, max_results=max_results) or [])


def _gather_live_web_facts(
    topic: str,
    keywords: list[str] | None = None,
    max_facts: int = 5,
    language: str = "en",
) -> list[str]:
    """Fetch clean, topic-relevant factual snippets via free DuckDuckGo search ($0)."""
    clean_topic = re.sub(r"[—–|]+", " ", topic or "").strip()
    if not clean_topic:
        return []
    short_entity = re.sub(r"\([^)]*\)", " ", _clean_short_topic(topic)).strip()
    short_entity = re.sub(r"\s+", " ", short_entity) or clean_topic
    region_map = {"en": "us-en", "de": "de-de", "fr": "fr-fr"}
    region = region_map.get((language or "en").lower(), "wt-wt")

    facts: list[str] = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        kw_hint = " ".join(str(k) for k in (keywords or [])[:2]).strip()
        queries = [f'"{short_entity}"']
        if kw_hint and kw_hint.lower() not in short_entity.lower():
            queries.append(f'"{short_entity}" {kw_hint}')
        else:
            queries.append(f"{short_entity} release plot cast details")
        with DDGS() as ddgs:
            for q in queries[:2]:
                for r in _ddgs_text_with_region(ddgs, q, region=region, max_results=5):
                    body = re.sub(r"\s+", " ", str(r.get("body") or "")).strip()
                    title = re.sub(r"\s+", " ", str(r.get("title") or "")).strip()
                    for raw_candidate in (body, title):
                        if not raw_candidate:
                            continue
                        candidate_text = raw_candidate
                        if (language or "en").lower() == "en" and not is_english_text(candidate_text):
                            translated = translate_to_english(candidate_text)
                            if not translated:
                                continue
                            candidate_text = translated
                        sent = _clean_complete_sentence(candidate_text)
                        if not sent:
                            continue
                        if not _is_topic_relevant_fact(sent, topic):
                            continue
                        if (language or "en").lower() == "en" and not is_english_text(sent):
                            continue
                        if sent not in facts:
                            facts.append(sent)
                        break
                    if len(facts) >= max_facts:
                        return facts
    except Exception as exc:
        logger.debug("DDGS fact lookup skipped for '%s': %s", clean_topic, exc)
    return facts


def _clean_short_topic(topic: str) -> str:
    main_part = re.split(r"\s*[—–|:]\s*", (topic or "").strip(), maxsplit=1)[0].strip()
    words = main_part.split()
    return " ".join(words[:7]) if len(words) > 7 else (main_part or topic or "This Story")


def _truncate_at_word_boundary(text: str, max_chars: int = 140) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean) <= max_chars:
        return clean.rstrip(".,;:")
    cut = clean[:max_chars].rsplit(" ", 1)[0].strip()
    return (cut or clean[:max_chars]).rstrip(".,;:")


def _infer_category_label(topic: str, keywords: list[str] | None = None, niche: str = "") -> str:
    topic_blob = " ".join([topic or "", " ".join(str(k) for k in (keywords or []))]).lower()

    def _match_blob(blob: str) -> str | None:
        if any(
            w in blob
            for w in ("series", "season", "episode", "netflix", "hbo", "prime video", "tv show", "apple tv", "limited series", "showrunner")
        ) and not any(g in blob for g in ("game series", "video game", "gameplay")):
            return "SERIES EXPLAINED"
        if any(
            w in blob
            for w in ("game", "gaming", "gameplay", "fortnite", "gta", "playstation", "xbox", "nintendo", "switch", "zelda", "warcraft", "steam", "esports")
        ):
            return "GAMING EXPLAINED"
        if re.search(r"\b(ai|llm|gpt|claude|anthropic|deepmind|diffusion|openai|sora|runway|midjourney|world model|neural|gaussian)\b", blob):
            return "AI EXPLAINED"
        if any(w in blob for w in ("movie", "film", "cinema", "trailer", "box office", "director")):
            return "MOVIES EXPLAINED"
        return None

    matched = _match_blob(topic_blob)
    if matched:
        return matched
    # Only fall back to niche if it's a single focused pillar (never multi-pillar comma list)
    niche_clean = (niche or "").strip().lower()
    if niche_clean and "," not in niche_clean:
        niche_matched = _match_blob(niche_clean)
        if niche_matched:
            return niche_matched
    return "MOVIES EXPLAINED"


def _derive_scene_editorial_fields(
    scene_idx: int,
    total: int,
    vo_text: str,
    topic: str,
    raw_scene: dict[str, Any],
    default_context: str = "OFFICIAL FOOTAGE",
) -> dict[str, str]:
    """Populate Editorial HUD fields cleanly if not explicitly provided in raw JSON."""
    short_topic = _clean_short_topic(topic)
    kicker = str(raw_scene.get("kicker") or "").strip()
    h1 = str(raw_scene.get("headline_line1") or "").strip()
    h2 = str(raw_scene.get("headline_line2") or "").strip()
    fact_card = str(raw_scene.get("fact_card") or "").strip()
    layout_mode = str(raw_scene.get("layout_mode") or "auto").strip().lower()
    footage_ctx = str(raw_scene.get("footage_context") or default_context).strip()

    if layout_mode not in ("wide_panel", "fullscreen_portrait"):
        layout_mode = "wide_panel" if scene_idx % 2 == 1 else "fullscreen_portrait"

    if not kicker:
        if scene_idx == 1:
            kicker = "KEY CONTEXT"
        elif scene_idx == total:
            kicker = "YOUR TAKE"
        elif scene_idx >= max(2, total // 2) and scene_idx <= max(3, total // 2 + 1):
            kicker = "THE MECHANISM"
        else:
            kicker = f"POINT {scene_idx:02d}"

    words = [w.strip(".,!?;:\"'()[]") for w in (vo_text or "").split() if w.strip(".,!?;:\"'()[]")]
    if not h1:
        h1_words = words[:3] if len(words) >= 3 else short_topic.split()[:3]
        h1 = " ".join(h1_words).upper() or short_topic.upper()
    if not h2:
        tail = words[3:6] if len(words) >= 6 else words[-3:]
        h2 = " ".join(tail).upper() or "THE REAL STORY"
        if h2 == h1:
            h2 = "WHAT MATTERS NOW"

    if not fact_card:
        clean_sent = re.sub(r"\s+", " ", (vo_text or "")).strip()
        if len(clean_sent) <= 58:
            fact_card = clean_sent
        else:
            snippet = " ".join(clean_sent.split()[:9]).rstrip(",;:")
            fact_card = f"{snippet}..." if len(snippet) > 54 else snippet

    return {
        "kicker": kicker[:32],
        "headline_line1": h1[:28],
        "headline_line2": h2[:28],
        "fact_card": fact_card[:68],
        "layout_mode": layout_mode,
        "footage_context": footage_ctx[:36],
    }


class ScriptArchitectAgent:
    """AI Model 2: Crafts 100% original 65-85s topic-grounded TikTok scripts & visual prompts."""

    def __init__(
        self, config: AppConfig, memory: MemoryStore, skip_external_llm: bool = False
    ) -> None:
        self.config = config
        self.memory = memory
        if skip_external_llm:
            self.llm = None
        else:
            model_name = config.raw.get("models", {}).get("script_model")
            self.llm = build_llm_provider(config, model_override=model_name)

    def _select_best_opportunity(self, report: ResearchReport) -> ViralOpportunity:
        threshold = float(self.config.qa_and_publish.get("max_similarity_threshold", 0.55))
        ranked = sorted(report.opportunities, key=lambda o: o.score, reverse=True)
        for opp in ranked:
            sim = self.memory.max_similarity_against_history(
                f"{opp.topic} {opp.unique_angle}"
            )
            if sim < threshold:
                return opp
        return ranked[0]

    def _generate_with_llm(
        self,
        opp: ViralOpportunity,
        language: str,
        min_sec: int,
        max_sec: int,
        facts: list[str] | None = None,
        revision_feedback: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.llm:
            return None

        pipe_cfg = self.config.pipeline
        min_scenes = int(pipe_cfg.get("min_scenes", 10))
        max_scenes = int(pipe_cfg.get("max_scenes", 12))
        max_prompt_words = int(pipe_cfg.get("max_visual_prompt_words", 15))
        words_en = pipe_cfg.get("target_words_en", [155, 185])
        words_ar = pipe_cfg.get("target_words_ar", [135, 165])
        target_words = (
            f"{words_ar[0]} to {words_ar[1]} words"
            if language == "ar"
            else f"{words_en[0]} to {words_en[1]} words"
        )
        system_prompt = (
            "You are AI Model 2 (Viral Studio Pro Editorial Scriptwriter & Director). "
            f"Write a 100% original, fact-grounded TikTok editorial breakdown spoken in {min_sec} to {max_sec} seconds ({target_words}) "
            f"strictly about '{opp.topic}'.\n"
            "EDITORIAL & SEAMLESS LOOP RULES:\n"
            "1. Channel covers 4 connected pillars ONLY: Movies, Series, Games, AI Trends (NO hardware reviews, unboxings, or shopping recommendations).\n"
            "2. Write the opening hook and final connecting phrase together so the ending links grammatically and logically back into the opening sentence on replay. Adapt the connector to the topic (never force 'And that's why...').\n"
            "3. Deliver the promised explanation completely BEFORE the ending, and place any follow/save invitation naturally BEFORE the final connecting phrase.\n"
            "4. For AI stories: explicitly distinguish RELEASED FEATURE vs ANNOUNCEMENT / DEMO vs RUMOR / SPECULATION, and label synthetic visuals as SYNTHETIC / AI-GENERATED.\n"
            f"5. Break into {min_scenes} to {max_scenes} cohesive editorial sections (~6.0 to 7.5 seconds each).\n"
            f"6. Every scene MUST include `visual_ai_prompt` (max {max_prompt_words} words), "
            "`kicker` (2-3 uppercase words), `headline_line1` (2-4 words), `headline_line2` (2-4 words), "
            "`fact_card` (only on key concrete-metric scenes; empty string otherwise to avoid clutter), `footage_context`, and `layout_mode` ('wide_panel' or 'fullscreen_portrait').\n"
            "7. Specify `video_motion_prompt` per scene from: zoom_in, zoom_out, pan_left, pan_right, whip_zoom."
        )
        payload: dict[str, Any] = {
            "opportunity": {
                "topic": opp.topic,
                "unique_angle": opp.unique_angle,
                "hook_pattern": opp.hook_pattern,
                "narrative_structure": opp.narrative_structure,
                "keywords": opp.keywords,
                "verified_web_facts": facts or [],
            },
            "language": language,
            "constraints": {
                "min_duration_sec": min_sec,
                "max_duration_sec": max_sec,
                "scene_count": f"{min_scenes} to {max_scenes} fast-paced scenes",
                "target_words": target_words,
                "max_visual_prompt_words": max_prompt_words,
            },
            "required_json_schema": {
                "opening_hook_3s": "string",
                "closing_cta": "string (complete final scene text ending on the seamless loop connector)",
                "loop_bridge_phrase": "exact final connecting clause that flows grammatically into opening_hook_3s",
                "loop_joined_preview": "loop_bridge_phrase + opening_hook_3s joined cleanly",
                "loop_cut_notes": "visual match & audio transition note for final-to-first replay cut",
                "scenes": [
                    {
                        "scene_number": 1,
                        "estimated_duration_sec": 6.8,
                        "voiceover_text": "string",
                        "visual_query": "string",
                        "visual_ai_prompt": f"Concise <= {max_prompt_words} words dark cinematic prompt about topic",
                        "video_motion_prompt": "zoom_in | zoom_out | pan_left | pan_right | whip_zoom",
                        "on_screen_focus": "string",
                        "kicker": "TWO THINGS",
                        "headline_line1": "CONFIRMED FACTS.",
                        "headline_line2": "NOT HYPE.",
                        "fact_card": "Key verified date or metric (<= 55 chars)",
                        "layout_mode": "wide_panel | fullscreen_portrait",
                    }
                ],
            },
        }
        if revision_feedback:
            payload["reviewer_critique_to_fix"] = revision_feedback

        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            return self.llm.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:
            logger.warning("Local Ollama script call skipped (%s); using topic script builder.", exc)
            return None

    def _build_topic_script(
        self,
        opp: ViralOpportunity,
        language: str,
        facts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate a 10-scene (~68-75s, ~160-175 word) script strictly grounded in `opp.topic`."""
        topic_full = (opp.topic or "Trending Topic").strip()
        short_topic = _clean_short_topic(topic_full)
        live_facts = (
            facts
            if facts is not None
            else _gather_live_web_facts(topic_full, opp.keywords, language=language)
        )
        angle_clean = re.sub(r"\s+", " ", (opp.unique_angle or "")).strip()

        def fact_or(idx: int, default_line: str) -> str:
            if idx < len(live_facts) and len(live_facts[idx].split()) >= 7:
                return live_facts[idx]
            return default_line

        if language == "ar":
            loop_bridge = "لأن من البداية،"
            hook_ar = f"هذا التفصيل حول {short_topic} يغيّر الطريقة التي تنظر بها إلى القصة كاملة."
            cta_ar = f"احفظ الحلقة للرجوع إليها قبل التحديث القادم، وراجع ما بدأ القصة {loop_bridge}"
            beats = [
                (6.8, hook_ar, f"{short_topic} opening", "السياق الأساسي", short_topic[:24], "الفرق الحقيقي", f"تفاصيل {short_topic[:28]} الموثقة"),
                (6.8, f"ما أُعلن رسمياً عن {short_topic} يختلف عن الانطباع السريع المنتشر على المنصات.", f"{short_topic} official showcase", "الحقائق أولاً", "الإعلان الرسمي", "ما تغيّر فعلاً", "فصل الخبر المؤكد عن التوقعات"),
                (6.8, f"النقطة الأولى تستحق التدقيق لأن التوقيت والأرقام يحددان حجم التأثير الفعلي.", f"{short_topic} core story", "القراءة الدقيقة", "الأرقام والتوقيت", "قبل الحكم", "الجدول الزمني يوضح الصورة"),
                (6.8, f"المتابعون ركزوا على العنوان العريض بينما التفاصيل التنفيذية تكشف اتجاهاً أوضح.", f"{short_topic} hidden details", "ما وراء العنوان", "تفاصيل التنفيذ", "ليست مجرد دعاية", "تغييرات ملموسة في التجربة"),
                (6.8, f"رأيي التحليلي أن الرهان الحقيقي في {short_topic} يكمن في البناء التدريجي لا الضجيج اللحظي.", f"{short_topic} turning point", "رؤية تحليلية", "التصميم التدريجي", "سر الاهتمام", "توازن بين المضمون والتوقيت"),
                (6.8, f"عند فصل الحقائق عن المبالغة نرى بوضوح ما هو مؤكد وما زال قيد الانتظار.", f"{short_topic} key moment", "حقائق لا ضجيج", "المؤكد رسمياً", "مقابل التوقعات", "لا استنتاجات غير موثقة"),
                (6.8, f"المقارنة مع الإصدارات السابقة تُظهر لماذا تراقب الشركات والجمهور هذه الخطوة عن قرب.", f"{short_topic} comparison", "المقارنة", "مقياس الصناعة", "خطوة مفصلية", "أثر مباشر على المنافسين"),
                (6.8, f"المشاهد الرسمية المعروضة توضح الوضع الحالي ولا تمثل بالضرورة المرحلة النهائية.", f"{short_topic} behind the scenes", "سياق اللقطات", "الموسم الحالي", "وليس النهاية", "لقطات رسمية للتوضيح"),
                (7.0, f"الأيام القادمة ستختبر ما إذا كان {short_topic} سيحافظ على زخمه الفعلي أمام الجمهور.", f"{short_topic} future", "الخطوة القادمة", "اختبار الجمهور", "خلال أيام", "ترقب للتحديثات الرسمية"),
                (7.0, cta_ar, f"{short_topic} debate", "حلقة متصلة", "النهاية تقود للبداية", "تابع التفاصيل", "التكرار يكشف الترابط"),
            ]
        else:
            f0 = fact_or(0, f"Official details around {short_topic} show why timing and execution matter more than surface hype.")
            f1 = fact_or(1, f"Behind the announcements, {short_topic} introduces concrete changes that reshape how users engage.")
            f2 = fact_or(2, f"Keep confirmed facts separate from speculation when evaluating what {short_topic} actually delivers.")
            f3 = fact_or(3, f"Comparing this release with earlier milestones explains why industry watchers are tracking every update.")
            angle_sentence = (
                f"{_truncate_at_word_boundary(angle_clean, 140)}."
                if (
                    angle_clean
                    and "symbolic cinematic ai" not in angle_clean.lower()
                    and "deep-dive breakdown of" not in angle_clean.lower()
                    and len(angle_clean.split()) >= 6
                )
                else f"The smart design in {short_topic} is how clear stakes, pacing, and measurable payoff work together."
            )
            cat_label = _infer_category_label(topic_full, opp.keywords)
            if cat_label in ("MOVIES EXPLAINED", "SERIES EXPLAINED"):
                hook_line = f"{short_topic} changes what viewers expected—once you look past the noise."
                context_line = f"Step back from the headlines on {topic_full} to see what actually drives the audience reaction."
                footage_line = f"Promotional clips from {short_topic} preview tone and casting, not every twist in the final cut."
                takeaway_line = f"Track official release dates for {short_topic} before buying into unverified leaks or rumor threads."
                loop_bridge = "because right from frame one,"
                cta_line = f"Which piece stands out most: story, cast, or execution? Follow for the next breakdown, {loop_bridge}"
            elif cat_label == "GAMING EXPLAINED":
                hook_line = f"{short_topic} has two things players should not confuse: confirmed patch facts and community hype."
                context_line = f"Look closely at {topic_full} and you can see how the schedule and gameplay loop fit together."
                footage_line = f"Trailer footage for {short_topic} illustrates current mechanics, not unannounced future updates."
                takeaway_line = f"Save the official dates for {short_topic} and check developer notes directly before grinding."
                loop_bridge = "because from day one,"
                cta_line = f"What brings you back: the mechanics, rewards, or finale? Save this before launch week, {loop_bridge}"
            else:
                hook_line = f"{short_topic} separates released AI capability from studio demo hype—once you check the real limits."
                context_line = f"Look closer at {topic_full} and distinguish shipped features from early demonstrations and speculation."
                footage_line = f"Screen recordings and studio demos show controlled examples, not unrestricted real-world deployment."
                takeaway_line = f"Verify whether an AI tool is shipped or demo-only before assuming it transforms movies or games overnight."
                loop_bridge = "because beneath every launch demo,"
                cta_line = f"Would you trust this in production today, or wait for shipped tools? Save this breakdown, {loop_bridge}"

            beats = [
                (6.8, hook_line, f"{short_topic} cinematic highlight", "WHAT YOU MISSED", "FACTS VS NOISE.", "KNOW THE DIFFERENCE.", ""),
                (6.8, context_line, f"{short_topic} official showcase", "THE CONTEXT", "LOOK CLOSER", "AT THE EVIDENCE", ""),
                (6.8, f0, f"{short_topic} key moment", "VERIFIED", "CONFIRMED SCOPE.", "NO GUESSWORK.", f"Verified update: {short_topic[:36]}"),
                (6.8, f1, f"{short_topic} behind the scenes", "MECHANICS", "MORE THAN", "SURFACE HYPE", ""),
                (6.8, f"My take? {angle_sentence}", f"{short_topic} turning point", "EDITORIAL TAKE", "THE DESIGN", "IS THE PAYOFF", ""),
                (6.8, f2, f"{short_topic} major reveal", "FACT CHECK", "SHIPPED VS DEMO.", "HYPE SECOND.", "Separate confirmed releases from demos"),
                (6.8, f3, f"{short_topic} fan reaction", "COMPARISON", "WHY WATCHERS", "CARE NOW", ""),
                (6.8, footage_line, f"{short_topic} industry stakes", "FOOTAGE NOTE", "CURRENT RELEASE.", "NOT SPECULATION.", "Official clips / verified demos only"),
                (6.8, takeaway_line, f"{short_topic} climax scene", "KEY TAKEAWAY", "VERIFY FIRST.", "CHECK THE SOURCE.", ""),
                (6.8, cta_line, f"{short_topic} finale shot", "SEAMLESS LOOP", "WHAT STANDS OUT?", "TELL US BELOW", ""),
            ]

        scenes = [
            {
                "scene_number": idx,
                "estimated_duration_sec": dur,
                "voiceover_text": txt,
                "visual_query": vq[:60],
                "visual_ai_prompt": " ".join(f"{vq}, dramatic cinematic 9:16".split()[:15]),
                "on_screen_focus": short_topic[:40],
                "kicker": kicker,
                "headline_line1": h1,
                "headline_line2": h2,
                "fact_card": fact_card,
                "layout_mode": "wide_panel" if idx % 2 == 1 else "fullscreen_portrait",
            }
            for idx, (dur, txt, vq, kicker, h1, h2, fact_card) in enumerate(beats, start=1)
        ]
        hook_text = beats[0][1]
        return {
            "opening_hook_3s": hook_text,
            "closing_cta": beats[-1][1],
            "loop_bridge_phrase": loop_bridge,
            "loop_joined_preview": f"{loop_bridge} {hook_text}",
            "loop_cut_notes": "Match final and opening framing/palette with <= 0.08s audio tail and continuous loopable ambient bed (no fade-to-black or end card).",
            "scenes": scenes,
        }

    def _fallback_script(self, opp: ViralOpportunity, language: str) -> dict[str, Any]:
        """Topic-grounded 10-scene script builder (replaces legacy static templates)."""
        return self._build_topic_script(opp, language)

    def _build_plan_from_raw(
        self,
        run_id: str,
        topic: str,
        unique_angle: str,
        language: str,
        raw_script: dict[str, Any],
        out_dir: Path,
    ) -> ScriptPlan:
        scenes_data = raw_script.get("scenes") or []
        if not scenes_data:
            raise ValueError("Script JSON must contain a non-empty 'scenes' list.")

        wps = 2.0 if language == "ar" else 2.4
        motion_cycle = ["zoom_in", "zoom_out", "pan_left", "pan_right", "whip_zoom"]
        default_ctx = str(
            raw_script.get("footage_context_label")
            or self.config.editorial_hud.get("default_footage_context", "OFFICIAL FOOTAGE")
        ).strip()
        is_en = (language or "en").lower() == "en"
        allow_translate = bool(self.config.project.get("translate_non_english_sources", True))
        if is_en:
            topic = ensure_english_text(topic, fallback="Trending Story Breakdown", allow_translate=allow_translate)
            unique_angle = ensure_english_text(
                unique_angle,
                fallback=f"Editorial analysis of {topic}.",
                allow_translate=allow_translate,
            )

        short_topic = _clean_short_topic(topic)
        total_scenes = len(scenes_data)
        scenes: list[ScriptScene] = []
        for idx, s in enumerate(scenes_data, start=1):
            vo_text = str(s.get("voiceover_text", "")).strip()
            if is_en and vo_text and not is_english_text(vo_text):
                fallback_vo = (
                    f"Confirmed reporting on {short_topic} highlights the concrete story details behind the headlines."
                )
                vo_text = ensure_english_text(vo_text, fallback=fallback_vo, allow_translate=allow_translate)
                if vo_text == fallback_vo:
                    logger.warning(
                        "Non-English text detected in English run '%s' scene %d; replaced with English fallback.",
                        run_id,
                        idx,
                    )
            raw_dur = s.get("estimated_duration_sec")
            if raw_dur is None or float(raw_dur) <= 0:
                words_in_scene = max(len(vo_text.split()), 1)
                est_dur = round(max(words_in_scene / wps, 2.5), 2)
            else:
                est_dur = float(raw_dur)

            v_prompt = str(s.get("visual_ai_prompt", "")).strip()
            v_query = str(s.get("visual_query") or v_prompt[:40] or topic[:40]).strip()
            raw_scene_clean = dict(s)
            if is_en:
                fc_raw = str(raw_scene_clean.get("fact_card") or "").strip()
                if fc_raw:
                    raw_scene_clean["fact_card"] = ensure_english_text(
                        fc_raw, fallback=f"Verified update: {short_topic[:36]}", allow_translate=allow_translate
                    )
                for hud_key in ("kicker", "headline_line1", "headline_line2"):
                    hud_val = str(raw_scene_clean.get(hud_key) or "").strip()
                    if hud_val and not is_english_text(hud_val):
                        raw_scene_clean[hud_key] = ensure_english_text(
                            hud_val, fallback="", allow_translate=allow_translate
                        )
            editorial = _derive_scene_editorial_fields(
                scene_idx=idx,
                total=total_scenes,
                vo_text=vo_text,
                topic=topic,
                raw_scene=raw_scene_clean,
                default_context=default_ctx,
            )
            scenes.append(
                ScriptScene(
                    scene_number=int(s.get("scene_number", idx)),
                    estimated_duration_sec=est_dur,
                    voiceover_text=vo_text,
                    visual_query=v_query,
                    visual_ai_prompt=v_prompt or v_query,
                    video_motion_prompt=str(
                        s.get("video_motion_prompt") or motion_cycle[idx % len(motion_cycle)]
                    ).strip(),
                    on_screen_focus=str(s.get("on_screen_focus") or v_query.title()).strip(),
                    kicker=editorial["kicker"],
                    headline_line1=editorial["headline_line1"],
                    headline_line2=editorial["headline_line2"],
                    fact_card=editorial["fact_card"],
                    layout_mode=editorial["layout_mode"],
                    footage_context=editorial["footage_context"],
                )
            )

        full_text = " ".join(scene.voiceover_text for scene in scenes if scene.voiceover_text)
        word_count = len(full_text.split())
        est_duration = round(max(sum(s.estimated_duration_sec for s in scenes), word_count / wps), 2)

        category_label = str(
            raw_script.get("category_label")
            or _infer_category_label(topic, niche=str(self.config.project.get("niche", "")))
        ).strip()
        attribution_line = str(
            raw_script.get("attribution_line")
            or self.config.editorial_hud.get(
                "default_attribution", "Official trailers & media | Commentary & analysis"
            )
        ).strip()

        opening_hook = str(
            raw_script.get("opening_hook_3s") or (scenes[0].voiceover_text if scenes else "")
        )
        closing_cta = str(
            raw_script.get("closing_cta") or (scenes[-1].voiceover_text if scenes else "")
        )
        loop_bridge = str(raw_script.get("loop_bridge_phrase") or closing_cta).strip()
        if is_en:
            if not is_english_text(opening_hook) and scenes:
                opening_hook = scenes[0].voiceover_text
            if not is_english_text(closing_cta) and scenes:
                closing_cta = scenes[-1].voiceover_text
            if not is_english_text(loop_bridge):
                loop_bridge = ensure_english_text(
                    loop_bridge, fallback="because right from frame one,", allow_translate=allow_translate
                )
        loop_joined = str(
            raw_script.get("loop_joined_preview")
            or f"{loop_bridge.rstrip()} {opening_hook.lstrip()}".strip()
        ).strip()
        if is_en and not is_english_text(loop_joined):
            loop_joined = f"{loop_bridge.rstrip()} {opening_hook.lstrip()}".strip()
        loop_notes = str(
            raw_script.get("loop_cut_notes")
            or "Match final and opening framing/palette; avoid end cards, fades, or trailing silence (>0.08s)."
        ).strip()

        plan = ScriptPlan(
            run_id=run_id,
            topic=topic,
            unique_angle=unique_angle,
            language=language,
            opening_hook_3s=opening_hook,
            closing_cta=closing_cta,
            full_voiceover_text=full_text,
            total_estimated_duration_sec=est_duration,
            word_count=word_count,
            scenes=scenes,
            category_label=category_label,
            footage_context_label=default_ctx,
            attribution_line=attribution_line,
            fact_sources=[str(x) for x in (raw_script.get("fact_sources") or []) if str(x).strip()],
            video_sources=[str(x) for x in (raw_script.get("video_sources") or []) if str(x).strip()],
            loop_bridge_phrase=loop_bridge,
            loop_joined_preview=loop_joined,
            loop_cut_notes=loop_notes,
        )

        self.memory.update_run_script(
            run_id=run_id,
            topic=plan.topic,
            hook=plan.opening_hook_3s,
            script_text=plan.full_voiceover_text,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "script_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_markdown(plan, out_dir / "script_plan.md")
        return plan

    def load_from_json(
        self, run_id: str, script_json_path: Path, out_dir: Path
    ) -> ScriptPlan:
        """Load and normalize a pre-authored script JSON."""
        raw_script = json.loads(Path(script_json_path).read_text(encoding="utf-8"))
        if bool(self.config.project.get("enforce_english_only", True)):
            language = "en"
        else:
            language = str(raw_script.get("language") or self.config.project.get("language", "en"))
        topic = str(raw_script.get("topic") or "Untitled TikTok Topic").strip()
        unique_angle = str(raw_script.get("unique_angle") or "").strip()
        return self._build_plan_from_raw(
            run_id=run_id,
            topic=topic,
            unique_angle=unique_angle,
            language=language,
            raw_script=raw_script,
            out_dir=out_dir,
        )

    def run(
        self,
        run_id: str,
        report: ResearchReport,
        out_dir: Path,
        revision_feedback: str | None = None,
    ) -> ScriptPlan:
        if bool(self.config.project.get("enforce_english_only", True)):
            language = "en"
        else:
            language = self.config.project.get("language", "en")
        min_sec = int(self.config.video.get("min_duration_sec", 65))
        max_sec = int(self.config.video.get("max_duration_sec", 85))

        selected_opp = self._select_best_opportunity(report)
        facts = _gather_live_web_facts(selected_opp.topic, selected_opp.keywords, language=language)
        raw_script = self._generate_with_llm(
            selected_opp,
            language,
            min_sec,
            max_sec,
            facts=facts,
            revision_feedback=revision_feedback,
        )
        if not raw_script or not raw_script.get("scenes"):
            raw_script = self._build_topic_script(selected_opp, language, facts=facts)

        return self._build_plan_from_raw(
            run_id=run_id,
            topic=selected_opp.topic,
            unique_angle=selected_opp.unique_angle,
            language=language,
            raw_script=raw_script,
            out_dir=out_dir,
        )

    def _write_markdown(self, plan: ScriptPlan, path: Path) -> None:
        lines = [
            f"# Script & Hook Plan ({plan.run_id})",
            f"- **Topic:** {plan.topic}",
            f"- **Unique Angle:** {plan.unique_angle}",
            f"- **Word Count:** {plan.word_count} | **Beats:** {len(plan.scenes)}",
            f"- **Estimated Duration:** {plan.total_estimated_duration_sec:.1f}s",
            f"- **3s Opening Hook:** {plan.opening_hook_3s}",
            "",
            "| Beat | Est. Duration | Voiceover Text | Visual Query |",
            "|---:|---:|---|---|",
        ]
        for s in plan.scenes:
            vo = s.voiceover_text.replace("|", "/")
            vq = s.visual_query.replace("|", "/")
            lines.append(f"| {s.scene_number} | {s.estimated_duration_sec:.1f}s | {vo} | `{vq}` |")
        path.write_text("\n".join(lines), encoding="utf-8")
