from __future__ import annotations

import json
import logging
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.config import AppConfig
from src.models import ResearchReport, ViralOpportunity
from src.providers.llm import build_llm_provider
from src.state.memory_store import MemoryStore
from src.utils.language_guard import ensure_english_text, is_english_text

logger = logging.getLogger(__name__)


class TrendResearchAgent:
    """AI Model 1: Researches currently viral TikTok topics & narrative hooks."""

    def __init__(
        self, config: AppConfig, memory: MemoryStore, skip_external_llm: bool = False
    ) -> None:
        self.config = config
        self.memory = memory
        if skip_external_llm:
            self.llm = None
        else:
            model_name = config.raw.get("models", {}).get("research_model")
            self.llm = build_llm_provider(config, model_override=model_name)

    def _fetch_rss_signals(self, limit_per_feed: int = 4) -> list[dict[str, str]]:
        signals: list[dict[str, str]] = []
        feeds: list[str] = list(self.config.research.get("rss_feeds", []))
        sources_rel = str(self.config.research.get("trend_sources_file", "") or "").strip()
        if sources_rel:
            src_path = Path(sources_rel)
            if not src_path.is_absolute():
                src_path = self.config.root_dir / src_path
            if src_path.exists():
                try:
                    reg = json.loads(src_path.read_text(encoding="utf-8"))
                    for p_data in (reg.get("pillars") or {}).values():
                        for f in (p_data.get("rss_feeds") or []):
                            u = str(f.get("url") or "").strip()
                            if u and u not in feeds:
                                feeds.append(u)
                except Exception:
                    pass
        for feed_url in feeds:
            try:
                req = urllib.request.Request(
                    feed_url, headers={"User-Agent": "AntigravityResearchBot/1.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    xml_bytes = resp.read()
                root = ET.fromstring(xml_bytes)
                items = root.findall(".//item")[:limit_per_feed]
                if not items:
                    items = [el for el in root.iter() if el.tag.lower().endswith("entry")][:limit_per_feed]
                for item in items:
                    title = (item.findtext("title") or "").strip()
                    desc = (item.findtext("description") or item.findtext("summary") or "").strip()[:240]
                    link = (item.findtext("link") or "").strip()
                    if not title:
                        for ch in item:
                            t_low = ch.tag.lower().split("}")[-1]
                            if t_low == "title" and ch.text:
                                title = ch.text.strip()
                            elif t_low in ("summary", "content") and ch.text and not desc:
                                desc = ch.text.strip()[:240]
                            elif t_low == "link" and not link:
                                link = (ch.attrib.get("href") or ch.text or "").strip()
                    if title:
                        signals.append(
                            {
                                "source": "rss",
                                "title": title,
                                "summary": desc,
                                "url": link,
                            }
                        )
            except Exception as exc:
                logger.debug("RSS fetch skipped for %s: %s", feed_url, exc)
        return signals

    def _fetch_duckduckgo_signals(self, niche: str, limit: int = 5) -> list[dict[str, str]]:
        signals: list[dict[str, str]] = []
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            queries: list[str] = list(self.config.research.get("search_queries", []))
            if niche:
                queries = [f"{niche} surprising facts breakthrough"] + queries[:1]
            with DDGS() as ddgs:
                for q in queries[:2]:
                    try:
                        results = ddgs.text(q, region="us-en", max_results=limit)
                    except TypeError:
                        results = ddgs.text(q, max_results=limit)
                    for r in results:
                        signals.append(
                            {
                                "source": "duckduckgo",
                                "title": r.get("title", ""),
                                "summary": r.get("body", "")[:240],
                                "url": r.get("href", ""),
                            }
                        )
        except Exception as exc:
            logger.debug("DuckDuckGo search skipped: %s", exc)
        return signals

    @staticmethod
    def _parse_approx_traffic(raw: str) -> int:
        cleaned = (raw or "").strip().upper().replace(",", "").replace("+", "")
        if not cleaned:
            return 0
        mult = 1
        if cleaned.endswith("K"):
            mult = 1_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith("M"):
            mult = 1_000_000
            cleaned = cleaned[:-1]
        try:
            return int(float(cleaned) * mult)
        except ValueError:
            digits = re.sub(r"[^0-9]", "", cleaned)
            return int(digits) * mult if digits else 0

    def _fetch_google_trends_rss(self, limit: int = 12) -> list[dict[str, Any]]:
        geo = str(self.config.research.get("google_trends_geo", "US") or "US").upper()
        urls = [
            f"https://trends.google.com/trending/rss?geo={geo}",
            f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}",
        ]
        signals: list[dict[str, Any]] = []
        for feed_url in urls:
            try:
                req = urllib.request.Request(
                    feed_url, headers={"User-Agent": "AntigravityResearchBot/1.0"}
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    xml_bytes = resp.read()
                root = ET.fromstring(xml_bytes)
                items = root.findall(".//item")[:limit]
                for item in items:
                    title = (item.findtext("title") or "").strip()
                    desc = (item.findtext("description") or "").strip()[:240]
                    link = (item.findtext("link") or "").strip()
                    traffic_raw = ""
                    for child in item:
                        if child.tag.lower().endswith("approx_traffic") and child.text:
                            traffic_raw = child.text.strip()
                            break
                    if title:
                        signals.append(
                            {
                                "source": "google_trends_rss",
                                "title": title,
                                "summary": desc,
                                "url": link,
                                "approx_traffic": self._parse_approx_traffic(traffic_raw),
                            }
                        )
                if signals:
                    break
            except Exception as exc:
                logger.debug("Google Trends RSS skipped for %s: %s", feed_url, exc)
        return signals

    def _fetch_curated_trends_queue(self) -> list[dict[str, Any]]:
        rel_path = str(
            self.config.research.get("trends_queue_file", "data/trends_queue.json")
            or "data/trends_queue.json"
        )
        queue_path = Path(rel_path)
        if not queue_path.is_absolute():
            queue_path = self.config.root_dir / queue_path
        if not queue_path.exists():
            return []
        try:
            payload = json.loads(queue_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Curated trends queue read skipped for %s: %s", queue_path, exc)
            return []

        items = payload.get("trends", []) if isinstance(payload, dict) else []
        signals: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "PENDING")).strip().upper()
            if status != "PENDING":
                continue
            topic = str(item.get("topic", "")).strip()
            if not topic:
                continue
            category = str(item.get("category", "general")).strip() or "general"
            summary = str(item.get("summary", "")).strip()
            signals.append(
                {
                    "source": "curated_trends_queue",
                    "title": topic,
                    "summary": f"[{category}] {summary}".strip(),
                    "url": str(item.get("url", "")).strip(),
                    "approx_traffic": int(item.get("approx_traffic") or 0),
                    "priority": int(item.get("priority") or 80),
                    "category": category,
                }
            )
        return signals

    def _fetch_duckduckgo_tiktok_trends(self, limit: int = 5) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            queries = list(
                self.config.research.get(
                    "tiktok_trend_queries",
                    [
                        "trending on tiktok today viral story",
                        "tiktok viral trend explained",
                    ],
                )
            )
            max_queries = int(self.config.research.get("max_trend_queries", 6))
            with DDGS() as ddgs:
                for q in queries[:max_queries]:
                    try:
                        results = ddgs.text(q, region="us-en", max_results=limit)
                    except TypeError:
                        results = ddgs.text(q, max_results=limit)
                    for r in results:
                        title = str(r.get("title", "")).strip()
                        if title:
                            signals.append(
                                {
                                    "source": "duckduckgo_tiktok_trends",
                                    "title": title,
                                    "summary": str(r.get("body", ""))[:240],
                                    "url": str(r.get("href", "")),
                                    "approx_traffic": 0,
                                }
                            )
        except Exception as exc:
            logger.debug("DuckDuckGo TikTok trend search skipped: %s", exc)
        return signals

    def _is_copyright_risky(self, text: str) -> str | None:
        lower = text.lower()
        default_blocked = [
            "full movie",
            "leaked clip",
            "music video",
            "official trailer",
            "live stream",
            "nfl highlights",
            "nba highlights",
            "ufc fight",
            "celebrity gossip",
            "deepfake",
            "movie scene",
            "tv episode",
            "reupload",
        ]
        blocked = self.config.research.get("blocked_copyright_keywords", default_blocked)
        for kw in blocked:
            clean_kw = str(kw).strip().lower()
            if clean_kw and clean_kw in lower:
                return clean_kw
        return None

    def _score_trend_signal(self, signal: dict[str, Any]) -> float:
        source = str(signal.get("source", "")).lower()
        title = str(signal.get("title", "")).strip()
        summary = str(signal.get("summary", "")).strip()
        combined = f"{title} {summary}".lower()
        traffic = int(signal.get("approx_traffic") or 0)

        cues = (
            "tiktok", "viral", "trending", "explained", "why",
            "secret", "psychology", "ai", "mystery", "history", "science",
            "game", "gaming", "gta", "zelda", "switch", "playstation", "xbox",
            "movie", "film", "box office", "series", "season", "netflix",
            "anime", "manga", "llm", "agent", "gemini", "openai", "claude",
            "tech", "technology", "cyber", "cloud", "software", "chip", "it",
        )
        has_cue = any(c in combined for c in cues)

        if source == "curated_trends_queue":
            priority = max(0, int(signal.get("priority") or 80))
            score = 0.78 + min(0.18, priority / 1000.0) + (0.02 if has_cue else 0.0)
        elif source == "google_trends_rss":
            base = 0.65
            traffic_bonus = min(0.30, math.log10(max(traffic, 1000)) * 0.05) if traffic > 0 else 0.05
            score = base + traffic_bonus + (0.03 if has_cue else 0.0)
        elif source == "duckduckgo_tiktok_trends":
            score = 0.72 + (0.08 if has_cue else 0.0)
        else:
            score = 0.55 + (0.06 if has_cue else 0.0)

        if len(title) < 8:
            score -= 0.20
        return round(max(0.05, min(0.99, score)), 4)

    def discover_top_tiktok_trend(
        self, signals: list[dict[str, Any]], out_dir: Path
    ) -> dict[str, Any]:
        sim_threshold = float(self.config.qa_and_publish.get("max_similarity_threshold", 0.55))
        past_records = self.memory.get_past_topics_and_scripts()
        past_topics = [r["topic"] for r in past_records if r.get("topic")]
        past_topics_lower = [t.lower() for t in past_topics]

        seen_titles: set[str] = set()
        filtered_copyright: list[dict[str, str]] = []
        filtered_duplicates: list[dict[str, Any]] = []
        eligible: list[dict[str, Any]] = []

        for sig in signals:
            title = str(sig.get("title", "")).strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)

            combined = f"{title} {sig.get('summary', '')}"
            risk_kw = self._is_copyright_risky(combined)
            if risk_kw:
                filtered_copyright.append({"topic": title, "reason": f"blocked_keyword:{risk_kw}"})
                continue

            sim = self.memory.max_similarity_against_history(combined)
            is_substr_dup = any(key in pt or pt in key for pt in past_topics_lower if len(pt) >= 6)
            if sim >= sim_threshold or is_substr_dup:
                filtered_duplicates.append({"topic": title, "similarity": sim})
                continue

            score = self._score_trend_signal(sig)
            eligible.append(
                {
                    "topic": title,
                    "source": sig.get("source", "unknown"),
                    "approx_traffic": int(sig.get("approx_traffic") or 0),
                    "summary": str(sig.get("summary", ""))[:180],
                    "url": str(sig.get("url", "")),
                    "score": score,
                }
            )

        ranked = sorted(eligible, key=lambda x: (x["score"], x["approx_traffic"]), reverse=True)
        if ranked:
            selected = ranked[0]
        else:
            offline = self._offline_opportunities(
                niche=self.config.project.get("niche", ""),
                signals=[],
                past_topics=past_topics,
            )
            first_opp = offline["opportunities"][0]
            selected = {
                "topic": first_opp["topic"],
                "source": "offline_curated_trend",
                "approx_traffic": 0,
                "summary": first_opp["unique_angle"],
                "url": "",
                "score": float(first_opp.get("score", 0.90)),
            }

        discovery = {
            "mode": "auto_top_trend",
            "total_signals_scanned": len(signals),
            "filtered_out_copyright": filtered_copyright,
            "filtered_out_memory_duplicate": filtered_duplicates,
            "ranked_candidates": ranked[:5],
            "selected_top_trend": selected,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trend_discovery.json").write_text(
            json.dumps(discovery, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return discovery

    def _fetch_ytdlp_viral_patterns(self, niche: str, max_items: int = 5) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        try:
            import yt_dlp

            query = f"ytsearch{max_items}:{niche} shorts documentary mystery explained"
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(query, download=False)
            for entry in (info or {}).get("entries", []) or []:
                patterns.append(
                    {
                        "source": "youtube_ytdlp",
                        "title": entry.get("title", ""),
                        "view_count": entry.get("view_count") or 0,
                        "duration": entry.get("duration") or 0,
                        "id": entry.get("id", ""),
                    }
                )
        except Exception as exc:
            logger.debug("yt-dlp research skipped: %s", exc)
        return patterns

    def _synthesize_with_llm(
        self,
        niche: str,
        signals: list[dict[str, Any]],
        past_topics: list[str],
        max_opportunities: int,
        revision_feedback: str | None = None,
        trend_topic: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.llm:
            return None

        system_prompt = (
            "You are AI Model 1 (Viral TikTok Research Strategist). "
            "Never copy or plagiarize scripts. Analyze current viral TikTok storytelling mechanics: "
            "jarring 3-second open loops, 2-3s visual pacing, visual-first storytelling (scenes that look "
            "cinematic in AI-generated images/videos), mid-video twists, and comment-bait endings for "
            "65-85s TikTok Creator Rewards videos."
        )
        task_text = (
            f"Build {max_opportunities} original, monetization-safe, zero-plagiarism deep-dive angles riding "
            f"the live TikTok trend '{trend_topic}' (avoid reused movie/broadcast clips or celebrity deepfakes; "
            "favor original commentary, hidden economics, psychology, lore, or behind-the-scenes analysis)."
            if trend_topic
            else f"Propose {max_opportunities} original viral video opportunities in niche: {niche}"
        )
        payload: dict[str, Any] = {
            "task": task_text,
            "target_language": self.config.project.get("language", "en"),
            "avoid_past_topics": past_topics[:15],
            "fresh_signals": signals[:15],
            "required_json_schema": {
                "analyzed_patterns": {
                    "top_3s_hook_archetypes": ["string"],
                    "retention_pacing_notes": "string",
                    "save_and_share_triggers": ["string"],
                },
                "opportunities": [
                    {
                        "topic": "string",
                        "unique_angle": "string",
                        "target_audience": "string",
                        "hook_pattern": "string",
                        "narrative_structure": "string",
                        "viral_triggers": ["string"],
                        "keywords": ["string"],
                        "source_signals": ["string"],
                        "score": 0.9,
                    }
                ],
            },
        }
        if trend_topic:
            payload["live_tiktok_trend_to_ride"] = trend_topic
        if revision_feedback:
            payload["reviewer_critique_to_fix"] = revision_feedback

        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            return self.llm.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:
            logger.error("AI Model 1 (Research) Gemini call failed: %s", exc)
            return None

    def _offline_opportunities(
        self,
        niche: str,
        signals: list[dict[str, Any]],
        past_topics: list[str],
        trend_topic: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic topic-grounded opportunities when local LLM is offline."""
        signal_titles = [s.get("title", "") for s in signals if s.get("title")][:3]
        candidates: list[dict[str, Any]] = []
        if trend_topic:
            clean_kw = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", trend_topic)][:6]
            candidates.append(
                {
                    "topic": trend_topic,
                    "unique_angle": (
                        f"Deep-dive breakdown of '{trend_topic}': why it exploded, the core numbers behind it, "
                        "and what it means next using real B-roll footage."
                    ),
                    "target_audience": "General adult TikTok audience (18-45)",
                    "hook_pattern": (
                        f"Curiosity hook: 'Everyone is watching {trend_topic}—but almost nobody noticed what happens next.'"
                    ),
                    "narrative_structure": "3s Trend Hook -> Core Facts -> ~35s Pattern Interrupt -> Stakes & Fallout -> Debate CTA",
                    "viral_triggers": ["Live Trend Wave", "Insider Breakdown", "Comment Debate"],
                    "keywords": clean_kw or ["trending"],
                    "source_signals": [f"live_tiktok_trend:{trend_topic}", *signal_titles],
                    "score": 0.98,
                }
            )

        for sig in signals[:5]:
            t = str(sig.get("title") or "").strip()
            if not t:
                continue
            clean_kw = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", t)][:6]
            candidates.append(
                {
                    "topic": t,
                    "unique_angle": str(sig.get("summary") or f"Original TikTok deep-dive on {t}.").strip()[:220],
                    "target_audience": "General adult TikTok audience (18-45)",
                    "hook_pattern": f"Wait until you see the real story behind {t}.",
                    "narrative_structure": "3s Hook -> Key Facts -> Twist -> Impact -> Loop CTA",
                    "viral_triggers": ["Curiosity Gap", "Real Footage", "Comment Debate"],
                    "keywords": clean_kw or ["trending"],
                    "source_signals": [str(sig.get("source") or "signal")],
                    "score": float(self._score_trend_signal(sig)),
                }
            )

        if not candidates:
            fallback_topic = f"Biggest {niche.split(',')[0].strip() or 'Tech'} Breakthroughs Explained"
            candidates.append(
                {
                    "topic": fallback_topic,
                    "unique_angle": f"Original analytical breakdown of {fallback_topic}.",
                    "target_audience": "General adult TikTok audience (18-45)",
                    "hook_pattern": f"Nobody expected how fast {fallback_topic} would change everything.",
                    "narrative_structure": "3s Hook -> Evidence -> Twist -> Impact -> Loop CTA",
                    "viral_triggers": ["Curiosity Gap", "Share to Friend"],
                    "keywords": ["trending", "explained"],
                    "source_signals": ["fallback_topic"],
                    "score": 0.85,
                }
            )

        past_lower = " ".join(past_topics).lower()
        filtered = [c for c in candidates if c["topic"].lower() not in past_lower]
        if not filtered:
            filtered = candidates

        return {
            "analyzed_patterns": {
                "top_3s_hook_archetypes": [
                    "Direct trend curiosity gap in first 3 seconds",
                    "Mid-video reveal around second 35",
                    "Debate-focused loopable ending",
                ],
                "retention_pacing_notes": "Keep sentences punchy and shift B-roll clips every 6-7 seconds.",
                "save_and_share_triggers": ["Verifiable facts", "Clear numbers", "Open verdict"],
            },
            "opportunities": filtered,
        }

    def run(
        self,
        run_id: str,
        niche: str,
        out_dir: Path,
        revision_feedback: str | None = None,
        trend_topic: str | None = None,
        auto_top_trend: bool = False,
    ) -> ResearchReport:
        sources = set(
            self.config.research.get(
                "sources", ["google_trends_rss", "duckduckgo_tiktok_trends", "rss"]
            )
        )
        if auto_top_trend and not trend_topic:
            sources.update(["google_trends_rss", "duckduckgo_tiktok_trends"])

        signals: list[dict[str, Any]] = []

        if "curated_trends_queue" in sources:
            signals.extend(self._fetch_curated_trends_queue())
        if "google_trends_rss" in sources:
            signals.extend(self._fetch_google_trends_rss())
        if "duckduckgo_tiktok_trends" in sources:
            signals.extend(self._fetch_duckduckgo_tiktok_trends())
        if "rss" in sources:
            signals.extend(self._fetch_rss_signals())
        if "duckduckgo" in sources:
            signals.extend(self._fetch_duckduckgo_signals(niche))
        if "youtube_ytdlp" in sources:
            signals.extend(self._fetch_ytdlp_viral_patterns(niche))

        if bool(self.config.project.get("enforce_english_only", True)):
            allow_translate = bool(self.config.project.get("translate_non_english_sources", True))
            if trend_topic and not is_english_text(trend_topic):
                trend_topic = ensure_english_text(
                    trend_topic, fallback=trend_topic, allow_translate=allow_translate
                )
            translated_signals: list[dict[str, Any]] = []
            for sig in signals:
                t_raw = str(sig.get("title") or "").strip()
                s_raw = str(sig.get("summary") or "").strip()
                t_en = ensure_english_text(t_raw, fallback="", allow_translate=allow_translate)
                if not t_en:
                    continue
                s_en = ensure_english_text(s_raw, fallback="", allow_translate=allow_translate) if s_raw else ""
                sig_copy = dict(sig)
                sig_copy["title"] = t_en
                sig_copy["summary"] = s_en
                translated_signals.append(sig_copy)
            signals = translated_signals

        discovery: dict[str, Any] | None = None
        if auto_top_trend and not trend_topic:
            discovery = self.discover_top_tiktok_trend(signals, out_dir)
            trend_topic = str(discovery["selected_top_trend"]["topic"])
            logger.info("[AI Model 1 - Auto Top Trend] Selected #1 trend: %s", trend_topic)

        past_records = self.memory.get_past_topics_and_scripts()
        past_topics = [r["topic"] for r in past_records if r.get("topic")]

        max_opps = int(self.config.research.get("max_opportunities", 3))
        synthesized = self._synthesize_with_llm(
            niche,
            signals,
            past_topics,
            max_opps,
            revision_feedback=revision_feedback,
            trend_topic=trend_topic,
        )
        if not synthesized or not synthesized.get("opportunities"):
            synthesized = self._offline_opportunities(
                niche, signals, past_topics, trend_topic=trend_topic
            )

        analyzed_patterns = dict(synthesized.get("analyzed_patterns", {}))
        if discovery is not None:
            analyzed_patterns["auto_top_trend_selection"] = discovery["selected_top_trend"]

        opportunities = [
            ViralOpportunity(
                topic=item.get("topic", "Untitled Topic"),
                unique_angle=item.get("unique_angle", ""),
                target_audience=item.get("target_audience", "General Adult"),
                hook_pattern=item.get("hook_pattern", ""),
                narrative_structure=item.get("narrative_structure", ""),
                viral_triggers=list(item.get("viral_triggers", [])),
                keywords=list(item.get("keywords", [])),
                source_signals=list(item.get("source_signals", [])),
                score=float(item.get("score", 0.8)),
            )
            for item in synthesized.get("opportunities", [])
        ]

        if discovery is not None and trend_topic and opportunities:
            if trend_topic.lower() not in opportunities[0].topic.lower():
                opportunities[0].topic = trend_topic

        report = ResearchReport(
            run_id=run_id,
            niche=niche,
            created_at=datetime.now(timezone.utc).isoformat(),
            analyzed_patterns=analyzed_patterns,
            opportunities=opportunities,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "research_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_markdown(report, out_dir / "research_report.md")
        return report

    def _write_markdown(self, report: ResearchReport, path: Path) -> None:
        lines = [
            f"# Trend & Pattern Research Report ({report.run_id})",
            f"- **Niche:** {report.niche}",
            f"- **Created At:** {report.created_at}",
            "",
            "## Viral Pattern Analysis",
            "```json",
            json.dumps(report.analyzed_patterns, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Content Opportunities",
        ]
        for idx, opp in enumerate(report.opportunities, start=1):
            lines.extend(
                [
                    f"### {idx}. {opp.topic} (Score: {opp.score:.2f})",
                    f"- **Unique Angle:** {opp.unique_angle}",
                    f"- **Target Audience:** {opp.target_audience}",
                    f"- **3s Hook Pattern:** {opp.hook_pattern}",
                    f"- **Narrative Structure:** {opp.narrative_structure}",
                    f"- **Viral Triggers:** {', '.join(opp.viral_triggers)}",
                    f"- **Keywords:** {', '.join(opp.keywords)}",
                    "",
                ]
            )
        path.write_text("\n".join(lines), encoding="utf-8")
