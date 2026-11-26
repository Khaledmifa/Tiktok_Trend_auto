from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.models import (
    MediaManifest,
    PlatformMetadata,
    QACheckResult,
    ReviewPackage,
    ReviewerVerdict,
    ScriptPlan,
)
from src.providers.llm import build_llm_provider
from src.state.memory_store import MemoryStore
from src.utils.ffmpeg_tools import probe_media_duration, resolve_ffmpeg_executable
from src.utils.language_guard import is_english_text, is_subtitle_file_english

logger = logging.getLogger(__name__)


class QAPublisherAgent:
    """AI Publisher & Human-in-the-Loop Gate: Audits video quality/compliance, prepares Review Package, and gates publishing on HITL approval."""

    def __init__(
        self, config: AppConfig, memory: MemoryStore, skip_external_llm: bool = False
    ) -> None:
        self.config = config
        self.memory = memory
        if skip_external_llm:
            self.llm = None
        else:
            model_name = config.models.get("metadata_model")
            self.llm = build_llm_provider(config, model_override=model_name)

    def _run_qa(
        self, plan: ScriptPlan, manifest: MediaManifest, verdict: ReviewerVerdict | None = None
    ) -> QACheckResult:
        issues: list[str] = []
        video_path = Path(manifest.final_video_path)
        srt_path = Path(manifest.srt_path)
        ass_path = Path(manifest.ass_path)

        floor_sec = float(self.config.video.get("hard_monetization_floor_sec", 61))
        max_sec = float(self.config.video.get("max_duration_sec", 95)) + 5.0
        sim_threshold = float(self.config.qa_and_publish.get("max_similarity_threshold", 0.55))

        duration = manifest.actual_duration_sec
        if duration <= 0 and video_path.exists():
            ffmpeg_exe = resolve_ffmpeg_executable()
            duration = probe_media_duration(ffmpeg_exe, video_path)

        duration_ok = floor_sec <= duration <= max_sec
        if not duration_ok:
            issues.append(
                f"Duration {duration:.1f}s outside required monetization range [{floor_sec}s, {max_sec}s]."
            )

        audio_present = False
        if video_path.exists() and video_path.stat().st_size > 10000:
            ffmpeg_exe = resolve_ffmpeg_executable()
            probe = subprocess.run(
                [ffmpeg_exe, "-i", str(video_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            audio_present = "Audio:" in probe.stderr
        if not audio_present:
            issues.append("No audio stream detected in final rendered MP4.")

        captions_synced = (
            srt_path.exists()
            and srt_path.stat().st_size > 20
            and ass_path.exists()
            and ass_path.stat().st_size > 50
        )
        if not captions_synced:
            issues.append("Subtitle files (.srt/.ass) missing or empty.")

        completed_statuses = ("READY_FOR_REVIEW", "APPROVED_AND_PACKAGED")
        topic_sim = self.memory.max_topic_similarity_against_history(
            plan.topic,
            exclude_run_id=plan.run_id,
            statuses=completed_statuses,
        )
        full_sim = self.memory.max_similarity_against_history(
            f"{plan.topic} {plan.opening_hook_3s} {plan.full_voiceover_text}",
            exclude_run_id=plan.run_id,
            statuses=completed_statuses,
        )
        max_sim = max(topic_sim, full_sim) if topic_sim >= 0.35 else topic_sim
        originality_ok = max_sim < sim_threshold
        if not originality_ok:
            issues.append(
                f"High similarity ({max_sim:.2f}) against previously produced video in memory store."
            )

        ai_ok = True if verdict is None else (verdict.verdict == "PASS")
        if verdict is not None and not ai_ok:
            issues.append(
                f"AI Reviewer ({verdict.responsible_agent}): {verdict.error_diagnosis}"
            )

        english_ok = True
        if bool(self.config.project.get("enforce_english_only", True)):
            if (plan.language or "en").lower() != "en":
                english_ok = False
            elif any(s.voiceover_text and not is_english_text(s.voiceover_text) for s in plan.scenes):
                english_ok = False
            elif srt_path.exists():
                srt_ok, _ = is_subtitle_file_english(srt_path)
                if not srt_ok:
                    english_ok = False
            if not english_ok:
                issues.append("Non-English text detected; all videos must be strictly in English.")

        return QACheckResult(
            passed=duration_ok and audio_present and captions_synced and originality_ok and ai_ok and english_ok,
            duration_ok=duration_ok,
            audio_present=audio_present,
            captions_synced=captions_synced,
            originality_ok=originality_ok,
            max_past_similarity=max_sim,
            ai_review_verdict=verdict.verdict if verdict else "SKIPPED",
            issues=issues,
        )

    def _build_metadata(
        self, plan: ScriptPlan, metadata_override: dict[str, Any] | None = None
    ) -> PlatformMetadata:
        tt_cfg = self.config.qa_and_publish.get("tiktok", {})
        ai_flag = bool(self.config.qa_and_publish.get("ai_generated_disclosure", True))
        tt_disclose = bool(tt_cfg.get("disclose_ai_content", ai_flag))

        if metadata_override:
            return PlatformMetadata(
                tiktok_title=str(metadata_override.get("tiktok_title", plan.topic[:80])),
                tiktok_caption=str(
                    metadata_override.get("tiktok_caption", plan.opening_hook_3s[:140])
                ),
                tiktok_description=str(
                    metadata_override.get("tiktok_description", plan.unique_angle)
                ),
                tiktok_hashtags=list(
                    metadata_override.get("tiktok_hashtags", ["learnontiktok", "storytime", "explained"])
                ),
                tiktok_disclose_ai_content=tt_disclose,
            )

        if self.llm:
            try:
                raw = self.llm.generate_json(
                    prompt=json.dumps(
                        {
                            "topic": plan.topic,
                            "angle": plan.unique_angle,
                            "hook": plan.opening_hook_3s,
                            "language": plan.language,
                            "schema": {
                                "tiktok_title": "punchy title <= 75 chars",
                                "tiktok_caption": "high-retention curiosity caption <= 140 chars",
                                "tiktok_description": "1-2 concise sentences summarizing original angle",
                                "tiktok_hashtags": ["4-6 hashtags without #"],
                            },
                        },
                        ensure_ascii=False,
                    ),
                    system_prompt="Create compliant, high-retention metadata exclusively for TikTok Creator Rewards.",
                )
                return PlatformMetadata(
                    tiktok_title=str(raw.get("tiktok_title", plan.topic[:80])),
                    tiktok_caption=str(raw.get("tiktok_caption", plan.opening_hook_3s[:140])),
                    tiktok_description=str(raw.get("tiktok_description", plan.unique_angle)),
                    tiktok_hashtags=list(
                        raw.get("tiktok_hashtags", ["learnontiktok", "storytime", "explained"])
                    ),
                    tiktok_disclose_ai_content=tt_disclose,
                )
            except Exception as exc:
                logger.warning("Metadata LLM generation failed (%s); using deterministic metadata.", exc)

        cat_tags = {
            "MOVIES EXPLAINED": ["movies", "movietok", "cinemaexplained", "filmbreakdown"],
            "SERIES EXPLAINED": ["tvseries", "tvtok", "streaming", "seriesexplained"],
            "GAMING EXPLAINED": ["gaming", "gamingnews", "videogames", "gameexplained"],
            "AI EXPLAINED": ["ai", "aitrends", "artificialintelligence", "aiforgaming"],
        }.get(getattr(plan, "category_label", ""), ["movies", "gaming", "ai", "explained"])
        clean_caption = (
            f"{plan.opening_hook_3s[:135].rstrip()} "
            "Official clips & commentary."
        ).strip()
        return PlatformMetadata(
            tiktok_title=plan.topic[:80],
            tiktok_caption=clean_caption,
            tiktok_description=(
                f"{plan.unique_angle} "
                "Commentary and analysis with AI-generated narration."
            ),
            tiktok_hashtags=cat_tags[:4],
            tiktok_disclose_ai_content=tt_disclose,
        )

    def prepare_review_package(
        self,
        run_id: str,
        plan: ScriptPlan,
        manifest: MediaManifest,
        out_dir: Path,
        reviewer_verdict: ReviewerVerdict | None = None,
        metadata_override: dict[str, Any] | None = None,
    ) -> ReviewPackage:
        """Stop at Human-in-the-Loop Gate: never auto-publish."""
        qa = self._run_qa(plan, manifest, verdict=reviewer_verdict)
        metadata = self._build_metadata(plan, metadata_override=metadata_override)
        status = "READY_FOR_REVIEW" if qa.passed else "QA_FAILED"

        review_md_path = out_dir / "review_package.md"
        pkg = ReviewPackage(
            run_id=run_id,
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            topic=plan.topic,
            unique_angle=plan.unique_angle,
            video_path=manifest.final_video_path,
            srt_path=manifest.srt_path,
            duration_sec=manifest.actual_duration_sec,
            qa=qa,
            metadata=metadata,
            review_markdown_path=str(review_md_path),
            reviewer_verdict=reviewer_verdict,
        )

        (out_dir / "review_package.json").write_text(
            json.dumps(pkg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_review_markdown(pkg, manifest, review_md_path)
        if bool(self.config.qa_and_publish.get("tiktok", {}).get("generate_publish_kit", True)):
            self._write_publish_kit(pkg, plan, manifest, out_dir / "publish_kit.md")

        self.memory.update_run_review_ready(
            run_id=run_id,
            video_path=manifest.final_video_path,
            duration_sec=manifest.actual_duration_sec,
            status=status,
        )
        return pkg

    def _write_publish_kit(
        self, pkg: ReviewPackage, plan: ScriptPlan, manifest: MediaManifest, path: Path
    ) -> None:
        hashtags_str = " ".join(f"#{h.lstrip('#')}" for h in pkg.metadata.tiktok_hashtags[:5])
        disclosure = (
            "ON (Required — AI/synthetic content disclosure)"
            if pkg.metadata.tiktok_disclose_ai_content
            else "OFF"
        )
        am = manifest.audio_metrics or {}
        audio_summary = (
            f"integrated={am.get('integrated_lufs', 'N/A')} LUFS | "
            f"true_peak={am.get('true_peak_dbtp', 'N/A')} dBTP | "
            f"clipped_samples={am.get('clipped_samples', 'N/A')}"
            if am.get("measured")
            else "Not measured"
        )
        lines = [
            f"# Viral Studio Pro — TikTok Publish Kit ({pkg.run_id})",
            "",
            "- **Upload Mode:** Manual upload by creator ONLY (automated public upload disabled).",
            f"- **Title:** {pkg.metadata.tiktok_title}",
            f"- **Video File:** `{pkg.video_path}`",
            f"- **Cover Image:** `{manifest.cover_path or 'N/A'}`",
            f"- **Loop QA Transition Clip:** `{manifest.loop_qa_path or 'N/A'}`",
            f"- **Duration:** `{pkg.duration_sec:.2f}s` (`> 61s` floor; length alone does not guarantee monetization eligibility)",
            f"- **Audio Mastering:** `{audio_summary}`",
            f"- **AI/Synthetic Disclosure Toggle:** `{disclosure}`",
            f"- **Location Tag Guidance:** `None (leave location blank for global reach)`",
            f"- **Footage Context Note:** `{plan.footage_context_label}` — {plan.attribution_line}",
            "",
            "## Seamless Loop Check (QA Back-to-Back Before Upload)",
            f"1. **Opening Hook:** {plan.opening_hook_3s}",
            f"2. **Final Connecting Phrase:** {plan.loop_bridge_phrase or plan.closing_cta}",
            f"3. **Joined Replay Transition:** {plan.loop_joined_preview or (plan.closing_cta + ' ' + plan.opening_hook_3s)}",
            f"4. **Editing & Sound Notes:** {plan.loop_cut_notes or 'Play loop_qa_transition.mp4 back-to-back to verify visual & audio continuity; treat improved rewatching as a testable hypothesis, not a view guarantee.'}",
            "",
            "## Copy-Paste Hook Caption",
            "```text",
            pkg.metadata.tiktok_caption,
            "```",
            "",
            "## Copy-Paste Targeted Hashtags (3-5)",
            "```text",
            hashtags_str,
            "```",
            "",
            "## Full Ready-to-Paste Post Text",
            "```text",
            f"{pkg.metadata.tiktok_caption}\n\n{hashtags_str}",
            "```",
            "",
            "## Full Voiceover Script",
            "```text",
            plan.full_voiceover_text,
            "```",
            "",
            "## Description / Context",
            pkg.metadata.tiktok_description,
            "",
        ]
        if plan.fact_sources:
            lines.append("## Fact Sources")
            lines.extend(f"- {src}" for src in plan.fact_sources)
            lines.append("")
        if plan.video_sources:
            lines.append("## Video Sources")
            lines.extend(f"- {src}" for src in plan.video_sources)
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_review_markdown(
        self, pkg: ReviewPackage, manifest: MediaManifest, path: Path
    ) -> None:
        qa_badge = "PASS" if pkg.qa.passed else "FAILED"
        lines = [
            f"# حزمة المراجعة النهائية — بوابة الموافقة البشرية ({pkg.run_id})",
            "",
            f"- **الحالة (Status):** `{pkg.status}`",
            f"- **الموضوع:** {pkg.topic}",
            f"- **الزاوية الأصلية:** {pkg.unique_angle}",
            f"- **المدة الفعلية:** `{pkg.duration_sec:.2f}s` (الحد الأدنى للربح > 60 ثانية)",
            f"- **مسار الفيديو النهائي:** `{pkg.video_path}`",
            f"- **فيديو فحص التكرار المتصل (Loop QA):** `{manifest.loop_qa_path or 'N/A'}`",
            f"- **ملف الترجمة (SRT):** `{pkg.srt_path}`",
            f"- **نتيجة فحص الجودة (QA):** `{qa_badge}` | التشابه مع الفيديوهات السابقة: `{pkg.qa.max_past_similarity:.2f}`",
            "",
        ]
        if pkg.qa.issues:
            lines.append("## ملاحظات فحص الجودة")
            for iss in pkg.qa.issues:
                lines.append(f"- {iss}")
            lines.append("")

        if pkg.reviewer_verdict is not None:
            rv = pkg.reviewer_verdict
            lines.extend(
                [
                    "## تقرير المراجع البصري الذكي (AI Model 6 - Vision Reviewer)",
                    f"- **قرار المراجعة:** `{rv.verdict}` (جولة التصحيح: `{rv.revision_round}`)",
                    f"- **الوكيل المسؤول عند الإرجاع:** `{rv.responsible_agent}`",
                    f"- **التشخيص:** {rv.error_diagnosis}",
                ]
            )
            if rv.scene_audits:
                lines.append("- **تقييم المشاهد (1-10):**")
                for sa in rv.scene_audits:
                    note = f" — {sa.issue}" if sa.issue else ""
                    lines.append(
                        f"  - Scene {sa.scene_number}: `{sa.score}/10` (match=`{sa.matches_script}`){note}"
                    )
            lines.append("")

        lines.extend(
            [
                "## البيانات الوصفية المقترحة (TikTok)",
                f"- **العنوان (Title):** {pkg.metadata.tiktok_title}",
                f"- **النص (Caption):** {pkg.metadata.tiktok_caption}",
                f"- **الوصف (Description):** {pkg.metadata.tiktok_description}",
                f"- **الهاشتاغات:** {' '.join('#' + h for h in pkg.metadata.tiktok_hashtags)}",
                f"- **تفعيل علامة المحتوى المولّد بالذكاء الاصطناعي:** `{pkg.metadata.tiktok_disclose_ai_content}`",
                "",
                "## مصادر المرئيات المستخدمة",
            ]
        )
        for src in manifest.visual_sources_used:
            p_prov = src.get("photo_provider") or src.get("provider", "unknown")
            v_prov = src.get("video_provider", "")
            lines.append(
                f"- Scene {src['scene']}: photo=`{p_prov}` video=`{v_prov}` ({src['attribution']})"
            )

        lines.extend(
            [
                "",
                "---",
                "### للموافقة وتجهيز النشر بعد المعاينة:",
                f"```powershell\npython run_pipeline.py --approve {pkg.run_id}\n```",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")

    def approve_and_publish(
        self,
        run_id: str,
        out_dir_override: Path | None = None,
        tiktok_bundle_override: Path | None = None,
    ) -> dict[str, Any]:
        """Executed ONLY after explicit human confirmation."""
        record = self.memory.get_run(run_id)
        if not record:
            raise ValueError(f"Run ID '{run_id}' not found in memory store.")
        if record["status"] not in ("READY_FOR_REVIEW", "APPROVED_AND_PACKAGED"):
            raise RuntimeError(
                f"Run '{run_id}' is in status '{record['status']}' and cannot be approved."
            )

        out_dir = out_dir_override or (self.config.tiktok_dir / run_id)
        if not out_dir_override and not (out_dir / "review_package.json").exists() and record.get("video_path"):
            cand_dir = Path(str(record["video_path"])).parent
            if (cand_dir / "review_package.json").exists():
                out_dir = cand_dir
        review_json = json.loads((out_dir / "review_package.json").read_text(encoding="utf-8"))
        meta = review_json["metadata"]

        tiktok_bundle = tiktok_bundle_override or (self.config.tiktok_dir / run_id)
        tiktok_bundle.mkdir(parents=True, exist_ok=True)
        src_video = Path(review_json["video_path"])
        dest_video = tiktok_bundle / src_video.name
        if src_video.exists() and src_video.resolve() != dest_video.resolve():
            shutil.copy2(src_video, dest_video)
        publish_kit = out_dir / "publish_kit.md"
        dest_kit = tiktok_bundle / "publish_kit.md"
        if publish_kit.exists() and publish_kit.resolve() != dest_kit.resolve():
            shutil.copy2(publish_kit, dest_kit)
        cover_img = out_dir / "cover.jpg"
        dest_cover = tiktok_bundle / "cover.jpg"
        if cover_img.exists() and cover_img.resolve() != dest_cover.resolve():
            shutil.copy2(cover_img, dest_cover)

        hashtags_str = " ".join(f"#{h}" for h in meta["tiktok_hashtags"])
        (tiktok_bundle / "tiktok_post_details.txt").write_text(
            "\n".join(
                [
                    f"Title: {meta.get('tiktok_title', review_json['topic'])}",
                    f"Caption: {meta['tiktok_caption']} {hashtags_str}",
                    f"Description: {meta.get('tiktok_description', '')}",
                    f"Hashtags: {hashtags_str}",
                    f"AI Content Disclosure Toggle: {'ON' if meta['tiktok_disclose_ai_content'] else 'OFF'}",
                    f"Video File: {src_video.name}",
                ]
            ),
            encoding="utf-8",
        )

        publish_info: dict[str, Any] = {
            "platform": "tiktok",
            "tiktok_bundle_dir": str(tiktok_bundle),
        }

        self.memory.mark_approved(run_id, publish_info)
        return publish_info
