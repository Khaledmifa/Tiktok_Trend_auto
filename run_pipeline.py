from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.agents.montage_agent import MontageAgent
from src.agents.photo_agent import PhotoGeneratorAgent
from src.agents.publisher_agent import QAPublisherAgent
from src.agents.research_agent import TrendResearchAgent
from src.agents.reviewer_agent import AIReviewerAgent
from src.agents.script_agent import ScriptArchitectAgent
from src.agents.video_agent import VideoGeneratorAgent
from src.config import PROJECT_ROOT, load_config
from src.state.memory_store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")


def _has_image_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        directory.glob(f"*{ext}")
        for ext in (".jpg", ".jpeg", ".png", ".webp")
        if any(directory.glob(f"*{ext}"))
    )


def run_production_pipeline(
    config_path: str,
    niche_override: str | None = None,
    reuse_photos_dir: str | None = None,
    topic_override: str | None = None,
    run_id_override: str | None = None,
    script_json_path: str | None = None,
    metadata_json_path: str | None = None,
    skip_external_llm: bool = False,
    scenes_override: int | None = None,
    auto_top_trend: bool = False,
    discover_trend_only: bool = False,
    language_override: str | None = None,
    revision_feedback: str | None = None,
    no_telegram_notify: bool = False,
    keywords_override: str | None = None,
) -> int:
    config = load_config(config_path)
    if scenes_override is not None and scenes_override > 0:
        pipe_cfg = config.raw.setdefault("pipeline", {})
        pipe_cfg["default_scenes"] = scenes_override
        pipe_cfg["min_scenes"] = scenes_override
        pipe_cfg["max_scenes"] = scenes_override

    lang_clean = language_override.strip().lower() if language_override else ""
    if bool(config.project.get("enforce_english_only", True)):
        if lang_clean and lang_clean != "en":
            logger.warning(
                "Strict English-only rule active: ignoring --lang '%s' and enforcing 'en'.",
                lang_clean,
            )
            lang_clean = ""
        config.raw.setdefault("project", {})["language"] = "en"
    elif lang_clean:
        config.raw.setdefault("project", {})["language"] = lang_clean

    skip_llm = skip_external_llm or (
        bool(script_json_path)
        and bool(config.pipeline.get("skip_external_llm_when_script_provided", True))
    )

    memory = MemoryStore(config.data_dir)
    base_run_id = run_id_override or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_id = f"{base_run_id}__{lang_clean}" if lang_clean else base_run_id
    niche = niche_override or config.project.get(
        "niche", "Games, Movies, Series, Anime, IT, AI, Technology"
    )
    run_dir = (
        config.tiktok_dir / base_run_id / lang_clean
        if lang_clean
        else config.tiktok_dir / base_run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = run_dir
    tiktok_run_dir = run_dir

    memory.record_run_start(run_id, niche)
    logger.info(
        "Starting TikTok Production Run %s | Niche: %s | skip_external_llm=%s | auto_top_trend=%s",
        run_id,
        niche,
        skip_llm,
        auto_top_trend,
    )

    research_agent = TrendResearchAgent(config, memory, skip_external_llm=skip_llm)
    script_agent = ScriptArchitectAgent(config, memory, skip_external_llm=skip_llm)
    photo_agent = PhotoGeneratorAgent(config)
    video_agent = VideoGeneratorAgent(config)
    montage_agent = MontageAgent(config)
    reviewer_agent = AIReviewerAgent(config, skip_external_llm=skip_llm)
    publisher_agent = QAPublisherAgent(config, memory, skip_external_llm=skip_llm)

    max_revisions = int(config.raw.get("models", {}).get("max_revision_rounds", 2))
    effective_max_revisions = 0 if skip_llm else max_revisions

    # --- Stage 1 & 2: Research + Script (or Antigravity-authored script JSON / revision reuse) ---
    if revision_feedback:
        from src.utils.revision_engine import apply_revision_feedback

        rev_summary = apply_revision_feedback(revision_feedback, out_dir, config)
        logger.info(
            "[Revision Engine] Matched=%s | Actions=%s | Warnings=%s",
            rev_summary.get("matched"),
            rev_summary.get("applied_actions"),
            rev_summary.get("warnings"),
        )

    effective_script_json = script_json_path
    if not effective_script_json and revision_feedback and (out_dir / "script_plan.json").exists():
        effective_script_json = str(out_dir / "script_plan.json")
    effective_metadata_json = metadata_json_path
    if not effective_metadata_json and (out_dir / "metadata.json").exists():
        effective_metadata_json = str(out_dir / "metadata.json")

    report = None
    if effective_script_json:
        plan = script_agent.load_from_json(
            run_id=run_id, script_json_path=Path(effective_script_json), out_dir=out_dir
        )
        logger.info(
            "[Hybrid Handoff - Script JSON] '%s' (%d scenes, %d words).",
            plan.topic,
            len(plan.scenes),
            plan.word_count,
        )
    else:
        report = research_agent.run(
            run_id=run_id,
            niche=niche,
            out_dir=out_dir,
            revision_feedback=revision_feedback,
            trend_topic=topic_override,
            auto_top_trend=auto_top_trend,
        )
        logger.info("[AI Model 1 - Research] Found %d opportunities.", len(report.opportunities))

        if discover_trend_only:
            top_opp = report.opportunities[0] if report.opportunities else None
            print("\n" + "=" * 76)
            print(f"RUN ID         : {run_id}")
            print(f"TOP TREND      : {top_opp.topic if top_opp else 'N/A'}")
            print(f"UNIQUE ANGLE   : {top_opp.unique_angle if top_opp else 'N/A'}")
            print(f"REPORT PATH    : {out_dir / 'research_report.json'}")
            print(f"DISCOVERY PATH : {out_dir / 'trend_discovery.json'}")
            print(f"TIKTOK FOLDER  : {tiktok_run_dir}")
            print("=" * 76)
            return 0

        plan = script_agent.run(
            run_id=run_id, report=report, out_dir=out_dir, revision_feedback=revision_feedback
        )
        logger.info(
            "[AI Model 2 - Script] '%s' (%d scenes, %d words).",
            plan.topic,
            len(plan.scenes),
            plan.word_count,
        )

    # --- Stage 5a: Voiceover & Exact Scene Timing ---
    audio_path, timings, total_dur, scene_durations = montage_agent.prepare_audio_and_scene_timings(
        plan, out_dir
    )

    # --- Auto-Harvest Real B-Roll & Web Images ($0 yt-dlp + DDGS) into tiktok/<run_id>/ ---
    if bool(config.pipeline.get("auto_harvest_ytdlp", True)):
        try:
            if str(config.root_dir) not in sys.path:
                sys.path.insert(0, str(config.root_dir))
            from scripts.auto_media_harvester import harvest_media_for_trend

            parsed_kw = [k.strip() for k in (keywords_override or "").split(",") if k.strip()]
            harvest_res = harvest_media_for_trend(
                topic=plan.topic,
                run_dir=tiktok_run_dir,
                scene_count=len(plan.scenes),
                sourcing_keywords=parsed_kw,
                max_height=int(config.video.get("harvest_max_height", 2160)),
            )
            logger.info(
                "[Auto-Harvester $0] Staged %d video clips and %d images in %s.",
                len(harvest_res.get("clips_created", [])),
                len(harvest_res.get("photos_created", [])),
                tiktok_run_dir,
            )
        except Exception as exc:
            logger.warning("Auto-harvest skipped safely (%s); falling back to staged/web visuals.", exc)

    # --- Stage 3: Visuals Preparation (staged files in tiktok/<run_id>/ or $0 web fallback) ---
    auto_tiktok = bool(config.pipeline.get("auto_reuse_tiktok_media_dir", True))
    auto_run_photos = bool(config.pipeline.get("auto_reuse_run_photos_dir", True))
    extra_photo_dirs: list[Path] = []
    if auto_tiktok:
        extra_photo_dirs.append(tiktok_run_dir)
    if auto_run_photos:
        extra_photo_dirs.append(out_dir / "photos")

    reuse_dir = Path(reuse_photos_dir) if reuse_photos_dir else None
    if reuse_dir is None:
        if auto_tiktok and _has_image_files(tiktok_run_dir):
            reuse_dir = tiktok_run_dir
        elif auto_run_photos and _has_image_files(out_dir / "photos"):
            reuse_dir = out_dir / "photos"
    photos = photo_agent.run(
        plan=plan,
        out_dir=out_dir,
        reuse_photos_dir=reuse_dir,
        extra_reuse_dirs=extra_photo_dirs,
    )

    # --- Stage 4: AI Model 4 (AI Video / Motion Generation per Scene) ---
    clip_dirs: list[Path] = []
    if auto_tiktok:
        clip_dirs.append(tiktok_run_dir)
    if (out_dir / "custom_clips").is_dir():
        clip_dirs.append(out_dir / "custom_clips")
    clips = video_agent.run(
        plan=plan,
        photos=photos,
        scene_durations=scene_durations,
        out_dir=out_dir,
        custom_clips_dir=clip_dirs or None,
    )

    # --- Stage 5b: AI Model 5 (Montage Assembly) ---
    manifest = montage_agent.assemble(
        run_id=run_id,
        audio_path=audio_path,
        timings=timings,
        total_duration_sec=total_dur,
        photos=photos,
        clips=clips,
        out_dir=out_dir,
    )

    # --- Stage 6: AI Model 6 (Multimodal Vision Reviewer & Feedback Loop) ---
    verdict = reviewer_agent.review(
        plan=plan, manifest=manifest, photos=photos, clips=clips, out_dir=out_dir, revision_round=0
    )

    for rev_round in range(1, effective_max_revisions + 1):
        if verdict.verdict == "PASS":
            break

        logger.warning(
            "[AI Model 6 - Reviewer] REVISION_REQUIRED -> Responsible: '%s' | Flagged scenes: %s | Diagnosis: %s",
            verdict.responsible_agent,
            verdict.flagged_scenes,
            verdict.error_diagnosis,
        )

        resp = verdict.responsible_agent
        if resp == "research":
            report = research_agent.run(
                run_id=run_id,
                niche=niche,
                out_dir=out_dir,
                revision_feedback=verdict.error_diagnosis,
                trend_topic=topic_override,
                auto_top_trend=auto_top_trend,
            )
            plan = script_agent.run(
                run_id=run_id, report=report, out_dir=out_dir, revision_feedback=verdict.error_diagnosis
            )
            audio_path, timings, total_dur, scene_durations = montage_agent.prepare_audio_and_scene_timings(
                plan, out_dir
            )
            photos = photo_agent.run(plan=plan, out_dir=out_dir)
            clips = video_agent.run(plan=plan, photos=photos, scene_durations=scene_durations, out_dir=out_dir)

        elif resp == "script":
            if report is None:
                logger.warning("Cannot auto-re-run script revision without ResearchReport; stopping loop.")
                break
            plan = script_agent.run(
                run_id=run_id, report=report, out_dir=out_dir, revision_feedback=verdict.error_diagnosis
            )
            audio_path, timings, total_dur, scene_durations = montage_agent.prepare_audio_and_scene_timings(
                plan, out_dir
            )
            photos = photo_agent.run(plan=plan, out_dir=out_dir)
            clips = video_agent.run(plan=plan, photos=photos, scene_durations=scene_durations, out_dir=out_dir)

        elif resp == "photo":
            target_scenes = list(verdict.flagged_scenes)
            if not target_scenes and verdict.scene_audits:
                target_scenes = [
                    a.scene_number for a in verdict.scene_audits if a.score < 7
                ]
                if not target_scenes:
                    lowest = sorted(verdict.scene_audits, key=lambda a: a.score)[:3]
                    target_scenes = [a.scene_number for a in lowest if a.scene_number]
            if not target_scenes:
                target_scenes = [s.scene_number for s in plan.scenes[:3]]

            corrected_prompts = {
                a.scene_number: (a.fixed_image_prompt or next(
                    (s.visual_ai_prompt for s in plan.scenes if s.scene_number == a.scene_number), ""
                ))
                for a in verdict.scene_audits
                if a.scene_number in target_scenes
            }
            for sn in target_scenes:
                if sn not in corrected_prompts:
                    corrected_prompts[sn] = next(
                        (s.visual_ai_prompt for s in plan.scenes if s.scene_number == sn), ""
                    )
            photos = photo_agent.regenerate_scenes(
                photos, corrected_prompts, out_dir, revision_round=rev_round
            )
            motion_overrides = {
                a.scene_number: a.fixed_motion_prompt
                for a in verdict.scene_audits
                if a.fixed_motion_prompt and a.scene_number in target_scenes
            }
            clips = video_agent.regenerate_clips(
                plan=plan,
                photos=photos,
                existing_clips=clips,
                flagged_scenes=set(target_scenes),
                scene_durations=scene_durations,
                motion_overrides=motion_overrides,
                out_dir=out_dir,
            )

        elif resp == "video":
            flagged = set(verdict.flagged_scenes) or {s.scene_number for s in plan.scenes}
            motion_overrides = {
                a.scene_number: a.fixed_motion_prompt
                for a in verdict.scene_audits
                if a.fixed_motion_prompt
            }
            clips = video_agent.regenerate_clips(
                plan=plan,
                photos=photos,
                existing_clips=clips,
                flagged_scenes=flagged,
                scene_durations=scene_durations,
                motion_overrides=motion_overrides,
                out_dir=out_dir,
            )

        # Re-run Montage (also handles resp == "montage") and re-review
        manifest = montage_agent.assemble(
            run_id=run_id,
            audio_path=audio_path,
            timings=timings,
            total_duration_sec=total_dur,
            photos=photos,
            clips=clips,
            out_dir=out_dir,
            montage_overrides=verdict.montage_fixes if resp == "montage" else None,
        )
        verdict = reviewer_agent.review(
            plan=plan, manifest=manifest, photos=photos, clips=clips, out_dir=out_dir, revision_round=rev_round
        )

    metadata_override = None
    if effective_metadata_json:
        metadata_override = json.loads(Path(effective_metadata_json).read_text(encoding="utf-8"))

    # --- Final Human-in-the-Loop Gate ---
    review_pkg = publisher_agent.prepare_review_package(
        run_id=run_id,
        plan=plan,
        manifest=manifest,
        out_dir=out_dir,
        reviewer_verdict=verdict,
        metadata_override=metadata_override,
    )

    frames_manifest_path = out_dir / "review_frames" / "frames_manifest.json"
    print("\n" + "=" * 76)
    print(f"STATUS            : {review_pkg.status}")
    print(f"RUN ID            : {review_pkg.run_id}")
    print(f"TOPIC             : {review_pkg.topic}")
    print(f"DURATION          : {review_pkg.duration_sec:.2f}s (QA Passed: {review_pkg.qa.passed})")
    print(f"AI REVIEW VERDICT : {verdict.verdict} (Rounds: {verdict.revision_round})")
    print(f"REVIEW DIAGNOSIS  : {verdict.error_diagnosis}")
    print(f"VIDEO             : {review_pkg.video_path}")
    print(f"FRAMES MANIFEST   : {frames_manifest_path}")
    print(f"REVIEW PACKAGE    : {review_pkg.review_markdown_path}")
    print(f"RUN FOLDER        : {out_dir}")
    print(f"TIKTOK FOLDER     : {tiktok_run_dir}")
    print("=" * 76)
    print(
        f"HUMAN-IN-THE-LOOP GATE ACTIVE: Inspect the video, then approve with:\n"
        f"  python run_pipeline.py --approve {review_pkg.run_id}"
    )

    if (
        not no_telegram_notify
        and review_pkg.status == "READY_FOR_REVIEW"
        and review_pkg.qa.passed
        and bool(config.telegram.get("enabled", True))
        and bool(config.telegram.get("auto_notify_on_ready", True))
    ):
        try:
            if str(config.root_dir) not in sys.path:
                sys.path.insert(0, str(config.root_dir))
            from scripts.verify_video_gate import verify_video
            from scripts.send_to_telegram import deliver_run_to_telegram

            floor_sec = float(config.video.get("hard_monetization_floor_sec", 61))
            gate = verify_video(Path(review_pkg.video_path), min_sec=floor_sec)
            if gate.get("passed"):
                tg_res = deliver_run_to_telegram(
                    out_dir=out_dir,
                    lang=plan.language,
                    config=config,
                    graceful=True,
                )
                logger.info("Telegram delivery status: %s", tg_res.get("ok") or tg_res.get("reason"))
        except Exception as exc:
            logger.warning("Telegram notification skipped safely: %s", exc)

    return 0 if review_pkg.qa.passed else 2


def run_approval(config_path: str, run_id: str, language_override: str | None = None) -> int:
    config = load_config(config_path)
    memory = MemoryStore(config.data_dir)
    publisher_agent = QAPublisherAgent(config, memory, skip_external_llm=True)
    lang_clean = language_override.strip().lower() if language_override else ""
    if lang_clean:
        mem_run_id = run_id if run_id.endswith(f"__{lang_clean}") else f"{run_id}__{lang_clean}"
        base_id = run_id[: -len(f"__{lang_clean}")] if run_id.endswith(f"__{lang_clean}") else run_id
        result = publisher_agent.approve_and_publish(
            run_id=mem_run_id,
            out_dir_override=config.output_dir / base_id / lang_clean,
            tiktok_bundle_override=config.tiktok_dir / base_id / lang_clean,
        )
    else:
        result = publisher_agent.approve_and_publish(run_id=run_id)
    print(f"Approved run {run_id}. TikTok export bundle result:")
    print(result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Antigravity 6-AI-Model TikTok Vertical Video Studio (HITL-gated)"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--niche", default=None, help="Override target niche for this run")
    parser.add_argument(
        "--topic",
        default=None,
        help="Pin a specific live trending TikTok topic for AI Model 1 & 2",
    )
    parser.add_argument(
        "--auto-top-trend",
        action="store_true",
        help="Automatically discover, rank, filter, and pick the #1 eligible trending TikTok topic",
    )
    parser.add_argument(
        "--discover-trend-only",
        action="store_true",
        help="Run Stage 1 trend discovery & ranking only, then exit cleanly before script/media",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id (useful for Antigravity hybrid pre-staged runs)",
    )
    parser.add_argument(
        "--script-json",
        default=None,
        metavar="FILE",
        help="Path to a pre-authored script JSON (skips external research/script LLM calls)",
    )
    parser.add_argument(
        "--metadata-json",
        default=None,
        metavar="FILE",
        help="Path to a pre-authored TikTok metadata JSON override",
    )
    parser.add_argument(
        "--skip-external-llm",
        action="store_true",
        help="Skip external Gemini API calls (rely on Antigravity sub-agents / local fallbacks)",
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=None,
        help="Override target scene count for this run (e.g. 10)",
    )
    parser.add_argument(
        "--reuse-photos-dir",
        default=None,
        metavar="DIR",
        help="Optional directory with pre-generated scene_XX_photo.jpg/png files",
    )
    parser.add_argument(
        "--approve",
        default=None,
        metavar="RUN_ID",
        help="Explicitly approve a READY_FOR_REVIEW run to create the TikTok export bundle",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Optional localized language code (e.g. en, ar, de, fr) using per-language subfolders",
    )
    parser.add_argument(
        "--revision-feedback",
        default=None,
        help="Optional human edit/revision feedback to re-render an existing run_id",
    )
    parser.add_argument(
        "--no-telegram-notify",
        action="store_true",
        help="Skip one-way end-of-pipeline Telegram notification (used by interactive bot worker)",
    )
    parser.add_argument(
        "--keywords",
        default=None,
        help="Comma-separated sourcing keywords for yt-dlp / web media harvesting",
    )
    args = parser.parse_args()

    if args.approve:
        return run_approval(args.config, args.approve, language_override=args.lang)
    return run_production_pipeline(
        config_path=args.config,
        niche_override=args.niche,
        reuse_photos_dir=args.reuse_photos_dir,
        topic_override=args.topic,
        run_id_override=args.run_id,
        script_json_path=args.script_json,
        metadata_json_path=args.metadata_json,
        skip_external_llm=args.skip_external_llm,
        scenes_override=args.scenes,
        auto_top_trend=args.auto_top_trend,
        discover_trend_only=args.discover_trend_only,
        language_override=args.lang,
        revision_feedback=args.revision_feedback,
        no_telegram_notify=args.no_telegram_notify,
        keywords_override=args.keywords,
    )


if __name__ == "__main__":
    sys.exit(main())
