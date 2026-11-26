from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import AppConfig
from src.models import (
    GeneratedPhoto,
    GeneratedVideoClip,
    MediaManifest,
    ReviewerVerdict,
    SceneAuditItem,
    ScriptPlan,
)
from src.providers.llm import build_llm_provider
from src.utils.ffmpeg_tools import resolve_ffmpeg_executable

logger = logging.getLogger(__name__)


class AIReviewerAgent:
    """AI Model 6: Multimodal Vision Reviewer inspecting final frames + audio/script & routing fixes."""

    def __init__(self, config: AppConfig, skip_external_llm: bool = False) -> None:
        self.config = config
        if skip_external_llm:
            self.vision_llm = None
        else:
            model_name = config.raw.get("models", {}).get("reviewer_model")
            self.vision_llm = build_llm_provider(config, model_override=model_name, temperature=0.2)

    def _extract_scene_frames(
        self, video_path: Path, clips: list[GeneratedVideoClip], out_dir: Path
    ) -> list[tuple[int, Path]]:
        frames_dir = out_dir / "review_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_exe = resolve_ffmpeg_executable()

        extracted: list[tuple[int, Path]] = []
        cursor = 0.0
        for clip in clips:
            mid_ts = round(cursor + clip.duration_sec * 0.5, 2)
            cursor += clip.duration_sec
            frame_path = frames_dir / f"scene_{clip.scene_number:02d}.jpg"
            subprocess.run(
                [
                    ffmpeg_exe, "-y", "-ss", f"{mid_ts:.2f}",
                    "-i", str(video_path), "-vframes", "1", "-q:v", "3", str(frame_path),
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if frame_path.exists():
                extracted.append((clip.scene_number, frame_path))
        return extracted

    def _vision_review(
        self,
        plan: ScriptPlan,
        manifest: MediaManifest,
        photos: list[GeneratedPhoto],
        frame_pairs: list[tuple[int, Path]],
        revision_round: int,
    ) -> ReviewerVerdict | None:
        if not self.vision_llm or not frame_pairs:
            return None

        photo_map = {p.scene_number: p for p in photos}
        scene_rows = []
        contents: list[Any] = []

        system_prompt = (
            "You are AI Model 6 (Strict TikTok Quality Director & Multimodal Visual Reviewer). "
            "You inspect extracted frames from the rendered 1080x1920 (9:16) video alongside each scene's spoken script. "
            "Reject ANY scene whose visual has ANY of these defects:\n"
            "1. Does NOT match the spoken text (off-topic objects, watermarks, or jarring mismatches).\n"
            "2. Displayed photo/foreground is too small or postage-stamped with excessive empty side margins inside the 1080x1920 frame.\n"
            "3. Stretched/distorted aspect ratio, wrong dimensions, cropped faces/subjects, or blank backgrounds.\n"
            "4. Unreadable, cropped, or colliding burned-in captions.\n"
            "Identify the EXACT responsible agent ('research', 'script', 'photo', 'video', 'montage', or 'none') "
            "and write explicit corrected image prompts for every failed scene so AI Model 3 can regenerate them."
        )

        thumb_cfg = self.config.raw.get("models", {}).get("reviewer_thumb_size", [360, 640])
        thumb_size = (int(thumb_cfg[0]), int(thumb_cfg[1])) if len(thumb_cfg) == 2 else (360, 640)

        # Resize frames slightly before sending so scene frames fit comfortably in one multimodal call
        pil_images: list[Image.Image] = []
        try:
            for scene_num, frame_path in frame_pairs:
                scene_obj = next((s for s in plan.scenes if s.scene_number == scene_num), None)
                if not scene_obj:
                    continue
                scene_rows.append(
                    {
                        "scene_number": scene_num,
                        "voiceover_text": scene_obj.voiceover_text,
                        "photo_provider": photo_map[scene_num].provider if scene_num in photo_map else "",
                        "original_prompt": scene_obj.visual_ai_prompt,
                    }
                )
                with Image.open(frame_path) as im:
                    thumb = im.convert("RGB").copy()
                    thumb.thumbnail(thumb_size)
                    pil_images.append(thumb)

            prompt_text = json.dumps(
                {
                    "topic": plan.topic,
                    "duration_sec": manifest.actual_duration_sec,
                    "revision_round": revision_round,
                    "scenes_in_order_of_attached_images": scene_rows,
                    "required_json_schema": {
                        "verdict": "PASS | REVISION_REQUIRED",
                        "responsible_agent": "none | research | script | photo | video | montage",
                        "error_diagnosis": "Concise diagnosis of any visual, script, or montage errors",
                        "flagged_scenes": [1, 2],
                        "scene_audits": [
                            {
                                "scene_number": 1,
                                "matches_script": True,
                                "score": 9,
                                "issue": "",
                                "fixed_image_prompt": "",
                                "fixed_motion_prompt": "zoom_in",
                            }
                        ],
                        "montage_fixes": {
                            "margin_v": 520,
                            "bed_volume": 0.14,
                            "caption_note": "",
                        },
                    },
                },
                ensure_ascii=False,
            )
            contents.append(prompt_text)
            contents.extend(pil_images)

            raw = self.vision_llm.generate_json(
                prompt=prompt_text, system_prompt=system_prompt, contents=contents
            )
        except Exception as exc:
            logger.error("AI Model 6 (Vision Reviewer) failed: %s", exc)
            return None
        finally:
            for im in pil_images:
                im.close()

        audits = [
            SceneAuditItem(
                scene_number=int(a.get("scene_number", 0)),
                matches_script=bool(a.get("matches_script", True)),
                score=int(a.get("score", 8)),
                issue=str(a.get("issue", "")),
                fixed_image_prompt=str(a.get("fixed_image_prompt", "")),
                fixed_motion_prompt=str(a.get("fixed_motion_prompt", "")),
            )
            for a in raw.get("scene_audits", [])
        ]
        flagged = [int(x) for x in raw.get("flagged_scenes", [])]
        for a in audits:
            if (not a.matches_script or a.score < 6) and a.scene_number not in flagged:
                flagged.append(a.scene_number)

        verdict_str = str(raw.get("verdict", "PASS")).upper()
        responsible = str(raw.get("responsible_agent", "none")).lower()
        if flagged and verdict_str == "PASS":
            verdict_str = "REVISION_REQUIRED"
            if responsible == "none":
                responsible = "photo"

        return ReviewerVerdict(
            verdict=verdict_str,
            responsible_agent=responsible,
            error_diagnosis=str(raw.get("error_diagnosis", "")),
            flagged_scenes=sorted(set(flagged)),
            scene_audits=audits,
            montage_fixes=dict(raw.get("montage_fixes", {})),
            revision_round=revision_round,
        )

    def _heuristic_fallback_review(
        self,
        plan: ScriptPlan,
        manifest: MediaManifest,
        photos: list[GeneratedPhoto],
        revision_round: int,
        frame_pairs: list[tuple[int, Path]] | None = None,
        clips: list[GeneratedVideoClip] | None = None,
    ) -> ReviewerVerdict:
        """Deterministic safety audit verifying dimensions, duration, and non-blank visual assets."""
        floor_sec = float(self.config.video.get("hard_monetization_floor_sec", 61))
        target_w = int(self.config.video.get("width", 1080))
        target_h = int(self.config.video.get("height", 1920))
        expected_res = f"{target_w}x{target_h}"
        live_video_scenes = {
            c.scene_number
            for c in (clips or [])
            if c.provider == "preexisting_custom_video"
        }

        frame_dim_issues: dict[int, str] = {}
        if frame_pairs:
            for sn, frame_path in frame_pairs:
                if frame_path.exists():
                    with Image.open(frame_path) as im:
                        fw, fh = im.size
                    if (fw, fh) != (target_w, target_h):
                        frame_dim_issues[sn] = f"Frame dimensions {fw}x{fh} do not match {expected_res}."

        flagged: list[int] = []
        audits: list[SceneAuditItem] = []

        for p in photos:
            has_live_broll = p.scene_number in live_video_scenes
            is_offline = p.provider == "offline_atmosphere" and not has_live_broll
            dim_err = frame_dim_issues.get(p.scene_number, "")
            has_issue = is_offline or bool(dim_err)
            if has_issue:
                flagged.append(p.scene_number)
            issue_msg = (
                "Offline fallback placeholder used instead of real footage/image."
                if is_offline
                else dim_err
            )
            audits.append(
                SceneAuditItem(
                    scene_number=p.scene_number,
                    matches_script=not has_issue,
                    score=3 if is_offline else (4 if dim_err else 8),
                    issue=issue_msg,
                    fixed_image_prompt=(
                        next((s.visual_ai_prompt for s in plan.scenes if s.scene_number == p.scene_number), "")
                    ),
                )
            )

        if manifest.resolution and manifest.resolution != expected_res:
            return ReviewerVerdict(
                verdict="REVISION_REQUIRED",
                responsible_agent="montage",
                error_diagnosis=f"Video resolution {manifest.resolution} does not match standard {expected_res}.",
                flagged_scenes=flagged,
                scene_audits=audits,
                revision_round=revision_round,
            )

        if manifest.actual_duration_sec < floor_sec:
            return ReviewerVerdict(
                verdict="REVISION_REQUIRED",
                responsible_agent="montage",
                error_diagnosis=f"Video duration {manifest.actual_duration_sec:.1f}s is below 61s threshold.",
                flagged_scenes=flagged,
                scene_audits=audits,
                revision_round=revision_round,
            )

        if flagged:
            return ReviewerVerdict(
                verdict="REVISION_REQUIRED",
                responsible_agent="photo",
                error_diagnosis=(
                    f"Scenes {flagged} failed asset/dimension checks; routing back to Photo/Video framing."
                ),
                flagged_scenes=flagged,
                scene_audits=audits,
                revision_round=revision_round,
            )

        return ReviewerVerdict(
            verdict="PASS",
            responsible_agent="none",
            error_diagnosis="All technical checks, 1080x1920 dimensions, and visual assets verified.",
            flagged_scenes=[],
            scene_audits=audits,
            revision_round=revision_round,
        )

    def _write_frames_manifest(
        self,
        plan: ScriptPlan,
        manifest: MediaManifest,
        photos: list[GeneratedPhoto],
        frame_pairs: list[tuple[int, Path]],
        out_dir: Path,
        revision_round: int,
    ) -> None:
        frames_dir = out_dir / "review_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        photo_map = {p.scene_number: p for p in photos}
        scene_map = {s.scene_number: s for s in plan.scenes}
        frames_payload = []
        for sn, frame_path in frame_pairs:
            scene_obj = scene_map.get(sn)
            frame_size: list[int] = [0, 0]
            if frame_path.exists():
                with Image.open(frame_path) as im:
                    frame_size = [im.width, im.height]
            frames_payload.append(
                {
                    "scene_number": sn,
                    "frame_path": str(frame_path),
                    "frame_size": frame_size,
                    "voiceover_text": scene_obj.voiceover_text if scene_obj else "",
                    "visual_ai_prompt": scene_obj.visual_ai_prompt if scene_obj else "",
                    "photo_provider": photo_map[sn].provider if sn in photo_map else "",
                }
            )
        (frames_dir / "frames_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": plan.run_id,
                    "topic": plan.topic,
                    "duration_sec": manifest.actual_duration_sec,
                    "revision_round": revision_round,
                    "frames": frames_payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def review(
        self,
        plan: ScriptPlan,
        manifest: MediaManifest,
        photos: list[GeneratedPhoto],
        clips: list[GeneratedVideoClip],
        out_dir: Path,
        revision_round: int,
    ) -> ReviewerVerdict:
        frame_pairs = self._extract_scene_frames(Path(manifest.final_video_path), clips, out_dir)
        self._write_frames_manifest(
            plan=plan,
            manifest=manifest,
            photos=photos,
            frame_pairs=frame_pairs,
            out_dir=out_dir,
            revision_round=revision_round,
        )
        verdict = self._vision_review(plan, manifest, photos, frame_pairs, revision_round)
        if verdict is None:
            verdict = self._heuristic_fallback_review(
                plan, manifest, photos, revision_round, frame_pairs=frame_pairs, clips=clips
            )

        (out_dir / f"reviewer_verdict_r{revision_round}.json").write_text(
            json.dumps(verdict.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return verdict
