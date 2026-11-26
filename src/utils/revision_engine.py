from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.providers.llm import build_llm_provider

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _normalize_digits(text: str) -> str:
    """Convert Arabic/Persian numerals in user feedback into ASCII digits."""
    return (text or "").translate(_ARABIC_DIGITS)


def _to_seconds(token: str) -> float:
    clean = token.strip()
    if ":" in clean:
        parts = [float(p) for p in clean.split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    return float(clean)


def parse_time_ranges(feedback: str) -> list[tuple[float, float]]:
    """Parse English and Arabic time ranges (e.g. '00:15 to 00:22', 'from 15 seconds to 21 seconds', 'من الثانية ١٥ الى ٢١')."""
    norm = _normalize_digits(feedback)
    ranges: list[tuple[float, float]] = []

    colon_pat = r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:to|and|&|until|till|-|–|إلى|الى|حتى|و)\s*(\d{1,2}:\d{2}(?::\d{2})?)"
    for m in re.finditer(colon_pat, norm, flags=re.IGNORECASE):
        t0 = _to_seconds(m.group(1))
        t1 = _to_seconds(m.group(2))
        if t1 > t0 >= 0.0:
            ranges.append((t0, t1))

    if not ranges:
        sec_pat = (
            r"(?:from\s+|between\s+|من\s+)?(?:the\s+|ثانية\s*|الثانية\s*)?"
            r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s|ثوانٍ|ثواني|ثانية)?\s*"
            r"(?:to|and|until|-|–|إلى|الى|حتى|و)\s*"
            r"(?:the\s+|ثانية\s*|الثانية\s*)?(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s|ثوانٍ|ثواني|ثانية)?"
        )
        for m in re.finditer(sec_pat, norm, flags=re.IGNORECASE):
            full_seg = m.group(0).lower()
            has_time_unit = (
                any(w in full_seg for w in ("second", "sec", "ثان", "ثوان"))
                or bool(re.search(r"\d+\s*(?:s\b|-|to\b|إلى|الى)", full_seg))
            )
            if not has_time_unit:
                continue
            t0 = _to_seconds(m.group(1))
            t1 = _to_seconds(m.group(2))
            if t1 > t0 >= 0.0:
                ranges.append((t0, t1))

    if not ranges:
        single_pat = r"(?:starting\s+at|from|ابتداءا?\s*من|عند)\s*(?:الثانية|ثانية|second|sec)?\s*(\d{1,2}:\d{2}|\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s|ثوانٍ|ثواني|ثانية)?"
        for m in re.finditer(single_pat, norm, flags=re.IGNORECASE):
            t0 = _to_seconds(m.group(1))
            if t0 >= 0.0:
                ranges.append((t0, t0 + 7.0))

    return ranges


def parse_scene_numbers(feedback: str) -> list[int]:
    """Parse explicit scene/clip numbers from English and Arabic text."""
    norm = _normalize_digits(feedback)
    nums: list[int] = []
    pat = r"(?:scenes?|clips?|المشهد|مشهد|اللقطة|لقطة)\s*#?\s*(\d+(?:\s*(?:,|and|&|و)\s*\d+)*)"
    for m in re.finditer(pat, norm, flags=re.IGNORECASE):
        for n_str in re.findall(r"\d+", m.group(1)):
            val = int(n_str)
            if 1 <= val <= 30 and val not in nums:
                nums.append(val)
    return nums


def _find_matching_scenes(
    feedback: str,
    timings_path: Path,
) -> list[int]:
    targets = set(parse_scene_numbers(feedback))
    time_ranges = parse_time_ranges(feedback)
    if time_ranges and timings_path.exists():
        try:
            timings = json.loads(timings_path.read_text(encoding="utf-8"))
            for row in timings:
                sn = int(row.get("scene_number", 0))
                s0 = float(row.get("start_sec", 0.0))
                s1 = float(row.get("end_sec", 0.0))
                for t0, t1 in time_ranges:
                    overlap = min(s1, t1) - max(s0, t0)
                    if overlap >= 1.0 and sn > 0:
                        targets.add(sn)
        except Exception:
            pass
    return sorted(targets)


def _delete_and_shift_scene_files(out_dir: Path, keep_scene_numbers: list[int], total_old: int) -> None:
    """Shift kept staged scene clips/photos sequentially into 1..M and clear removed/stale artifacts."""
    old_to_new = {old_sn: new_idx for new_idx, old_sn in enumerate(keep_scene_numbers, start=1)}
    photos_dir = out_dir / "photos"

    # Rename kept files to temporary names first to avoid collisions
    temp_moves: list[tuple[Path, Path]] = []
    for old_sn in range(1, total_old + 1):
        new_sn = old_to_new.get(old_sn)
        # Top-level scene_XX.mp4 / scene_XX_clip.mp4 and scene_XX.* photos
        candidates = [
            (out_dir / f"scene_{old_sn:02d}.mp4", out_dir / f"_shift_tmp_{new_sn:02d}.mp4" if new_sn else None, out_dir / f"scene_{new_sn:02d}.mp4" if new_sn else None),
            (out_dir / f"scene_{old_sn:02d}_clip.mp4", out_dir / f"_shift_tmp_{new_sn:02d}_clip.mp4" if new_sn else None, out_dir / f"scene_{new_sn:02d}_clip.mp4" if new_sn else None),
        ]
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidates.append(
                (out_dir / f"scene_{old_sn:02d}{ext}", out_dir / f"_shift_tmp_{new_sn:02d}{ext}" if new_sn else None, out_dir / f"scene_{new_sn:02d}{ext}" if new_sn else None)
            )
            if photos_dir.is_dir():
                candidates.append(
                    (photos_dir / f"scene_{old_sn:02d}_photo{ext}", photos_dir / f"_shift_tmp_{new_sn:02d}_photo{ext}" if new_sn else None, photos_dir / f"scene_{new_sn:02d}_photo{ext}" if new_sn else None)
                )
                candidates.append(
                    (photos_dir / f"scene_{old_sn:02d}{ext}", photos_dir / f"_shift_tmp_{new_sn:02d}{ext}" if new_sn else None, photos_dir / f"scene_{new_sn:02d}{ext}" if new_sn else None)
                )

        for src, tmp_dst, final_dst in candidates:
            if not src.exists():
                continue
            if new_sn is None or tmp_dst is None or final_dst is None:
                src.unlink(missing_ok=True)
            else:
                src.rename(tmp_dst)
                temp_moves.append((tmp_dst, final_dst))

    for tmp_dst, final_dst in temp_moves:
        if tmp_dst.exists():
            final_dst.unlink(missing_ok=True)
            tmp_dst.rename(final_dst)

    for folder in (out_dir / "clips", out_dir / "hud"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
    for stale in ("voiceover.mp3", "scene_timings.json", "word_times.json"):
        (out_dir / stale).unlink(missing_ok=True)


def apply_revision_feedback(
    feedback: str,
    out_dir: Path,
    config: AppConfig,
) -> dict[str, Any]:
    """Parse and apply real revision edits before re-rendering `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    rev_log_path = out_dir / "revision_feedback_log.json"
    rev_entries = json.loads(rev_log_path.read_text(encoding="utf-8")) if rev_log_path.exists() else []
    rev_entries.append({"timestamp": now_iso, "feedback": feedback})
    rev_log_path.write_text(json.dumps(rev_entries, indent=2, ensure_ascii=False), encoding="utf-8")

    norm = _normalize_digits(feedback)
    fb_low = norm.lower()
    applied_actions: list[str] = []
    warnings: list[str] = []

    # 1. Strict English monolingual voice/language enforcement
    if any(
        w in fb_low
        for w in (
            "german", "english", "language", "accent", "voice",
            "ألمانية", "المانية", "الماني", "ألماني", "انكليزي", "إنكليزي",
            "إنجليزي", "الانجليزية", "الانكليزية", "الإنجليزية", "اللغة", "الصوت",
        )
    ):
        config.raw.setdefault("project", {})["language"] = "en"
        config.raw.setdefault("tts", {}).setdefault("voices", {})["en"] = "en-US-AndrewNeural"
        (out_dir / "voiceover.mp3").unlink(missing_ok=True)
        applied_actions.append(
            "تم تثبيت الصوت الإنجليزي أحادي اللغة (en-US-AndrewNeural) وإعادة توليد التعليق الصوتي بالكامل"
        )

    # 2. Voiceover rate / speed adjustment
    rate_match = re.search(r"([+-]\d+%)", fb_low)
    if rate_match:
        new_rate = rate_match.group(1)
        config.raw.setdefault("tts", {})["rate"] = new_rate
        (out_dir / "voiceover.mp3").unlink(missing_ok=True)
        applied_actions.append(f"تم تعديل سرعة التعليق الصوتي إلى {new_rate}")
    elif any(w in fb_low for w in ("faster", "speed up", "أسرع", "اسرع")):
        config.raw.setdefault("tts", {})["rate"] = "+10%"
        (out_dir / "voiceover.mp3").unlink(missing_ok=True)
        applied_actions.append("تم تسريع التعليق الصوتي إلى +10%")
    elif any(w in fb_low for w in ("slower", "slow down", "أبطأ", "ابطأ")):
        config.raw.setdefault("tts", {})["rate"] = "-2%"
        (out_dir / "voiceover.mp3").unlink(missing_ok=True)
        applied_actions.append("تم إبطاء التعليق الصوتي إلى -2%")

    # 3. Delete scenes or time ranges and remerge cleanly
    is_delete_req = any(
        w in fb_low
        for w in ("delete", "remove", "cut", "drop", "trim", "remerge", "احذف", "حذف", "قص", "إزالة", "ازالة")
    )
    is_replace_req = any(
        w in fb_low
        for w in ("replace", "swap", "photo", "image", "visual", "clip", "استبدل", "تغيير", "غير", "صور", "صورة", "لقطة", "لقطات")
    )
    matched_scenes = _find_matching_scenes(norm, out_dir / "scene_timings.json")
    floor_sec = float(config.video.get("hard_monetization_floor_sec", 61))

    script_path = out_dir / "script_plan.json"
    handled_delete = False
    if is_delete_req and matched_scenes and script_path.exists():
        plan_data = json.loads(script_path.read_text(encoding="utf-8"))
        scenes_list = list(plan_data.get("scenes") or [])
        to_delete = set(matched_scenes)
        kept = [s for s in scenes_list if int(s.get("scene_number", 0)) not in to_delete]
        remaining_dur = sum(float(s.get("estimated_duration_sec") or 6.5) for s in kept)

        if len(kept) >= 5 and remaining_dur >= floor_sec:
            keep_nums = [int(s["scene_number"]) for s in kept]
            for new_idx, sc in enumerate(kept, start=1):
                sc["scene_number"] = new_idx
            full_vo = " ".join(str(s.get("voiceover_text") or "").strip() for s in kept if s.get("voiceover_text"))
            plan_data["scenes"] = kept
            plan_data["full_voiceover_text"] = full_vo
            plan_data["word_count"] = len(full_vo.split())
            plan_data["opening_hook_3s"] = str(kept[0].get("voiceover_text") or "")
            plan_data["closing_cta"] = str(kept[-1].get("voiceover_text") or "")
            plan_data["total_estimated_duration_sec"] = round(remaining_dur, 2)
            script_path.write_text(json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8")

            _delete_and_shift_scene_files(out_dir, keep_nums, len(scenes_list))
            applied_actions.append(
                f"تم حذف المشاهد {sorted(to_delete)} وإعادة ترقيم ودمج الفيديو ({round(remaining_dur, 1)}s)"
            )
            handled_delete = True
        else:
            warnings.append(
                f"تعذر حذف المشاهد {sorted(to_delete)} لأن المدة المتبقية ({remaining_dur:.1f}s) تقل عن حد الربح ({floor_sec:.0f}s)؛ تم استبدال لقطاتها بدل الحذف."
            )
            for sn in sorted(to_delete):
                for p in (
                    out_dir / f"scene_{sn:02d}.mp4",
                    out_dir / f"scene_{sn:02d}_clip.mp4",
                    out_dir / "clips" / f"clip_{sn:02d}.mp4",
                ):
                    p.unlink(missing_ok=True)
                photos_dir = out_dir / "photos"
                if photos_dir.is_dir():
                    for ph in photos_dir.glob(f"scene_{sn:02d}*"):
                        ph.unlink(missing_ok=True)
            applied_actions.append(f"تم حذف وإعادة جلب لقطات المشاهد {sorted(to_delete)} للحفاظ على مدة > {floor_sec:.0f} ثانية")
            handled_delete = True

    # 4. Replace visuals/clips when requested explicitly (and not already deleted)
    if is_replace_req and not handled_delete:
        photos_dir = out_dir / "photos"
        clips_dir = out_dir / "clips"
        if matched_scenes:
            for sn in matched_scenes:
                (out_dir / f"scene_{sn:02d}.mp4").unlink(missing_ok=True)
                (out_dir / f"scene_{sn:02d}_clip.mp4").unlink(missing_ok=True)
                (clips_dir / f"clip_{sn:02d}.mp4").unlink(missing_ok=True)
                if photos_dir.is_dir():
                    for ph in photos_dir.glob(f"scene_{sn:02d}*"):
                        ph.unlink(missing_ok=True)
            applied_actions.append(f"تم مسح وإعادة جلب لقطات المشاهد {matched_scenes}")
        else:
            if photos_dir.is_dir():
                for ph in photos_dir.glob("scene_*_photo.*"):
                    ph.unlink(missing_ok=True)
            if clips_dir.is_dir():
                for cl in clips_dir.glob("clip_*.mp4"):
                    cl.unlink(missing_ok=True)
            if any(w in fb_low for w in ("clip", "video", "لقطة", "لقطات", "فيديو")):
                for sc in out_dir.glob("scene_*.mp4"):
                    sc.unlink(missing_ok=True)
            applied_actions.append("تم تحديث وإعادة تجهيز اللقطات والصور المرئية")

    result = {
        "timestamp": now_iso,
        "feedback": feedback,
        "applied_actions": applied_actions,
        "warnings": warnings,
        "matched": bool(applied_actions),
    }
    (out_dir / "revision_applied.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def compute_file_sha256(path: Path) -> str:
    """Return hex SHA-256 checksum of `path`, or empty string if missing."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def try_local_ai_revision_diagnosis(
    job: dict[str, Any], out_dir: Path, config: AppConfig
) -> dict[str, Any] | None:
    """Attempt $0 local Ollama diagnosis & surgical script/scene edit before human-agent escalation."""
    llm = build_llm_provider(config, temperature=0.2)
    if llm is None:
        return None

    script_path = out_dir / "script_plan.json"
    timings_path = out_dir / "scene_timings.json"
    review_path = out_dir / "review_package.json"

    plan_data = json.loads(script_path.read_text(encoding="utf-8")) if script_path.exists() else {}
    timings_data = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else []
    review_data = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}

    system_prompt = (
        "You are an AI Video QA Editor diagnosing why automated edit rounds failed to satisfy the creator. "
        "Return strict JSON matching the requested schema. Set escalate_to_antigravity=true if visual inspection, "
        "re-harvesting new footage, or code/audio fixes beyond text/scene deletion are required."
    )
    payload = {
        "trend_title": job.get("trend_title", ""),
        "edit_history": job.get("edit_history", []),
        "scenes": plan_data.get("scenes", []),
        "scene_timings": timings_data,
        "qa_status": review_data.get("status", ""),
        "qa_issues": review_data.get("qa", {}).get("issues", []),
        "required_json_schema": {
            "diagnosis_ar": "concise Arabic diagnosis",
            "can_auto_fix": False,
            "escalate_to_antigravity": True,
            "delete_scenes": [],
            "replace_clips_scenes": [],
            "updated_scenes": [{"scene_number": 1, "voiceover_text": "string"}],
            "tts_rate": "",
        },
    }
    try:
        raw = llm.generate_json(prompt=json.dumps(payload, ensure_ascii=False), system_prompt=system_prompt)
    except Exception:
        return None

    diag_ar = str(raw.get("diagnosis_ar") or "").strip()
    if not bool(raw.get("can_auto_fix")) or bool(raw.get("escalate_to_antigravity")):
        return {"fixed": False, "diagnosis_ar": diag_ar}

    applied: list[str] = []
    floor_sec = float(config.video.get("hard_monetization_floor_sec", 61))
    scenes_list = list(plan_data.get("scenes") or [])

    upd_map = {
        int(u["scene_number"]): str(u.get("voiceover_text") or "").strip()
        for u in (raw.get("updated_scenes") or [])
        if isinstance(u, dict) and u.get("scene_number") and str(u.get("voiceover_text") or "").strip()
    }
    if upd_map and scenes_list:
        for sc in scenes_list:
            sn = int(sc.get("scene_number", 0))
            if sn in upd_map:
                sc["voiceover_text"] = upd_map[sn]
        applied.append(f"تحديث نص المشاهد {sorted(upd_map)}")

    del_scenes = sorted({int(x) for x in (raw.get("delete_scenes") or []) if isinstance(x, int) or str(x).isdigit()})
    if del_scenes and scenes_list:
        kept = [s for s in scenes_list if int(s.get("scene_number", 0)) not in set(del_scenes)]
        rem_dur = sum(float(s.get("estimated_duration_sec") or 6.5) for s in kept)
        if len(kept) >= 5 and rem_dur >= floor_sec:
            keep_nums = [int(s["scene_number"]) for s in kept]
            old_total = len(scenes_list)
            for new_idx, sc in enumerate(kept, start=1):
                sc["scene_number"] = new_idx
            scenes_list = kept
            _delete_and_shift_scene_files(out_dir, keep_nums, old_total)
            applied.append(f"حذف المشاهد {del_scenes}")

    if scenes_list and script_path.exists():
        full_vo = " ".join(str(s.get("voiceover_text") or "").strip() for s in scenes_list if s.get("voiceover_text"))
        plan_data["scenes"] = scenes_list
        plan_data["full_voiceover_text"] = full_vo
        plan_data["word_count"] = len(full_vo.split())
        plan_data["opening_hook_3s"] = str(scenes_list[0].get("voiceover_text") or "")
        plan_data["closing_cta"] = str(scenes_list[-1].get("voiceover_text") or "")
        plan_data["total_estimated_duration_sec"] = round(
            sum(float(s.get("estimated_duration_sec") or 6.5) for s in scenes_list), 2
        )
        script_path.write_text(json.dumps(plan_data, indent=2, ensure_ascii=False), encoding="utf-8")

    rep_scenes = sorted({int(x) for x in (raw.get("replace_clips_scenes") or []) if str(x).isdigit()})
    for sn in rep_scenes:
        (out_dir / f"scene_{sn:02d}.mp4").unlink(missing_ok=True)
        (out_dir / f"scene_{sn:02d}_clip.mp4").unlink(missing_ok=True)
        (out_dir / "clips" / f"clip_{sn:02d}.mp4").unlink(missing_ok=True)
        photos_dir = out_dir / "photos"
        if photos_dir.is_dir():
            for ph in photos_dir.glob(f"scene_{sn:02d}*"):
                ph.unlink(missing_ok=True)
    if rep_scenes:
        applied.append(f"تحديث لقطات المشاهد {rep_scenes}")

    tts_rate = str(raw.get("tts_rate") or "").strip()
    if re.fullmatch(r"[+-]\d+%", tts_rate):
        config.raw.setdefault("tts", {})["rate"] = tts_rate
        applied.append(f"سرعة الصوت {tts_rate}")

    (out_dir / "voiceover.mp3").unlink(missing_ok=True)
    if not applied:
        return {"fixed": False, "diagnosis_ar": diag_ar}
    return {"fixed": True, "diagnosis_ar": diag_ar, "applied_actions": applied}


def create_ai_escalation_ticket(
    job: dict[str, Any],
    out_dir: Path,
    queue_path: Path,
    diagnosis_ar: str = "",
) -> dict[str, Any]:
    """Create/upsert an escalation ticket for Antigravity review after >3 edit rounds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    run_id = str(job.get("run_id") or out_dir.name)
    rev_round = int(job.get("revision_round") or 4)
    ticket_md = out_dir / "AI_ESCALATION_TICKET.md"

    ticket: dict[str, Any] = {
        "ticket_id": f"esc_{run_id}_r{rev_round}",
        "status": "OPEN",
        "created_at": now_iso,
        "job_id": str(job.get("job_id") or ""),
        "run_id": run_id,
        "trend_title": str(job.get("trend_title") or ""),
        "category": str(job.get("category") or ""),
        "category_ar": str(job.get("category_ar") or ""),
        "revision_round": rev_round,
        "edit_history": list(job.get("edit_history") or []),
        "latest_feedback": str((job.get("edit_history") or [""])[-1]),
        "local_ai_diagnosis_ar": diagnosis_ar,
        "run_dir": str(out_dir),
        "script_plan_path": str(out_dir / "script_plan.json"),
        "final_video_path": str(out_dir / "final_9x16.mp4"),
        "review_frames_dir": str(out_dir / "review_frames"),
        "ticket_markdown_path": str(ticket_md),
    }

    queue_data: dict[str, Any] = {"updated_at": now_iso, "tickets": []}
    if queue_path.exists():
        try:
            loaded = json.loads(queue_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("tickets"), list):
                queue_data = loaded
        except Exception:
            pass

    existing_tickets = [
        t for t in queue_data.get("tickets", [])
        if not (t.get("run_id") == run_id and str(t.get("status", "")).upper() == "OPEN")
    ]
    existing_tickets.append(ticket)
    queue_data["updated_at"] = now_iso
    queue_data["tickets"] = existing_tickets
    queue_path.write_text(json.dumps(queue_data, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# AI Escalation Ticket ({ticket['ticket_id']})",
        "",
        f"- **Status:** `{ticket['status']}`",
        f"- **Run ID:** `{run_id}`",
        f"- **Trend:** {ticket['trend_title']}",
        f"- **Edit Rounds:** `{rev_round}`",
        f"- **Created At:** `{now_iso}`",
        f"- **Video:** `{ticket['final_video_path']}`",
        f"- **Script Plan:** `{ticket['script_plan_path']}`",
        f"- **Review Frames:** `{ticket['review_frames_dir']}`",
        "",
        "## Edit Feedback History",
    ]
    for i, msg in enumerate(ticket["edit_history"], start=1):
        md_lines.append(f"{i}. {msg}")
    if diagnosis_ar:
        md_lines.extend(["", "## Local AI Diagnosis", diagnosis_ar])
    md_lines.extend(
        [
            "",
            "## Next Action (Antigravity)",
            f"1. Inspect frames & video in `{out_dir}` and apply fixes.",
            f"2. Re-render and deliver fixed preview: `python scripts/telegram_bot.py --deliver-review-run {run_id}`",
        ]
    )
    ticket_md.write_text("\n".join(md_lines), encoding="utf-8")
    return ticket


def resolve_ai_escalation_ticket(run_id: str, queue_path: Path) -> None:
    """Mark any OPEN escalation ticket for `run_id` as RESOLVED."""
    if not queue_path.exists():
        return
    try:
        queue_data = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return
    tickets = queue_data.get("tickets")
    if not isinstance(tickets, list):
        return
    changed = False
    now_iso = datetime.now(timezone.utc).isoformat()
    for t in tickets:
        if t.get("run_id") == run_id and str(t.get("status", "")).upper() == "OPEN":
            t["status"] = "RESOLVED"
            t["resolved_at"] = now_iso
            changed = True
    if changed:
        queue_data["updated_at"] = now_iso
        queue_path.write_text(json.dumps(queue_data, indent=2, ensure_ascii=False), encoding="utf-8")

