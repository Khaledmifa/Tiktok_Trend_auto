from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from src.config import AppConfig
from src.models import GeneratedPhoto, GeneratedVideoClip, ScriptPlan, ScriptScene
from src.providers.editorial_hud import render_scene_hud_overlay
from src.utils.ffmpeg_tools import detect_h264_encoder, probe_media_duration, resolve_ffmpeg_executable

logger = logging.getLogger(__name__)


class VideoGeneratorAgent:
    """AI Model 4: Turns live B-roll clips or photos into normalized 9:16 vertical motion clips."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        models_cfg = config.raw.get("models", {})
        self.video_mode = models_cfg.get("video_mode", "cinematic_i2v_motion")

    @staticmethod
    def _probe_video_dimensions(ffmpeg_exe: str, media_path: Path) -> tuple[int, int]:
        proc = subprocess.run(
            [ffmpeg_exe, "-i", str(media_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stderr.splitlines():
            if "Video:" in line:
                match = re.search(r"(\d{2,5})x(\d{2,5})", line)
                if match:
                    return int(match.group(1)), int(match.group(2))
        return 0, 0

    def _encode_attempts(self, ffmpeg_exe: str) -> list[list[str]]:
        clip_crf = int(self.config.video.get("clip_crf", 14))
        prefer_nvenc = str(self.config.video.get("preferred_encoder", "h264_nvenc")).lower() == "h264_nvenc"
        detected = detect_h264_encoder(ffmpeg_exe, prefer_nvenc=prefer_nvenc)
        attempts: list[list[str]] = []
        if detected == "h264_nvenc":
            attempts.append(
                [
                    "-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr",
                    "-cq", str(clip_crf), "-b:v", "0", "-pix_fmt", "yuv420p",
                ]
            )
        attempts.append(
            [
                "-c:v", "libx264", "-preset", "fast", "-crf", str(clip_crf),
                "-pix_fmt", "yuv420p",
            ]
        )
        return attempts

    def _normalize_video_clip(
        self,
        raw_mp4: Path,
        clip_path: Path,
        duration_sec: float,
        layout_mode: str = "wide_panel",
        hud_png: Path | None = None,
        start_offset_sec: float = 0.0,
        slowmo_factor: float = 1.0,
    ) -> None:
        ffmpeg_exe = resolve_ffmpeg_executable()
        w = int(self.config.video.get("width", 1080))
        h = int(self.config.video.get("height", 1920))
        fps = int(self.config.video.get("fps", 30))
        scale_factor = max(w, 1) / 1080.0
        hud_cfg = self.config.editorial_hud
        wide_h = int(round(int(hud_cfg.get("wide_panel_height", 608)) * scale_factor))
        wide_y = int(round(int(hud_cfg.get("wide_panel_y", 475)) * scale_factor))
        blur_radius = max(12, int(round(24 * scale_factor)))
        grade_enabled = bool(self.config.video.get("cinematic_color_grade", True))
        grade_filter = (
            ",eq=contrast=1.06:saturation=1.08:brightness=-0.01,vignette=PI/5"
            if grade_enabled
            else ""
        )
        src_w, src_h = self._probe_video_dimensions(ffmpeg_exe, raw_mp4)
        raw_dur = probe_media_duration(ffmpeg_exe, raw_mp4)

        use_slowmo = (
            0.70 <= slowmo_factor < 0.99
            and raw_dur > 0
            and (raw_dur - start_offset_sec) >= (duration_sec * slowmo_factor + 0.15)
        )
        pts_prefix = f"setpts=PTS/{slowmo_factor:.3f}," if use_slowmo else ""

        effective_mode = layout_mode
        if src_w <= src_h and src_w > 0:
            effective_mode = "fullscreen_portrait"
        elif effective_mode == "fullscreen_portrait" and src_w < 1280:
            # Avoid heavy upscaling blur if source is low-res landscape
            effective_mode = "wide_panel"

        input_args = ["-y"]
        if start_offset_sec > 0.05 and raw_dur > start_offset_sec + 1.0:
            input_args.extend(["-ss", f"{start_offset_sec:.3f}"])
        else:
            input_args.extend(["-stream_loop", "-1"])
        input_args.extend(["-i", str(raw_mp4)])
        if hud_png and hud_png.exists():
            input_args.extend(["-i", str(hud_png)])

        if effective_mode != "fullscreen_portrait" and src_w > src_h and src_w > 0:
            if hud_png and hud_png.exists():
                fg_scale = (
                    f"scale={w}:{wide_h}:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={w}:{wide_h},unsharp=5:5:0.35:5:5:0.0"
                )
                overlay_expr = f"(W-w)/2:{wide_y}"
            else:
                fg_w = max(w - int(round(20 * scale_factor)), 640)
                y_pos = max(int(round(140 * scale_factor)), int(round(h * 0.39)))
                fg_scale = f"scale={fg_w}:-2:flags=lanczos,unsharp=5:5:0.35:5:5:0.0"
                overlay_expr = f"(W-w)/2:max({int(round(140 * scale_factor))}\\,round({y_pos}-h/2))"
            filter_parts = [
                f"[0:v]{pts_prefix}scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},"
                f"boxblur={blur_radius}:4,eq=brightness=-0.35:saturation=0.9[bg]",
                f"[0:v]{pts_prefix}{fg_scale}[fg]",
                f"[bg][fg]overlay={overlay_expr},fps={fps}{grade_filter},setsar=1[base]",
            ]
        else:
            filter_parts = [
                f"[0:v]{pts_prefix}scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={w}:{h},unsharp=5:5:0.35:5:5:0.0,fps={fps}{grade_filter},setsar=1[base]"
            ]

        if hud_png and hud_png.exists():
            filter_parts.append("[base][1:v]overlay=0:0:format=auto,format=yuv420p[vout]")
        else:
            filter_parts.append("[base]format=yuv420p[vout]")

        last_err = ""
        for v_args in self._encode_attempts(ffmpeg_exe):
            cmd = [
                ffmpeg_exe,
                *input_args,
                "-t",
                f"{duration_sec:.3f}",
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[vout]",
                *v_args,
                "-an",
                str(clip_path),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode == 0:
                return
            last_err = proc.stderr[-600:]
        raise RuntimeError(f"Video clip normalization failed: {last_err}")

    def _render_ai_motion_clip(
        self,
        photo: GeneratedPhoto,
        motion_style: str,
        duration_sec: float,
        clip_path: Path,
        hud_png: Path | None = None,
    ) -> GeneratedVideoClip:
        ffmpeg_exe = resolve_ffmpeg_executable()
        w = int(self.config.video.get("width", 1080))
        h = int(self.config.video.get("height", 1920))
        fps = int(self.config.video.get("fps", 30))
        zoom_end = float(self.config.video.get("ken_burns_zoom_end", 1.12))
        grade_enabled = bool(self.config.video.get("cinematic_color_grade", True))
        grade_filter = (
            ",eq=contrast=1.08:saturation=1.10:brightness=-0.02,vignette=PI/5"
            if grade_enabled
            else ""
        )

        frames = max(int(round(duration_sec * fps)), int(fps * 0.8))
        zoom_step = max((zoom_end - 1.0) / max(frames, 1), 0.0006)
        ms = motion_style.lower()

        if "out" in ms:
            zp = (
                f"zoompan=z='if(eq(on,1),{zoom_end:.2f},max(zoom-{zoom_step:.6f},1.0))':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            )
        elif "left" in ms:
            zp = (
                f"zoompan=z='1.08':x='(iw-iw/zoom)*(1 - on/{frames})':y='ih/2-(ih/zoom/2)'"
            )
        elif "right" in ms:
            zp = (
                f"zoompan=z='1.08':x='(iw-iw/zoom)*(on/{frames})':y='ih/2-(ih/zoom/2)'"
            )
        elif "whip" in ms:
            zp = (
                f"zoompan=z='min(1.0 + 0.16*sin(min(on/12,1)*PI/2), 1.16)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            )
        else:
            zp = (
                f"zoompan=z='min(zoom+{zoom_step:.6f},{zoom_end:.2f})':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            )

        canvas_w = int(round(w * 1.25)) if w >= 2160 else (w * 2)
        canvas_h = int(round(h * 1.25)) if h >= 3840 else (h * 2)
        base_vf = (
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={canvas_w}:{canvas_h},{zp}:d={frames}:s={w}x{h}:fps={fps}"
            f"{grade_filter},setsar=1"
        )
        last_err = ""
        for v_args in self._encode_attempts(ffmpeg_exe):
            if hud_png and hud_png.exists():
                cmd = [
                    ffmpeg_exe, "-y", "-i", photo.image_path, "-i", str(hud_png),
                    "-filter_complex", f"[0:v]{base_vf}[base];[base][1:v]overlay=0:0:format=auto,format=yuv420p[vout]",
                    "-map", "[vout]", "-t", f"{duration_sec:.3f}",
                    *v_args, "-an", str(clip_path),
                ]
            else:
                cmd = [
                    ffmpeg_exe, "-y", "-i", photo.image_path,
                    "-vf", f"{base_vf},format=yuv420p", "-t", f"{duration_sec:.3f}",
                    *v_args, "-an", str(clip_path),
                ]
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if proc.returncode == 0:
                last_err = ""
                break
            last_err = proc.stderr[-800:]
        if last_err:
            raise RuntimeError(f"FFmpeg motion render failed for scene {photo.scene_number}: {last_err}")

        return GeneratedVideoClip(
            scene_number=photo.scene_number,
            clip_path=str(clip_path),
            duration_sec=duration_sec,
            provider="cinematic_i2v_motion",
            motion_style=motion_style,
        )

    def _find_custom_scene_sources(
        self, scene_num: int, custom_clips_dir: Path | list[Path] | None
    ) -> list[Path]:
        if not custom_clips_dir:
            return []
        dirs = [custom_clips_dir] if isinstance(custom_clips_dir, Path) else list(custom_clips_dir)
        for clip_dir in dirs:
            if not clip_dir or not clip_dir.is_dir():
                continue
            part_a = clip_dir / f"scene_{scene_num:02d}_a.mp4"
            part_b = clip_dir / f"scene_{scene_num:02d}_b.mp4"
            if (
                part_a.exists()
                and part_a.stat().st_size > 20_000
                and part_b.exists()
                and part_b.stat().st_size > 20_000
            ):
                return [part_a, part_b]
            for name in (f"scene_{scene_num:02d}_clip.mp4", f"scene_{scene_num:02d}.mp4"):
                candidate = clip_dir / name
                if candidate.exists() and candidate.stat().st_size > 20_000:
                    return [candidate]
        return []

    def _concat_subclips(self, sub_paths: list[Path], out_path: Path) -> None:
        ffmpeg_exe = resolve_ffmpeg_executable()
        concat_txt = out_path.with_suffix(".concat.txt")
        concat_txt.write_text(
            "\n".join(f"file '{p.name}'" for p in sub_paths),
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                [
                    ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_txt.name,
                    "-c", "copy", out_path.name,
                ],
                cwd=out_path.parent,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Sub-clip concat failed: {proc.stderr[-500:]}")
        finally:
            concat_txt.unlink(missing_ok=True)
            for p in sub_paths:
                p.unlink(missing_ok=True)

    def _try_custom_scene_clip(
        self,
        scene_num: int,
        duration_sec: float,
        clip_path: Path,
        custom_clips_dir: Path | list[Path] | None,
        layout_mode: str = "wide_panel",
        hud_png: Path | None = None,
    ) -> GeneratedVideoClip | None:
        sources = self._find_custom_scene_sources(scene_num, custom_clips_dir)
        if not sources:
            return None

        hud_cfg = self.config.editorial_hud
        double_cuts = bool(hud_cfg.get("double_cuts_per_scene", False))
        alternate = bool(hud_cfg.get("alternate_layouts", True))
        slowmo = float(hud_cfg.get("slowmo_factor", 1.0))
        ffmpeg_exe = resolve_ffmpeg_executable()

        if len(sources) == 2:
            d1 = round(duration_sec * 0.5, 3)
            d2 = round(max(0.2, duration_sec - d1), 3)
            sub1 = clip_path.with_name(f"{clip_path.stem}_part1.mp4")
            sub2 = clip_path.with_name(f"{clip_path.stem}_part2.mp4")
            alt_mode = "fullscreen_portrait" if layout_mode == "wide_panel" else "wide_panel"
            self._normalize_video_clip(
                sources[0], sub1, d1, layout_mode=layout_mode, hud_png=hud_png, slowmo_factor=1.0
            )
            self._normalize_video_clip(
                sources[1],
                sub2,
                d2,
                layout_mode=alt_mode if alternate else layout_mode,
                hud_png=hud_png,
                slowmo_factor=slowmo,
            )
            self._concat_subclips([sub1, sub2], clip_path)
        else:
            src = sources[0]
            raw_dur = probe_media_duration(ffmpeg_exe, src)
            if double_cuts and raw_dur >= max(duration_sec + 1.5, 7.0):
                d1 = round(duration_sec * 0.5, 3)
                d2 = round(max(0.2, duration_sec - d1), 3)
                sub1 = clip_path.with_name(f"{clip_path.stem}_part1.mp4")
                sub2 = clip_path.with_name(f"{clip_path.stem}_part2.mp4")
                alt_mode = "fullscreen_portrait" if layout_mode == "wide_panel" else "wide_panel"
                offset2 = round(max(d1 + 0.3, raw_dur * 0.5), 3)
                self._normalize_video_clip(
                    src, sub1, d1, layout_mode=layout_mode, hud_png=hud_png, start_offset_sec=0.0, slowmo_factor=1.0
                )
                self._normalize_video_clip(
                    src,
                    sub2,
                    d2,
                    layout_mode=alt_mode if alternate else layout_mode,
                    hud_png=hud_png,
                    start_offset_sec=offset2,
                    slowmo_factor=slowmo,
                )
                self._concat_subclips([sub1, sub2], clip_path)
            else:
                self._normalize_video_clip(
                    src, clip_path, duration_sec, layout_mode=layout_mode, hud_png=hud_png, slowmo_factor=1.0
                )

        return GeneratedVideoClip(
            scene_number=scene_num,
            clip_path=str(clip_path),
            duration_sec=duration_sec,
            provider="preexisting_custom_video",
            motion_style=f"live_broll:{layout_mode}",
        )

    def _build_hud_for_scene(
        self,
        plan: ScriptPlan,
        scene: ScriptScene,
        hud_dir: Path,
    ) -> Path | None:
        hud_cfg = self.config.editorial_hud
        if not bool(hud_cfg.get("enabled", False)):
            return None
        hud_path = hud_dir / f"hud_{scene.scene_number:02d}.png"
        return render_scene_hud_overlay(
            hud_path,
            width=int(self.config.video.get("width", 1080)),
            height=int(self.config.video.get("height", 1920)),
            brand_label=str(hud_cfg.get("brand_label", "VIRAL STUDIO PRO")),
            accent_hex=str(hud_cfg.get("accent_color", "#CDFF43")),
            kicker=scene.kicker,
            headline_line1=scene.headline_line1,
            headline_line2=scene.headline_line2,
            footage_context=scene.footage_context or plan.footage_context_label,
            attribution_line=plan.attribution_line,
            fact_card=scene.fact_card,
            scene_index=scene.scene_number,
            total_scenes=len(plan.scenes),
            category_label=plan.category_label,
            assets_fonts_dir=self.config.assets_dir / "fonts",
            compact_context_badge=bool(hud_cfg.get("compact_context_badge", False)),
            show_fact_card_only_when_distinct=bool(
                hud_cfg.get("show_fact_card_only_when_distinct", False)
            ),
        )

    def _build_single_clip(
        self,
        photo: GeneratedPhoto,
        motion_style: str,
        duration_sec: float,
        clips_dir: Path,
        custom_clips_dir: Path | list[Path] | None = None,
        layout_mode: str = "wide_panel",
        hud_png: Path | None = None,
    ) -> GeneratedVideoClip:
        clip_path = clips_dir / f"clip_{photo.scene_number:02d}.mp4"
        custom_clip = self._try_custom_scene_clip(
            photo.scene_number,
            duration_sec,
            clip_path,
            custom_clips_dir,
            layout_mode=layout_mode,
            hud_png=hud_png,
        )
        if custom_clip:
            logger.info("[AI Model 4 - Video] Used custom live B-roll clip for scene %d.", photo.scene_number)
            return custom_clip
        return self._render_ai_motion_clip(photo, motion_style, duration_sec, clip_path, hud_png=hud_png)

    def run(
        self,
        plan: ScriptPlan,
        photos: list[GeneratedPhoto],
        scene_durations: dict[int, float],
        out_dir: Path,
        custom_clips_dir: Path | list[Path] | None = None,
    ) -> list[GeneratedVideoClip]:
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        hud_dir = out_dir / "hud"
        photo_map = {p.scene_number: p for p in photos}
        clips: list[GeneratedVideoClip] = []
        for scene in plan.scenes:
            photo = photo_map[scene.scene_number]
            dur = scene_durations[scene.scene_number]
            hud_png = self._build_hud_for_scene(plan, scene, hud_dir)
            layout = scene.layout_mode if scene.layout_mode in ("wide_panel", "fullscreen_portrait") else (
                "wide_panel" if scene.scene_number % 2 == 1 else "fullscreen_portrait"
            )
            clips.append(
                self._build_single_clip(
                    photo,
                    scene.video_motion_prompt,
                    dur,
                    clips_dir,
                    custom_clips_dir=custom_clips_dir,
                    layout_mode=layout,
                    hud_png=hud_png,
                )
            )
        (out_dir / "video_clips_manifest.json").write_text(
            json.dumps([c.__dict__ for c in clips], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return clips

    def regenerate_clips(
        self,
        plan: ScriptPlan,
        photos: list[GeneratedPhoto],
        existing_clips: list[GeneratedVideoClip],
        flagged_scenes: set[int],
        scene_durations: dict[int, float],
        motion_overrides: dict[int, str],
        out_dir: Path,
    ) -> list[GeneratedVideoClip]:
        clips_dir = out_dir / "clips"
        hud_dir = out_dir / "hud"
        photo_map = {p.scene_number: p for p in photos}
        scene_map = {s.scene_number: s for s in plan.scenes}
        updated: list[GeneratedVideoClip] = []
        for clip in existing_clips:
            if clip.scene_number in flagged_scenes:
                scene = scene_map[clip.scene_number]
                motion = motion_overrides.get(
                    clip.scene_number, scene.video_motion_prompt
                )
                hud_png = self._build_hud_for_scene(plan, scene, hud_dir)
                updated.append(
                    self._build_single_clip(
                        photo_map[clip.scene_number],
                        motion,
                        scene_durations[clip.scene_number],
                        clips_dir,
                        layout_mode=scene.layout_mode,
                        hud_png=hud_png,
                    )
                )
            else:
                updated.append(clip)
        (out_dir / "video_clips_manifest.json").write_text(
            json.dumps([c.__dict__ for c in updated], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return updated
