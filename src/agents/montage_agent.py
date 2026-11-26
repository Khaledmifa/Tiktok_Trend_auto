from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.models import (
    GeneratedPhoto,
    GeneratedVideoClip,
    MediaManifest,
    ScriptPlan,
    WordTiming,
)
from src.providers.editorial_hud import wrap_caption_balanced
from src.providers.sound_design import (
    generate_bass_hit_sfx,
    generate_dark_ambient_bed,
    generate_whoosh_sfx,
)
from src.providers.tts import build_tts_provider
from src.utils.ffmpeg_tools import (
    build_loop_qa_preview,
    detect_h264_encoder,
    extract_cover_frame,
    measure_audio_metrics,
    probe_media_duration,
    resolve_ffmpeg_executable,
)

logger = logging.getLogger(__name__)

_POWER_WORDS = {
    "NEVER", "SECRET", "SECRETS", "BANNED", "CIA", "DEAD", "SILENCE",
    "TERRIFYING", "FROZE", "SHOCKED", "TWIST", "MILLION", "TRUTH",
    "ILLEGAL", "TRAP", "WOW", "MYSTERY", "NOBODY", "BRAIN", "ERASE",
    "HIDDEN", "LOUDER", "BROKE", "SPY", "DISASTER", "KILLED",
}


def _format_srt_ts(seconds: float) -> str:
    ms_total = max(0, int(round(seconds * 1000)))
    hrs, rem = divmod(ms_total, 3600000)
    mins, rem = divmod(rem, 60000)
    secs, ms = divmod(rem, 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _format_ass_ts(seconds: float) -> str:
    cs_total = max(0, int(round(seconds * 100)))
    hrs, rem = divmod(cs_total, 360000)
    mins, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def _sanitize_ass(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def _clean_alpha(word: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", word.upper())


class MontageAgent:
    """AI Model 5: Voiceover, kinetic ASS captions, studio SFX/music, and final 9:16 assembly."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _align_scenes_to_word_timings(
        self,
        plan: ScriptPlan,
        timings: list[WordTiming],
        target_total: float,
    ) -> tuple[dict[int, float], list[dict[str, Any]]]:
        """Align scene boundaries to actual WordBoundary timestamps when available."""
        scenes = plan.scenes
        if not scenes:
            return {}, []

        scene_word_counts = [max(len(s.voiceover_text.split()), 1) for s in scenes]
        total_script_words = sum(scene_word_counts) or 1

        if timings and len(timings) >= len(scenes):
            n_t = len(timings)
            starts: list[float] = [0.0]
            cum_words = 0
            for idx in range(len(scenes) - 1):
                cum_words += scene_word_counts[idx]
                mapped_idx = min(
                    n_t - 1,
                    max(idx + 1, int(round((cum_words / total_script_words) * n_t))),
                )
                cand_start = max(starts[-1] + 1.5, float(timings[mapped_idx].start_sec) - 0.08)
                # Ensure enough room remains for trailing scenes
                max_allowed = target_total - (len(scenes) - 1 - idx) * 1.5
                starts.append(round(min(cand_start, max(starts[-1] + 1.5, max_allowed)), 3))

            scene_durations: dict[int, float] = {}
            timeline_rows: list[dict[str, Any]] = []
            for idx, scene in enumerate(scenes):
                s_start = starts[idx]
                s_end = target_total if idx == len(scenes) - 1 else starts[idx + 1]
                dur = round(max(1.5, s_end - s_start), 3)
                scene_durations[scene.scene_number] = dur
                timeline_rows.append(
                    {
                        "scene_number": scene.scene_number,
                        "start_sec": round(s_start, 3),
                        "end_sec": round(s_start + dur, 3),
                        "duration_sec": dur,
                    }
                )
            return scene_durations, timeline_rows

        raw_est_sum = sum(max(s.estimated_duration_sec, 1.8) for s in scenes) or 1.0
        scene_durations = {}
        timeline_rows = []
        cursor = 0.0
        for scene in scenes:
            dur = round((max(scene.estimated_duration_sec, 1.8) / raw_est_sum) * target_total, 3)
            scene_durations[scene.scene_number] = dur
            timeline_rows.append(
                {
                    "scene_number": scene.scene_number,
                    "start_sec": round(cursor, 3),
                    "end_sec": round(cursor + dur, 3),
                    "duration_sec": dur,
                }
            )
            cursor += dur
        return scene_durations, timeline_rows

    def prepare_audio_and_scene_timings(
        self, plan: ScriptPlan, out_dir: Path
    ) -> tuple[Path, list[WordTiming], float, dict[int, float]]:
        """Generate voiceover first so AI Model 4 knows the exact duration of every scene."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ffmpeg_exe = resolve_ffmpeg_executable()
        audio_path = out_dir / "voiceover.mp3"

        tts_provider = build_tts_provider(self.config, plan.language)
        timings = tts_provider.synthesize(plan.full_voiceover_text, audio_path)

        audio_duration = probe_media_duration(ffmpeg_exe, audio_path)
        if audio_duration <= 0:
            raise RuntimeError(f"Generated voiceover audio is empty or unreadable: {audio_path}")

        floor_sec = float(self.config.video.get("hard_monetization_floor_sec", 61))
        safe_floor = round(floor_sec + (1.5 if floor_sec >= 60.0 else 0.5), 3)
        seamless_loop = bool(self.config.video.get("seamless_loop", False))
        if seamless_loop:
            tail_pad = float(self.config.video.get("loop_tail_pad_sec", 0.08))
            speech_end = max(
                audio_duration,
                float(timings[-1].end_sec) if timings else audio_duration,
            )
            target_total = max(round(speech_end + tail_pad, 3), safe_floor)
        else:
            target_total = max(audio_duration + 1.0, safe_floor)

        (out_dir / "word_times.json").write_text(
            json.dumps(
                [
                    {
                        "text": t.text,
                        "start_sec": round(t.start_sec, 4),
                        "end_sec": round(t.end_sec, 4),
                    }
                    for t in timings
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        scene_durations, timeline_rows = self._align_scenes_to_word_timings(
            plan, timings, target_total
        )
        (out_dir / "scene_timings.json").write_text(
            json.dumps(timeline_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return audio_path, timings, target_total, scene_durations

    def build_captions(
        self,
        timings: list[WordTiming],
        total_audio_sec: float,
        srt_path: Path,
        ass_path: Path,
        margin_v_override: int | None = None,
    ) -> None:
        if timings and timings[-1].end_sec > 0 and abs(timings[-1].end_sec - total_audio_sec) > 2.5:
            scale = total_audio_sec / timings[-1].end_sec
            timings = [
                WordTiming(text=t.text, start_sec=t.start_sec * scale, end_sec=t.end_sec * scale)
                for t in timings
            ]

        chunk_size = int(self.config.subtitles.get("words_per_caption_chunk", 2))
        break_on_sentence = bool(
            self.config.subtitles.get("break_on_sentence_boundary", chunk_size >= 3)
        )
        seamless_loop = bool(self.config.video.get("seamless_loop", False))
        uppercase = bool(self.config.subtitles.get("uppercase", True))
        active_hl = bool(self.config.subtitles.get("active_word_highlight", True))
        pop_anim = bool(self.config.subtitles.get("pop_animation", True))
        font_name = self.config.subtitles.get("font_name", "Arial Black")
        font_size = int(self.config.subtitles.get("font_size", 74))
        bold_flag = -1 if bool(self.config.subtitles.get("bold", True)) else 0
        outline_width = int(self.config.subtitles.get("outline_width", 6))
        shadow_depth = int(self.config.subtitles.get("shadow_depth", 4))
        max_line_width_px = int(self.config.subtitles.get("max_line_width_px", 790))
        use_fixed_pos = bool(self.config.subtitles.get("use_fixed_position", False))
        pos_x = int(self.config.subtitles.get("caption_center_x", 505))
        pos_y = int(self.config.subtitles.get("caption_center_y", 1300))
        fonts_dir = self.config.assets_dir / "fonts"
        max_cap_end = (
            max(total_audio_sec, timings[-1].end_sec if timings else 0.0)
            if seamless_loop
            else max(total_audio_sec, timings[-1].end_sec if timings else 0.0) + 0.3
        )

        # Group words into chunks, flushing cleanly at sentence-ending punctuation
        raw_groups: list[list[WordTiming]] = []
        current_group: list[WordTiming] = []
        max_words = max(chunk_size, 1)
        for w in timings:
            current_group.append(w)
            w_clean = w.text.strip()
            ends_sentence = break_on_sentence and w_clean.endswith((".", "?", "!", ";", "—"))
            if len(current_group) >= max_words or ends_sentence:
                raw_groups.append(current_group)
                current_group = []
        if current_group:
            raw_groups.append(current_group)

        srt_lines: list[str] = []
        chunk_idx = 1
        grouped_chunks: list[tuple[list[WordTiming], list[list[int]]]] = []
        for group in raw_groups:
            display_words = [
                (_sanitize_ass(w.text).upper() if uppercase else _sanitize_ass(w.text))
                for w in group
            ]
            line_groups = wrap_caption_balanced(
                display_words,
                font_size=font_size,
                max_width_px=max_line_width_px,
                assets_fonts_dir=fonts_dir,
                bold=bool(bold_flag),
            )
            grouped_chunks.append((group, line_groups))
            srt_text = "\n".join(
                " ".join(display_words[idx] for idx in line_idxs) for line_idxs in line_groups
            ).strip()
            if not srt_text:
                continue
            s_start = group[0].start_sec
            s_end = min(max_cap_end, max(group[-1].end_sec, s_start + 0.2))
            srt_lines.extend(
                [
                    str(chunk_idx),
                    f"{_format_srt_ts(s_start)} --> {_format_srt_ts(s_end)}",
                    srt_text,
                    "",
                ]
            )
            chunk_idx += 1
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

        # ASS script coordinates (font_size=57, pos=(505,1300), max_line_width=790) are calibrated
        # on a 1080x1920 virtual layout; libass scales PlayRes vector glyphs crisply to 4K (2160x3840).
        play_res_x, play_res_y = 1080, 1920
        primary_color = self.config.subtitles.get("primary_color", "&H00FFFFFF")
        highlight_color = self.config.subtitles.get("highlight_color", "&H0000EAFF")
        power_color = self.config.subtitles.get("power_color", "&H003B4DFF")
        outline_color = self.config.subtitles.get("outline_color", "&H00000000")
        raw_margin = (
            margin_v_override
            if margin_v_override is not None
            else self.config.subtitles.get("margin_v", 520)
        )
        margin_v = max(260, min(760, int(raw_margin)))
        alignment = 5 if use_fixed_pos else 2
        pos_tag = f"{{\\pos({pos_x},{pos_y})}}" if use_fixed_pos else ""

        ass_header = "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {play_res_x}",
                f"PlayResY: {play_res_y}",
                "WrapStyle: 2",
                "",
                "[V4+ Styles]",
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
                f"Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H90000000,{bold_flag},0,0,0,100,100,0,0,1,{outline_width},{shadow_depth},{alignment},50,50,{margin_v},1",
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            ]
        )

        events: list[str] = []
        last_end = 0.0
        for group, line_groups in grouped_chunks:
            if not group:
                continue

            display_words = [
                (_sanitize_ass(w.text).upper() if uppercase else _sanitize_ass(w.text))
                for w in group
            ]
            if not active_hl:
                txt = "\\N".join(
                    " ".join(display_words[idx] for idx in line_idxs) for line_idxs in line_groups
                )
                start_s = max(last_end, group[0].start_sec)
                end_s = min(max_cap_end, max(group[-1].end_sec, start_s + 0.22))
                last_end = end_s
                pop_tag = "{\\fscx112\\fscy112\\t(0,80,\\fscx100\\fscy100)}" if pop_anim else ""
                events.append(
                    f"Dialogue: 0,{_format_ass_ts(start_s)},{_format_ass_ts(end_s)},Default,,0,0,0,,{pos_tag}{pop_tag}{txt}"
                )
                continue

            for active_idx, active_word in enumerate(group):
                w_start = max(last_end, active_word.start_sec)
                if active_idx + 1 < len(group):
                    w_end = max(group[active_idx + 1].start_sec, w_start + 0.08)
                else:
                    w_end = max(active_word.end_sec, w_start + 0.14)
                w_end = min(max_cap_end, max(w_end, w_start + 0.06))
                last_end = w_end

                styled_lines: list[str] = []
                for line_idxs in line_groups:
                    line_parts: list[str] = []
                    for j in line_idxs:
                        word_txt = display_words[j]
                        if j == active_idx:
                            col = power_color if _clean_alpha(word_txt) in _POWER_WORDS else highlight_color
                            line_parts.append(f"{{\\c{col}}}{word_txt}{{\\c{primary_color}}}")
                        else:
                            line_parts.append(word_txt)
                    styled_lines.append(" ".join(line_parts))

                pop_tag = (
                    "{\\fscx114\\fscy114\\t(0,75,\\fscx100\\fscy100)}"
                    if (pop_anim and active_idx == 0)
                    else ""
                )
                events.append(
                    f"Dialogue: 0,{_format_ass_ts(w_start)},{_format_ass_ts(w_end)},Default,,0,0,0,,{pos_tag}{pop_tag}{'\\N'.join(styled_lines)}"
                )

        ass_path.write_text(ass_header + "\n" + "\n".join(events) + "\n", encoding="utf-8")

    def _find_background_music(self) -> Path | None:
        music_dir = self.config.assets_dir / "music"
        for ext in ("*.mp3", "*.wav", "*.m4a", "*.ogg"):
            matches = sorted(music_dir.glob(ext))
            if matches:
                return matches[0]
        return None

    def assemble(
        self,
        run_id: str,
        audio_path: Path,
        timings: list[WordTiming],
        total_duration_sec: float,
        photos: list[GeneratedPhoto],
        clips: list[GeneratedVideoClip],
        out_dir: Path,
        montage_overrides: dict[str, Any] | None = None,
    ) -> MediaManifest:
        overrides = montage_overrides or {}
        ffmpeg_exe = resolve_ffmpeg_executable()
        sfx_dir = out_dir / "sfx"
        sfx_dir.mkdir(parents=True, exist_ok=True)

        audio_dur = probe_media_duration(ffmpeg_exe, audio_path)
        srt_path = out_dir / "captions.srt"
        ass_path = out_dir / "captions.ass"
        self.build_captions(
            timings,
            audio_dur,
            srt_path,
            ass_path,
            margin_v_override=overrides.get("margin_v"),
        )

        # Concatenate video clips from AI Model 4
        concat_file = out_dir / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file 'clips/{Path(c.clip_path).name}'" for c in clips),
            encoding="utf-8",
        )
        stitched_video = out_dir / "stitched_silent.mp4"
        proc = subprocess.run(
            [
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                "-c", "copy", stitched_video.name,
            ],
            cwd=out_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            proc = subprocess.run(
                [
                    ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p", stitched_video.name,
                ],
                cwd=out_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed:\n{proc.stderr[-1000:]}")

        # Studio Sound Design
        sfx_cfg = self.config.raw.get("sound_design", {})
        sfx_enabled = bool(sfx_cfg.get("enabled", True))
        sample_rate = int(sfx_cfg.get("sample_rate", 44100))
        raw_bed = overrides.get("bed_volume")
        bed_vol = float(sfx_cfg.get("bed_volume", 0.14) if raw_bed is None else raw_bed)
        sfx_vol = float(sfx_cfg.get("sfx_volume", 0.34))
        whoosh_step = max(int(sfx_cfg.get("whoosh_every_n_cuts", 2)), 1)
        normalize_vo = bool(sfx_cfg.get("normalize_voiceover", True))
        target_lufs = float(sfx_cfg.get("target_lufs", -14.0))
        true_peak_db = float(sfx_cfg.get("true_peak_db", -1.5))
        apply_limiter = bool(sfx_cfg.get("apply_limiter", False))

        music_path = self._find_background_music()
        seamless_bed = bool(sfx_cfg.get("seamless_loop", False)) or bool(
            self.config.video.get("seamless_loop", False)
        )
        if sfx_enabled and not music_path:
            music_path = generate_dark_ambient_bed(
                sfx_dir / "dark_ambient_bed.wav",
                total_duration_sec,
                sample_rate=sample_rate,
                seamless_loop=seamless_bed,
            )
        bass_hit_path = (
            generate_bass_hit_sfx(sfx_dir / "bass_hit.wav", sample_rate=sample_rate) if sfx_enabled else None
        )
        whoosh_path = (
            generate_whoosh_sfx(sfx_dir / "whoosh.wav", sample_rate=sample_rate) if sfx_enabled else None
        )

        cut_timestamps: list[float] = []
        cursor = 0.0
        for idx, c in enumerate(clips):
            if idx > 0:
                cut_timestamps.append(cursor)
            cursor += c.duration_sec

        final_video_path = out_dir / "final_9x16.mp4"
        burn_in = bool(self.config.subtitles.get("enabled", True)) and bool(
            self.config.subtitles.get("burn_in", True)
        )
        crf = int(self.config.video.get("crf", 19))

        audio_fmt = f"aresample={sample_rate},aformat=sample_rates={sample_rate}:channel_layouts=stereo"
        cmd = [ffmpeg_exe, "-y", "-i", stitched_video.name, "-i", audio_path.name]
        filter_parts: list[str] = []
        vo_chain = (
            (f"loudnorm=I={target_lufs:.1f}:TP={true_peak_db:.1f}:LRA=7," if normalize_vo else "")
            + f"{audio_fmt},apad=whole_dur={total_duration_sec:.3f}"
        )
        filter_parts.append(f"[1:a]{vo_chain}[vo]")

        audio_mix_labels = ["[vo]"]
        next_idx = 2

        if music_path:
            cmd.extend(["-stream_loop", "-1", "-i", str(music_path)])
            filter_parts.append(
                f"[{next_idx}:a]{audio_fmt},volume={bed_vol:.2f},atrim=0:{total_duration_sec:.3f}[bgm]"
            )
            audio_mix_labels.append("[bgm]")
            next_idx += 1

        if sfx_enabled and bass_hit_path:
            twist_ms = int(total_duration_sec * 0.50 * 1000)
            for delay_ms, tag in ((50, "hit0"), (twist_ms, "hit1")):
                cmd.extend(["-i", str(bass_hit_path)])
                filter_parts.append(
                    f"[{next_idx}:a]{audio_fmt},adelay={delay_ms}|{delay_ms},volume={sfx_vol:.2f}[{tag}]"
                )
                audio_mix_labels.append(f"[{tag}]")
                next_idx += 1

        if sfx_enabled and whoosh_path and cut_timestamps:
            for c_idx, cut_sec in enumerate(cut_timestamps[::whoosh_step][:8]):
                delay_ms = max(0, int(cut_sec * 1000) - 120)
                cmd.extend(["-i", str(whoosh_path)])
                filter_parts.append(
                    f"[{next_idx}:a]{audio_fmt},adelay={delay_ms}|{delay_ms},volume={sfx_vol * 0.65:.2f}[w{c_idx}]"
                )
                audio_mix_labels.append(f"[w{c_idx}]")
                next_idx += 1

        if len(audio_mix_labels) > 1:
            limiter_suffix = ",alimiter=limit=0.89:attack=5:release=50" if apply_limiter else ""
            filter_parts.append(
                f"{''.join(audio_mix_labels)}amix=inputs={len(audio_mix_labels)}:duration=first:normalize=0{limiter_suffix}[aout]"
            )
            audio_map = "[aout]"
        elif apply_limiter:
            filter_parts.append("[vo]alimiter=limit=0.89:attack=5:release=50[aout]")
            audio_map = "[aout]"
        else:
            audio_map = "[vo]"

        if burn_in:
            filter_parts.append(f"[0:v]ass={ass_path.name}[vout]")
            video_map = "[vout]"
        else:
            video_map = "0:v"

        prefer_nvenc = str(self.config.video.get("preferred_encoder", "h264_nvenc")).lower() == "h264_nvenc"
        detected_enc = detect_h264_encoder(ffmpeg_exe, prefer_nvenc=prefer_nvenc)
        encode_profiles: list[list[str]] = []
        if detected_enc == "h264_nvenc":
            encode_profiles.append(
                [
                    "-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr",
                    "-cq", str(crf), "-b:v", "0", "-maxrate", "35M", "-bufsize", "70M",
                    "-profile:v", "high", "-pix_fmt", "yuv420p",
                ]
            )
        encode_profiles.append(
            [
                "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                "-profile:v", "high", "-pix_fmt", "yuv420p",
            ]
        )

        last_err = ""
        for v_args in encode_profiles:
            full_cmd = [
                *cmd,
                "-filter_complex", ";".join(filter_parts),
                "-map", video_map, "-map", audio_map,
                *v_args,
                "-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate), "-t", f"{total_duration_sec:.3f}",
                "-movflags", "+faststart", final_video_path.name,
            ]
            proc = subprocess.run(
                full_cmd, cwd=out_dir, capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if proc.returncode == 0:
                last_err = ""
                break
            last_err = proc.stderr[-1200:]
        if last_err:
            raise RuntimeError(f"FFmpeg montage assembly failed:\n{last_err}")

        actual_final = probe_media_duration(ffmpeg_exe, final_video_path)
        audio_metrics = measure_audio_metrics(ffmpeg_exe, final_video_path)
        cover_ts = round(min(max(actual_final * 0.14, 1.0), max(actual_final - 0.5, 1.0)), 2)
        cover_file = extract_cover_frame(ffmpeg_exe, final_video_path, out_dir / "cover.jpg", timestamp_sec=cover_ts)
        loop_qa_file = (
            build_loop_qa_preview(ffmpeg_exe, final_video_path, out_dir / "loop_qa_transition.mp4")
            if bool(self.config.video.get("seamless_loop", False))
            else None
        )

        width = int(self.config.video.get("width", 1080))
        height = int(self.config.video.get("height", 1920))
        fps = int(self.config.video.get("fps", 30))

        photo_map = {p.scene_number: p for p in photos}
        clip_map = {c.scene_number: c for c in clips}
        sources_summary = [
            {
                "scene": str(sn),
                "photo_provider": photo_map[sn].provider if sn in photo_map else "unknown",
                "video_provider": clip_map[sn].provider if sn in clip_map else "unknown",
                "attribution": photo_map[sn].attribution if sn in photo_map else "",
            }
            for sn in sorted(photo_map.keys())
        ]

        manifest = MediaManifest(
            run_id=run_id,
            audio_path=str(audio_path),
            srt_path=str(srt_path),
            ass_path=str(ass_path),
            final_video_path=str(final_video_path),
            actual_duration_sec=actual_final,
            resolution=f"{width}x{height}",
            fps=fps,
            visual_sources_used=sources_summary,
            has_background_music=music_path is not None,
            cover_path=str(cover_file) if cover_file else "",
            audio_metrics=audio_metrics,
            loop_qa_path=str(loop_qa_file) if loop_qa_file else "",
        )
        (out_dir / "media_manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return manifest
