from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def resolve_ffmpeg_executable() -> str:
    """Find FFmpeg from FFMPEG_PATH, system PATH, WinGet packages, or imageio-ffmpeg."""
    env_ffmpeg = os.getenv("FFMPEG_PATH", "").strip()
    if env_ffmpeg and Path(env_ffmpeg).is_file():
        return env_ffmpeg

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    local_appdata = os.getenv("LOCALAPPDATA", "")
    if local_appdata:
        winget_root = Path(local_appdata) / "Microsoft" / "WinGet"
        link_candidate = winget_root / "Links" / "ffmpeg.exe"
        if link_candidate.is_file():
            return str(link_candidate)
        pkg_matches = sorted((winget_root / "Packages").glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"))
        if pkg_matches:
            return str(pkg_matches[-1])

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "FFmpeg executable not found on PATH, WinGet, or imageio-ffmpeg. "
            "Install FFmpeg via system package manager or 'pip install imageio-ffmpeg'."
        ) from exc


def resolve_ffprobe_executable() -> str:
    """Find ffprobe across Windows, Linux, and macOS."""
    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe
    try:
        ffmpeg_exe = Path(resolve_ffmpeg_executable())
        name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
        sibling = ffmpeg_exe.with_name(name)
        if sibling.is_file():
            return str(sibling)
    except Exception:
        pass
    return "ffprobe"


def detect_h264_encoder(ffmpeg_exe: str | None = None, prefer_nvenc: bool = True) -> str:
    """Return h264_nvenc only when available in local FFmpeg; otherwise fall back to libx264."""
    if not prefer_nvenc:
        return "libx264"
    exe = ffmpeg_exe or resolve_ffmpeg_executable()
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if proc.returncode == 0 and "h264_nvenc" in (proc.stdout or ""):
            return "h264_nvenc"
    except Exception:
        pass
    return "libx264"


def probe_media_duration(ffmpeg_exe: str, media_path: Path) -> float:
    """Measure media duration in seconds by parsing `ffmpeg -i` stderr (works without ffprobe)."""
    proc = subprocess.run(
        [ffmpeg_exe, "-i", str(media_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr)
    if not match:
        return 0.0
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return round(hours * 3600 + minutes * 60 + seconds, 3)


def measure_audio_metrics(ffmpeg_exe: str, media_path: Path) -> dict[str, Any]:
    """Measure integrated LUFS, true peak (dBTP), and stereo clipped samples via FFmpeg."""
    if not media_path.exists():
        return {"measured": False, "error": f"File not found: {media_path}"}

    proc = subprocess.run(
        [
            ffmpeg_exe,
            "-hide_banner",
            "-nostats",
            "-i",
            str(media_path),
            "-vn",
            "-af",
            "astats=metadata=0:reset=0,loudnorm=I=-15:TP=-1.5:LRA=7:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stderr = proc.stderr or ""
    metrics: dict[str, Any] = {"measured": proc.returncode == 0}

    json_match = re.search(r"\{\s*\"input_i\"\s*:.*?\}", stderr, re.DOTALL)
    if json_match:
        try:
            ln = json.loads(json_match.group(0))
            metrics["integrated_lufs"] = round(float(ln["input_i"]), 2)
            metrics["true_peak_dbtp"] = round(float(ln["input_tp"]), 2)
            metrics["lra_lu"] = round(float(ln["input_lra"]), 2)
        except Exception:
            pass

    clipped_matches = re.findall(r"Number of clipped samples:\s*(\d+)", stderr)
    if clipped_matches:
        metrics["clipped_samples"] = sum(int(x) for x in clipped_matches)
    else:
        metrics["clipped_samples"] = 0

    peak_matches = re.findall(r"Peak level dB:\s*(-?\d+(?:\.\d+)?)", stderr)
    if peak_matches:
        metrics["peak_level_db"] = round(max(float(x) for x in peak_matches), 2)

    return metrics


def extract_cover_frame(
    ffmpeg_exe: str, video_path: Path, out_jpg: Path, timestamp_sec: float = 2.0
) -> Path | None:
    """Extract a high-quality JPEG cover frame from the rendered 9:16 video."""
    if not video_path.exists():
        return None
    dur = probe_media_duration(ffmpeg_exe, video_path)
    ts = max(0.2, min(timestamp_sec, max(dur - 0.5, 0.2))) if dur > 0 else timestamp_sec
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-ss",
            f"{ts:.2f}",
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(out_jpg),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode == 0 and out_jpg.exists() and out_jpg.stat().st_size > 10_000:
        return out_jpg
    return None


def build_loop_qa_preview(
    ffmpeg_exe: str,
    video_path: Path,
    out_path: Path,
    tail_sec: float = 3.0,
    head_sec: float = 3.0,
) -> Path | None:
    """Concatenate the final tail and opening head back-to-back to inspect loop continuity."""
    if not video_path.exists():
        return None
    dur = probe_media_duration(ffmpeg_exe, video_path)
    if dur <= 0.4:
        return None

    eff_tail = min(tail_sec, max(0.3, dur * 0.45))
    eff_head = min(head_sec, max(0.3, dur * 0.45))
    tail_start = max(0.0, dur - eff_tail)

    probe = subprocess.run(
        [ffmpeg_exe, "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    has_audio = "Audio:" in (probe.stderr or "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    filter_parts = [
        f"[0:v]trim=start={tail_start:.3f}:duration={eff_tail:.3f},setpts=PTS-STARTPTS[v0]",
        f"[0:v]trim=start=0:duration={eff_head:.3f},setpts=PTS-STARTPTS[v1]",
        "[v0][v1]concat=n=2:v=1:a=0[vout]",
    ]
    cmd = [ffmpeg_exe, "-y", "-i", str(video_path)]
    if has_audio:
        filter_parts.extend(
            [
                f"[0:a]atrim=start={tail_start:.3f}:duration={eff_tail:.3f},asetpts=PTS-STARTPTS[a0]",
                f"[0:a]atrim=start=0:duration={eff_head:.3f},asetpts=PTS-STARTPTS[a1]",
                "[a0][a1]concat=n=2:v=0:a=1[aout]",
            ]
        )
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(out_path),
            ]
        )
    else:
        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[vout]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(out_path),
            ]
        )

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 10_000:
        return out_path
    return None
