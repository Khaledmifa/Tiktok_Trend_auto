from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.utils.ffmpeg_tools import (
    measure_audio_metrics,
    resolve_ffmpeg_executable,
    resolve_ffprobe_executable,
)
from src.utils.language_guard import is_subtitle_file_english

FORBIDDEN_UPLOAD_PATTERNS = [
    r"tiktok[-_]?uploader",
    r"open\.tiktokapis\.com",
    r"youtube[-_]?upload",
    r"googleapis\.com/upload/youtube",
    r"auto[-_]?publish",
    r"upload_to_tiktok",
]


def _resolve_ffprobe() -> str:
    return resolve_ffprobe_executable()


def verify_video(video_path: Path, min_sec: float = 61.0) -> dict[str, Any]:
    """Validate rendered TikTok 9:16 MP4 using local ffprobe (0 AI tokens)."""
    issues: list[str] = []
    if not video_path.exists():
        return {
            "passed": False,
            "video_path": str(video_path),
            "issues": [f"Video file not found: {video_path}"],
        }

    size_bytes = video_path.stat().st_size
    if size_bytes < 100_000:
        issues.append(f"File too small ({size_bytes} bytes).")

    ffprobe_exe = _resolve_ffprobe()
    proc = subprocess.run(
        [
            ffprobe_exe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        issues.append(f"ffprobe failed: {proc.stderr.strip()}")
        return {"passed": False, "video_path": str(video_path), "issues": issues}

    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    duration_sec = float(fmt.get("duration") or 0.0)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    has_video = len(video_streams) > 0
    has_audio = len(audio_streams) > 0
    width = int(video_streams[0].get("width") or 0) if has_video else 0
    height = int(video_streams[0].get("height") or 0) if has_video else 0

    if duration_sec <= min_sec:
        issues.append(f"Duration {duration_sec:.2f}s must exceed {min_sec:.1f}s.")
    if not has_video:
        issues.append("No video stream found.")
    elif (width, height) not in {(1080, 1920), (2160, 3840)}:
        issues.append(f"Expected 2160x3840 (4K) or 1080x1920 vertical 9:16, got {width}x{height}.")
    if not has_audio:
        issues.append("No audio stream found.")

    audio_metrics: dict[str, Any] = {}
    if has_audio:
        try:
            audio_metrics = measure_audio_metrics(resolve_ffmpeg_executable(), video_path)
            if audio_metrics.get("measured") and int(audio_metrics.get("clipped_samples", 0)) > 0:
                issues.append(
                    f"Stereo audio clipping detected ({audio_metrics['clipped_samples']} clipped samples)."
                )
        except Exception:
            pass

    srt_path = video_path.parent / "captions.srt"
    if srt_path.exists():
        srt_en_ok, bad_cues = is_subtitle_file_english(srt_path)
        if not srt_en_ok:
            sample = bad_cues[0][:80] if bad_cues else ""
            issues.append(
                f"Non-English subtitle text detected in captions.srt (English-only rule enforced): '{sample}'"
            )

    return {
        "passed": len(issues) == 0,
        "video_path": str(video_path),
        "size_bytes": size_bytes,
        "duration_sec": round(duration_sec, 2),
        "width": width,
        "height": height,
        "has_video": has_video,
        "has_audio": has_audio,
        "audio_metrics": audio_metrics,
        "issues": issues,
    }


def _find_recent_final_video(command_line: str) -> Path | None:
    run_match = re.search(r"--run-id\s+([^\s\"']+)", command_line)
    lang_match = re.search(r"--lang\s+([^\s\"']+)", command_line)
    if run_match:
        run_id = run_match.group(1)
        base = WORKSPACE_ROOT / "tiktok" / run_id
        candidate = base / lang_match.group(1).lower() / "final_9x16.mp4" if lang_match else base / "final_9x16.mp4"
        if candidate.exists():
            return candidate

    tiktok_dir = WORKSPACE_ROOT / "tiktok"
    if not tiktok_dir.is_dir():
        return None
    now = time.time()
    recent = [
        p
        for p in tiktok_dir.glob("**/final_9x16.mp4")
        if (now - p.stat().st_mtime) <= 180.0
    ]
    if not recent:
        return None
    return max(recent, key=lambda p: p.stat().st_mtime)


def handle_pre_tool_hook() -> int:
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    tool_call = payload.get("toolCall", {})
    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    cmd = str(args.get("CommandLine", ""))

    for pattern in FORBIDDEN_UPLOAD_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            print(
                json.dumps(
                    {
                        "decision": "deny",
                        "reason": (
                            f"Blocked by Viral Studio Pro guard: automated public upload pattern "
                            f"'{pattern}' is forbidden. Upload manually from publish_kit.md."
                        ),
                    }
                )
            )
            return 0

    print(json.dumps({"decision": "allow"}))
    return 0


def handle_post_tool_hook(min_sec: float) -> int:
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    if payload.get("error"):
        print("{}")
        return 0

    tool_call = payload.get("toolCall", {})
    args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
    cmd = str(args.get("CommandLine", ""))
    lower = cmd.lower()

    is_video_build = (
        ("run_pipeline.py" in lower and "--approve" not in lower and "--discover-trend-only" not in lower)
        or "final_9x16.mp4" in lower
    )
    if is_video_build:
        target = _find_recent_final_video(cmd)
        if target is not None:
            report = verify_video(target, min_sec=min_sec)
            report_path = target.parent / "video_gate_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("{}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zero-token ffprobe video gate & Antigravity safety hook."
    )
    parser.add_argument("video_path", nargs="?", default=None, help="Path to MP4 file to verify")
    parser.add_argument("--min-sec", type=float, default=61.0, help="Minimum required duration (>61s)")
    parser.add_argument("--pre-tool-hook", action="store_true", help="Run as Antigravity PreToolUse hook")
    parser.add_argument("--post-tool-hook", action="store_true", help="Run as Antigravity PostToolUse hook")
    args = parser.parse_args()

    if args.pre_tool_hook:
        return handle_pre_tool_hook()
    if args.post_tool_hook:
        return handle_post_tool_hook(min_sec=args.min_sec)
    if not args.video_path:
        parser.error("video_path is required unless using --pre-tool-hook or --post-tool-hook")

    result = verify_video(Path(args.video_path), min_sec=args.min_sec)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
