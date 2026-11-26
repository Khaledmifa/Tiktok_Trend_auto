from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from src.agents.photo_agent import PhotoGeneratorAgent
from src.utils.ffmpeg_tools import probe_media_duration, resolve_ffmpeg_executable

logger = logging.getLogger(__name__)

BLOCKED_WATERMARK_DOMAINS: tuple[str, ...] = (
    "gettyimages",
    "shutterstock",
    "alamy",
    "istockphoto",
    "dreamstime",
    "123rf",
    "depositphotos",
    "vecteezy",
    "watermark",
)


def parse_cropdetect(stderr_text: str, frame_w: int = 1920, frame_h: int = 1080) -> str | None:
    """Return most frequent valid FFmpeg crop=W:H:X:Y if it strips bars without crushing the frame."""
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", stderr_text or "")
    if not matches:
        return None
    counts = Counter(matches)
    min_area = max(frame_w * frame_h * 0.50, 100_000)
    for crop_spec, _ in counts.most_common():
        parts = [int(p) for p in crop_spec.split(":")]
        cw, ch, cx, cy = parts
        if cw >= 320 and ch >= 240 and (cw * ch) >= min_area and (cx > 0 or cy > 0 or cw < frame_w or ch < frame_h):
            return f"crop={cw}:{ch}:{cx}:{cy}"
    return None


def compute_segment_starts(duration_sec: float, count: int, seg_len: float) -> list[float]:
    """Select evenly spaced segment start timestamps avoiding intros and end cards."""
    if count <= 0 or duration_sec <= seg_len + 2.0:
        return [0.0] * max(count, 0)
    low = min(max(3.0, duration_sec * 0.12), max(duration_sec - seg_len - 1.0, 0.0))
    high = max(low + 0.5, min(duration_sec * 0.88 - seg_len, duration_sec - seg_len - 1.0))
    if count == 1:
        return [round((low + high) / 2.0, 2)]
    span = max(high - low, 0.5)
    step = span / max(count - 1, 1)
    return [round(min(high, low + i * step), 2) for i in range(count)]


def _probe_dimensions(ffmpeg_exe: str, media_path: Path) -> tuple[int, int]:
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
    return 1920, 1080


def _detect_crop_filter(ffmpeg_exe: str, source_mp4: Path, duration_sec: float) -> str | None:
    sample_ts = round(max(2.0, min(duration_sec * 0.35, max(duration_sec - 4.0, 2.0))), 2)
    w, h = _probe_dimensions(ffmpeg_exe, source_mp4)
    proc = subprocess.run(
        [
            ffmpeg_exe,
            "-ss",
            f"{sample_ts:.2f}",
            "-t",
            "3",
            "-i",
            str(source_mp4),
            "-vf",
            "cropdetect=24:16:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return parse_cropdetect(proc.stderr, frame_w=w, frame_h=h)


def _clean_topic_for_query(topic: str) -> str:
    cleaned = re.sub(r"[—–|]+", " ", topic or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:90]


def _download_ytdlp_candidates(
    queries: list[str],
    tmp_dir: Path,
    max_downloads: int = 2,
    max_height: int = 2160,
) -> list[Path]:
    try:
        import yt_dlp
    except Exception as exc:
        logger.warning("yt_dlp unavailable (%s); skipping video harvest.", exc)
        return []

    tmp_dir.mkdir(parents=True, exist_ok=True)
    picked_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []

    search_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 15,
    }
    with yt_dlp.YoutubeDL(search_opts) as ydl:
        for q in queries[:3]:
            try:
                info = ydl.extract_info(f"ytsearch5:{q}", download=False)
                for entry in (info or {}).get("entries") or []:
                    if not isinstance(entry, dict):
                        continue
                    vid = str(entry.get("id") or "").strip()
                    url = str(entry.get("url") or entry.get("webpage_url") or "").strip()
                    dur = float(entry.get("duration") or 0)
                    title_low = str(entry.get("title") or "").lower()
                    if not vid or vid in picked_ids:
                        continue
                    if entry.get("is_live") or "live stream" in title_low or "#shorts" in title_low:
                        continue
                    if dur and (dur < 25 or dur > 600):
                        continue
                    if not url.startswith("http"):
                        url = f"https://www.youtube.com/watch?v={vid}"
                    picked_ids.add(vid)
                    candidates.append({"id": vid, "url": url, "title": str(entry.get("title") or vid)})
            except Exception as exc:
                logger.debug("yt-dlp search skipped for '%s': %s", q, exc)

    downloaded: list[Path] = []
    fmt_selector = (
        f"bestvideo[height<={max_height}]/"
        "bestvideo[height<=1080]/"
        f"best[height<={max_height}]/best"
    )
    for idx, cand in enumerate(candidates[:max_downloads]):
        out_tmpl = str(tmp_dir / f"src_{idx:02d}.%(ext)s")
        dl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 20,
            "retries": 2,
            "format": fmt_selector,
            "outtmpl": out_tmpl,
            "merge_output_format": "mp4",
        }
        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([cand["url"]])
            for fp in sorted(tmp_dir.glob(f"src_{idx:02d}.*")):
                if fp.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov") and fp.stat().st_size > 100_000:
                    downloaded.append(fp)
                    break
        except Exception as exc:
            logger.debug("yt-dlp download failed for %s: %s", cand["url"], exc)
    return downloaded


def _is_watermarked_url(url: str) -> bool:
    low = (url or "").lower()
    return any(dom in low for dom in BLOCKED_WATERMARK_DOMAINS)


def _harvest_ddgs_images(topic: str, target_scenes: list[int], photos_dir: Path) -> list[str]:
    if not target_scenes:
        return []
    created: list[str] = []
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return created

    clean_topic = _clean_topic_for_query(topic)
    query = f"{clean_topic} high resolution -getty -shutterstock -alamy"
    image_urls: list[str] = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.images(query, max_results=max(14, len(target_scenes) * 2)):
                url = str(item.get("image") or "").strip()
                if url.startswith("http") and not _is_watermarked_url(url):
                    image_urls.append(url)
    except Exception as exc:
        logger.debug("DDGS image search skipped: %s", exc)
        return created

    photos_dir.mkdir(parents=True, exist_ok=True)
    url_idx = 0
    for sn in target_scenes:
        out_path = photos_dir / f"scene_{sn:02d}_photo.jpg"
        while url_idx < len(image_urls):
            url = image_urls[url_idx]
            url_idx += 1
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 ViralStudioPro/1.0"},
                    timeout=12,
                )
                if not resp.ok or len(resp.content) < 10_000:
                    continue
                tmp_raw = photos_dir / f"_tmp_{sn:02d}.img"
                tmp_raw.write_bytes(resp.content)
                try:
                    with Image.open(tmp_raw) as im:
                        if im.width < 400 or im.height < 300:
                            continue
                        fitted = PhotoGeneratorAgent._fit_to_vertical_9x16(im.convert("RGB"))
                    fitted.save(out_path, format="JPEG", quality=93)
                    created.append(str(out_path))
                    break
                finally:
                    tmp_raw.unlink(missing_ok=True)
            except Exception:
                continue
    return created


def harvest_media_for_trend(
    topic: str,
    run_dir: Path,
    scene_count: int = 10,
    sourcing_keywords: list[str] | None = None,
    seg_duration: float = 9.5,
    max_height: int = 2160,
) -> dict[str, Any]:
    """$0 local media harvester: fetches real 4K/1080p clips via yt-dlp + clean web images into run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    photos_dir = run_dir / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "_tmp_harvest"

    existing_scenes: set[int] = set()
    for sn in range(1, scene_count + 1):
        has_clip = any((run_dir / n).exists() for n in (f"scene_{sn:02d}.mp4", f"scene_{sn:02d}_clip.mp4"))
        has_photo = any(
            (d / f"{stem}{ext}").exists()
            for d in (run_dir, photos_dir)
            for stem in (f"scene_{sn:02d}", f"scene_{sn:02d}_photo")
            for ext in (".jpg", ".jpeg", ".png", ".webp")
        )
        if has_clip or has_photo:
            existing_scenes.add(sn)

    missing_scenes = [sn for sn in range(1, scene_count + 1) if sn not in existing_scenes]
    if not missing_scenes:
        return {"ok": True, "clips_created": [], "photos_created": [], "source_videos": [], "skipped": "already_staged"}

    clean_topic = _clean_topic_for_query(topic)
    queries: list[str] = []
    for kw in (sourcing_keywords or []):
        clean_kw = _clean_topic_for_query(str(kw))
        if clean_kw and clean_kw not in queries:
            queries.append(clean_kw)
    for fallback_q in (f"{clean_topic} official trailer 1080p", f"{clean_topic} scenes showcase 1080p"):
        if fallback_q not in queries:
            queries.append(fallback_q)

    clips_created: list[str] = []
    photos_created: list[str] = []
    source_paths: list[str] = []

    try:
        downloaded = _download_ytdlp_candidates(queries, tmp_dir, max_downloads=2, max_height=max_height)
        source_paths = [str(p) for p in downloaded]
        if downloaded:
            ffmpeg_exe = resolve_ffmpeg_executable()
            # Distribute missing scenes across downloaded videos
            buckets: list[list[int]] = [[] for _ in downloaded]
            for idx, sn in enumerate(missing_scenes):
                buckets[idx % len(downloaded)].append(sn)

            for src_path, scene_bucket in zip(downloaded, buckets):
                if not scene_bucket:
                    continue
                dur = probe_media_duration(ffmpeg_exe, src_path)
                if dur <= 6.0:
                    continue
                crop_vf = _detect_crop_filter(ffmpeg_exe, src_path, dur)
                starts = compute_segment_starts(dur, len(scene_bucket), min(seg_duration, max(dur * 0.4, 4.0)))
                for sn, start_ts in zip(scene_bucket, starts):
                    clip_out = run_dir / f"scene_{sn:02d}.mp4"
                    vf_parts = [crop_vf] if crop_vf else []
                    vf_str = ",".join(vf_parts) if vf_parts else None
                    cmd = [
                        ffmpeg_exe,
                        "-y",
                        "-ss",
                        f"{start_ts:.2f}",
                        "-i",
                        str(src_path),
                        "-t",
                        f"{seg_duration:.2f}",
                    ]
                    if vf_str:
                        cmd.extend(["-vf", vf_str])
                    cmd.extend(
                        [
                            "-c:v",
                            "libx264",
                            "-preset",
                            "fast",
                            "-crf",
                            "14",
                            "-pix_fmt",
                            "yuv420p",
                            "-an",
                            str(clip_out),
                        ]
                    )
                    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                    if proc.returncode == 0 and clip_out.exists() and clip_out.stat().st_size > 25_000:
                        clips_created.append(str(clip_out))
                        frame_raw = tmp_dir / f"frame_{sn:02d}.jpg"
                        frame_out = photos_dir / f"scene_{sn:02d}_photo.jpg"
                        subprocess.run(
                            [
                                ffmpeg_exe,
                                "-y",
                                "-ss",
                                "1.5",
                                "-i",
                                str(clip_out),
                                "-vframes",
                                "1",
                                "-q:v",
                                "3",
                                str(frame_raw),
                            ],
                            capture_output=True,
                        )
                        if frame_raw.exists():
                            with Image.open(frame_raw) as im:
                                fitted = PhotoGeneratorAgent._fit_to_vertical_9x16(im.convert("RGB"))
                            fitted.save(frame_out, format="JPEG", quality=92)
                            photos_created.append(str(frame_out))
                        existing_scenes.add(sn)

        still_missing = [sn for sn in range(1, scene_count + 1) if sn not in existing_scenes]
        if still_missing:
            photos_created.extend(_harvest_ddgs_images(topic, still_missing, photos_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "ok": True,
        "clips_created": clips_created,
        "photos_created": photos_created,
        "source_videos": source_paths,
    }
