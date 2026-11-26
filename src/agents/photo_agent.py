from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageOps

from src.config import AppConfig
from src.models import GeneratedPhoto, ScriptPlan

logger = logging.getLogger(__name__)

BLOCKED_WATERMARK_DOMAINS = (
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


class PhotoGeneratorAgent:
    """AI Model 3: Prepares clean 9:16 vertical scene visuals from staged media or free web images ($0)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        models_cfg = config.raw.get("models", {})
        self.priority: list[str] = list(
            models_cfg.get(
                "photo_provider_priority",
                ["ddgs_web_image", "offline_card"],
            )
        )
        self._disabled_providers: set[str] = set()
        self._ddgs_cache: list[str] | None = None

    def _validate_image(self, path: Path, min_bytes: int = 10000) -> bool:
        if not path.exists() or path.stat().st_size < min_bytes:
            return False
        try:
            with Image.open(path) as img:
                w, h = img.size
                return w >= 400 and h >= 300
        except Exception:
            return False

    def _try_ddgs_image(self, scene_num: int, prompt: str, out_path: Path) -> GeneratedPhoto | None:
        if "ddgs_web_image" in self._disabled_providers:
            return None
        try:
            from duckduckgo_search import DDGS
        except Exception:
            self._disabled_providers.add("ddgs_web_image")
            return None

        query = f"{prompt[:75]} high resolution -getty -shutterstock -alamy"
        try:
            urls: list[str] = []
            with DDGS() as ddgs:
                for item in ddgs.images(query, max_results=6):
                    u = str(item.get("image") or "").strip()
                    low = u.lower()
                    if u.startswith("http") and not any(b in low for b in BLOCKED_WATERMARK_DOMAINS):
                        urls.append(u)
            for u in urls:
                resp = requests.get(
                    u, headers={"User-Agent": "Mozilla/5.0 ViralStudioPro/1.0"}, timeout=12
                )
                if not resp.ok or len(resp.content) < 10_000:
                    continue
                with Image.open(io.BytesIO(resp.content)) as im:
                    if im.width < 400 or im.height < 300:
                        continue
                    fitted = self._fit_to_vertical_9x16(
                        im.convert("RGB"),
                        target=(
                            int(self.config.video.get("width", 1080)),
                            int(self.config.video.get("height", 1920)),
                        ),
                    )
                fitted.save(out_path, format="JPEG", quality=93)
                if self._validate_image(out_path):
                    return GeneratedPhoto(
                        scene_number=scene_num,
                        image_path=str(out_path),
                        provider="ddgs_web_image",
                        prompt_used=query,
                        attribution="Web high-res image (9:16 fitted)",
                    )
        except Exception as exc:
            logger.debug("DDGS web image search skipped for scene %d: %s", scene_num, exc)
        return None

    def _create_offline_fallback(self, scene_num: int, prompt: str, out_path: Path) -> GeneratedPhoto:
        width = int(self.config.video.get("width", 1080))
        height = int(self.config.video.get("height", 1920))
        scale = max(width, 1) / 1080.0
        img = Image.new("RGB", (width, height), (10, 12, 18))
        draw = ImageDraw.Draw(img)
        cx, cy = width // 2, int(height * 0.44)
        max_r = int(round(740 * scale))
        min_r = int(round(40 * scale))
        step_r = max(12, int(round(18 * scale)))
        for radius in range(max_r, min_r, -step_r):
            shade = int(10 + (1 - radius / max_r) * 32)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(shade, int(shade * 1.1), int(shade * 1.35)),
            )
        img = img.filter(ImageFilter.GaussianBlur(radius=max(2, int(round(2 * scale)))))
        img.save(out_path, format="JPEG", quality=92)
        return GeneratedPhoto(
            scene_number=scene_num,
            image_path=str(out_path),
            provider="offline_atmosphere",
            prompt_used=prompt,
            attribution="Offline atmospheric fallback",
        )

    def _generate_single(
        self, scene_num: int, prompt: str, dest_dir: Path, seed_offset: int = 0
    ) -> GeneratedPhoto:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"scene_{scene_num:02d}_photo.jpg"
        for provider in self.priority:
            if provider in self._disabled_providers:
                continue
            photo: GeneratedPhoto | None = None
            if provider == "ddgs_web_image":
                photo = self._try_ddgs_image(scene_num, prompt, out_path)
            elif provider == "offline_card":
                photo = self._create_offline_fallback(scene_num, prompt, out_path)
            if photo:
                return photo
        return self._create_offline_fallback(scene_num, prompt, out_path)

    @staticmethod
    def _fit_to_vertical_9x16(rgb: Image.Image, target: tuple[int, int] = (1080, 1920)) -> Image.Image:
        if rgb.size == target:
            return rgb
        s = max(target[0], 1) / 1080.0
        aspect = rgb.width / max(rgb.height, 1)
        if aspect <= 0.72:
            return ImageOps.fit(rgb, target, method=Image.Resampling.LANCZOS)

        bg = ImageOps.fit(rgb, target, method=Image.Resampling.LANCZOS).filter(
            ImageFilter.GaussianBlur(radius=max(20, int(round(30 * s))))
        )
        dark_overlay = Image.new("RGB", target, (8, 10, 14))

        if aspect <= 0.90:
            canvas = Image.blend(bg, dark_overlay, alpha=0.60)
            scale = min(target[0] / max(rgb.width, 1), (1720 * s) / max(rgb.height, 1))
            fg_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            fg = rgb.resize(fg_size, Image.Resampling.LANCZOS)
            fg_x = (target[0] - fg.width) // 2
            fg_y = max(int(round(80 * s)), (target[1] - fg.height) // 2 - int(round(40 * s)))
            canvas.paste(fg, (fg_x, fg_y))
            return canvas

        canvas = Image.blend(bg, dark_overlay, alpha=0.62)
        scale = min((1060 * s) / max(rgb.width, 1), (1180 * s) / max(rgb.height, 1))
        fg_size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        fg = rgb.resize(fg_size, Image.Resampling.LANCZOS)
        fg_x = (target[0] - fg.width) // 2
        fg_y = max(int(round(120 * s)), int(target[1] * 0.41) - fg.height // 2)
        canvas.paste(fg, (fg_x, fg_y))
        return canvas

    def _try_reuse_scene_photo(
        self, scene_num: int, prompt: str, reuse_dir: Path, dest_dir: Path
    ) -> GeneratedPhoto | None:
        if not reuse_dir.is_dir():
            return None
        target = (
            int(self.config.video.get("width", 1080)),
            int(self.config.video.get("height", 1920)),
        )
        name_patterns = (f"scene_{scene_num:02d}_photo", f"scene_{scene_num:02d}")
        for stem in name_patterns:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                candidate = reuse_dir / f"{stem}{ext}"
                if candidate.exists() and self._validate_image(candidate):
                    out_path = dest_dir / f"scene_{scene_num:02d}_photo.jpg"
                    with Image.open(candidate) as img:
                        rgb = img.convert("RGB")
                    fitted = self._fit_to_vertical_9x16(rgb, target=target)
                    fitted.save(out_path, format="JPEG", quality=93)
                    return GeneratedPhoto(
                        scene_number=scene_num,
                        image_path=str(out_path),
                        provider="preexisting_custom_ai",
                        prompt_used=prompt,
                        attribution=f"Staged visual asset ({candidate.name})",
                    )
        return None

    def run(
        self,
        plan: ScriptPlan,
        out_dir: Path,
        reuse_photos_dir: Path | None = None,
        extra_reuse_dirs: list[Path] | None = None,
    ) -> list[GeneratedPhoto]:
        photos_dir = out_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        candidate_dirs: list[Path] = []
        for d in ([reuse_photos_dir] if reuse_photos_dir is not None else []) + list(extra_reuse_dirs or []):
            if d is not None and d not in candidate_dirs:
                candidate_dirs.append(d)

        results: list[GeneratedPhoto] = []
        for scene in plan.scenes:
            prompt = scene.visual_query or scene.visual_ai_prompt or scene.voiceover_text
            reused: GeneratedPhoto | None = None
            for reuse_candidate in candidate_dirs:
                reused = self._try_reuse_scene_photo(
                    scene.scene_number, prompt, reuse_candidate, photos_dir
                )
                if reused:
                    logger.info(
                        "[Visuals - Photo] Reused staged image for scene %d/%d from %s.",
                        scene.scene_number,
                        len(plan.scenes),
                        reuse_candidate,
                    )
                    break
            if reused:
                results.append(reused)
                continue
            logger.info("[Visuals - Photo] Sourcing scene %d/%d...", scene.scene_number, len(plan.scenes))
            results.append(self._generate_single(scene.scene_number, prompt, photos_dir))
        (out_dir / "photos_manifest.json").write_text(
            json.dumps([p.__dict__ for p in results], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return results

    def regenerate_scenes(
        self,
        existing_photos: list[GeneratedPhoto],
        corrected_prompts: dict[int, str],
        out_dir: Path,
        revision_round: int,
    ) -> list[GeneratedPhoto]:
        """Re-source ONLY the scenes flagged by Vision Reviewer."""
        photos_dir = out_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)
        updated: list[GeneratedPhoto] = []
        for photo in existing_photos:
            if photo.scene_number in corrected_prompts:
                new_prompt = corrected_prompts[photo.scene_number]
                logger.info(
                    "[Visuals - Photo Revision #%d] Updating scene %d.",
                    revision_round,
                    photo.scene_number,
                )
                updated.append(
                    self._generate_single(
                        photo.scene_number,
                        new_prompt,
                        photos_dir,
                        seed_offset=revision_round * 100,
                    )
                )
            else:
                updated.append(photo)
        (out_dir / "photos_manifest.json").write_text(
            json.dumps([p.__dict__ for p in updated], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return updated
