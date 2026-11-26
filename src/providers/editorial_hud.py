from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ACCENT_LIME = (205, 255, 67)  # #CDFF43
BRAND_DARK = (11, 14, 20)
TEXT_WHITE = (248, 250, 252)
TEXT_MUTED = (184, 192, 204)
TEXT_LIGHT_GRAY = (203, 211, 223)
CARD_FILL = (14, 18, 26, 205)
CARD_BORDER = (175, 188, 208, 115)
TRACK_BG = (52, 59, 72, 220)


def _maybe_reshape_rtl(text: str) -> str:
    """Apply Arabic shaping + BiDi ordering when optional packages exist."""
    if not text or not any("\u0600" <= ch <= "\u06ff" for ch in text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _candidate_font_paths(assets_fonts_dir: Path | None = None, regular: bool = False) -> list[Path]:
    candidates: list[Path] = []
    if assets_fonts_dir and assets_fonts_dir.is_dir():
        patterns = ("*.ttf", "*.otf")
        for pat in patterns:
            for p in sorted(assets_fonts_dir.glob(pat)):
                low = p.name.lower()
                if regular and "bold" in low:
                    continue
                candidates.append(p)
        for pat in patterns:
            candidates.extend(sorted(assets_fonts_dir.glob(pat)))

    windir = os.environ.get("WINDIR", "").strip()
    if windir:
        win_fonts = Path(windir) / "Fonts"
        names = (
            ("arial.ttf", "segoeui.ttf", "tahoma.ttf", "arialbd.ttf")
            if regular
            else ("arialbd.ttf", "segoeuib.ttf", "tahomabd.ttf", "arial.ttf")
        )
        for name in names:
            candidates.append(win_fonts / name)

    linux_candidates = (
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        )
        if regular
        else (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        )
    )
    candidates.extend(linux_candidates)
    return candidates


@lru_cache(maxsize=64)
def _load_cached_font(path_str: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path_str:
        try:
            return ImageFont.truetype(path_str, size=max(8, int(size)))
        except Exception:
            pass
    return ImageFont.load_default()


def resolve_bold_font(
    assets_fonts_dir: Path | None = None,
    size: int = 57,
    regular: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Locate a crisp sans-serif font across Windows and Linux without hardcoded drive letters."""
    for cand in _candidate_font_paths(assets_fonts_dir=assets_fonts_dir, regular=regular):
        if cand.is_file():
            return _load_cached_font(str(cand), size)
    return _load_cached_font("", size)


def measure_text_width_px(
    text: str,
    font_size: int = 57,
    bold: bool = True,
    assets_fonts_dir: Path | None = None,
) -> int:
    """Measure pixel width of a single line using Pillow font metrics."""
    clean = _maybe_reshape_rtl((text or "").strip())
    if not clean:
        return 0
    font = resolve_bold_font(assets_fonts_dir=assets_fonts_dir, size=font_size, regular=not bold)
    try:
        bbox = font.getbbox(clean)
        return max(0, int(bbox[2] - bbox[0]))
    except Exception:
        return int(len(clean) * font_size * 0.52)


def wrap_caption_balanced(
    words: Iterable[str],
    font_size: int = 57,
    max_width_px: int = 790,
    assets_fonts_dir: Path | None = None,
    bold: bool = True,
) -> list[list[int]]:
    """Group word indices into 1 line or a balanced 2-line split under max_width_px."""
    word_list = [str(w).strip() for w in words if str(w).strip()]
    n = len(word_list)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    full_line = " ".join(word_list)
    if measure_text_width_px(full_line, font_size=font_size, bold=bold, assets_fonts_dir=assets_fonts_dir) <= max_width_px:
        return [list(range(n))]

    best_split = max(1, n // 2)
    best_cost: tuple[int, int, int] | None = None
    for split_idx in range(1, n):
        line1 = " ".join(word_list[:split_idx])
        line2 = " ".join(word_list[split_idx:])
        w1 = measure_text_width_px(line1, font_size=font_size, bold=bold, assets_fonts_dir=assets_fonts_dir)
        w2 = measure_text_width_px(line2, font_size=font_size, bold=bold, assets_fonts_dir=assets_fonts_dir)
        overflow = max(0, w1 - max_width_px) + max(0, w2 - max_width_px)
        cost = (overflow, max(w1, w2), abs(w1 - w2))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_split = split_idx

    return [list(range(0, best_split)), list(range(best_split, n))]


def _parse_hex_rgb(hex_color: str, fallback: tuple[int, int, int] = ACCENT_LIME) -> tuple[int, int, int]:
    raw = (hex_color or "").strip().lstrip("#")
    if len(raw) == 6:
        try:
            return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
        except ValueError:
            return fallback
    return fallback


def _fit_font_for_lines(
    lines: list[str],
    start_size: int,
    min_size: int,
    max_width_px: int,
    assets_fonts_dir: Path | None = None,
    bold: bool = True,
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int]:
    non_empty = [ln for ln in lines if ln.strip()]
    for sz in range(start_size, min_size - 1, -2):
        fits = all(
            measure_text_width_px(ln, font_size=sz, bold=bold, assets_fonts_dir=assets_fonts_dir) <= max_width_px
            for ln in non_empty
        )
        if fits:
            return resolve_bold_font(assets_fonts_dir=assets_fonts_dir, size=sz, regular=not bold), sz
    return resolve_bold_font(assets_fonts_dir=assets_fonts_dir, size=min_size, regular=not bold), min_size


def _draw_vertical_gradient(
    img: Image.Image,
    y_start: int,
    y_end: int,
    alpha_start: int,
    alpha_end: int,
    rgb: tuple[int, int, int] = (8, 10, 16),
) -> None:
    width = img.width
    span = max(1, y_end - y_start)
    overlay = Image.new("RGBA", (width, span), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for dy in range(span):
        t = dy / max(1, span - 1)
        alpha = int(round(alpha_start + (alpha_end - alpha_start) * t))
        draw.line([(0, dy), (width, dy)], fill=(rgb[0], rgb[1], rgb[2], max(0, min(255, alpha))))
    img.alpha_composite(overlay, (0, y_start))


def render_scene_hud_overlay(
    out_path: Path,
    *,
    width: int = 1080,
    height: int = 1920,
    brand_label: str = "VIRAL STUDIO PRO",
    accent_hex: str = "#CDFF43",
    kicker: str = "",
    headline_line1: str = "",
    headline_line2: str = "",
    footage_context: str = "OFFICIAL FOOTAGE",
    attribution_line: str = "Official trailers & media | Commentary & analysis",
    fact_card: str = "",
    scene_index: int = 1,
    total_scenes: int = 10,
    category_label: str = "AI EXPLAINED",
    assets_fonts_dir: Path | None = None,
    compact_context_badge: bool = False,
    show_fact_card_only_when_distinct: bool = False,
) -> Path:
    """Render a crisp 9:16 (1080x1920 or 2160x3840 4K) transparent Editorial HUD overlay PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accent_rgb = _parse_hex_rgb(accent_hex, ACCENT_LIME)
    scale = max(width, 1) / 1080.0

    def sc(val: int | float) -> int:
        return max(1, int(round(val * scale))) if val > 0 else 0

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Translucent dark gradients top & bottom so typography pops cleanly over any footage
    _draw_vertical_gradient(canvas, 0, int(height * 0.27), 225, 0)
    _draw_vertical_gradient(canvas, int(height * 0.56), height, 0, 238)

    draw = ImageDraw.Draw(canvas)
    safe_x = sc(82)
    safe_max_w = sc(840)  # Leaves >= 158px*scale clear on right for TikTok UI buttons

    # 1. Brand pill at top-left safe area
    brand_txt = _maybe_reshape_rtl((brand_label or "VIRAL STUDIO PRO").strip().upper())
    brand_font = resolve_bold_font(assets_fonts_dir=assets_fonts_dir, size=sc(23), regular=False)
    b_bbox = brand_font.getbbox(brand_txt)
    bw = max(sc(80), int(b_bbox[2] - b_bbox[0]))
    bh = max(sc(20), int(b_bbox[3] - b_bbox[1]))
    pill_x0, pill_y0 = safe_x, sc(165)
    pill_x1 = min(safe_x + bw + sc(34), safe_x + sc(420))
    pill_y1 = pill_y0 + bh + sc(20)
    draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=sc(8), fill=accent_rgb + (255,))
    draw.text((pill_x0 + sc(17), pill_y0 + sc(8)), brand_txt, font=brand_font, fill=BRAND_DARK + (255,))

    # 2. Kicker label
    kicker_clean = _maybe_reshape_rtl((kicker or "").strip().upper())
    y_cursor = pill_y1 + sc(26)
    if kicker_clean:
        k_font, _ = _fit_font_for_lines([kicker_clean], sc(25), sc(18), safe_max_w, assets_fonts_dir=assets_fonts_dir)
        draw.text((safe_x, y_cursor), kicker_clean, font=k_font, fill=accent_rgb + (255,))
        y_cursor += sc(38)

    # 3. 2-line Section Headline (white + Acid-Lime)
    h1 = _maybe_reshape_rtl((headline_line1 or "").strip().upper())
    h2 = _maybe_reshape_rtl((headline_line2 or "").strip().upper())
    h_font, h_sz = _fit_font_for_lines(
        [h1, h2], start_size=sc(64), min_size=sc(40), max_width_px=safe_max_w, assets_fonts_dir=assets_fonts_dir
    )
    line_step = int(round(h_sz * 1.16))
    if h1:
        draw.text(
            (safe_x, y_cursor),
            h1,
            font=h_font,
            fill=TEXT_WHITE + (255,),
            stroke_width=sc(2),
            stroke_fill=(0, 0, 0, 160),
        )
        y_cursor += line_step
    if h2:
        draw.text(
            (safe_x, y_cursor),
            h2,
            font=h_font,
            fill=accent_rgb + (255,),
            stroke_width=sc(2),
            stroke_fill=(0, 0, 0, 160),
        )

    # 4. Footage context & attribution (compact top-right badge mode vs legacy wide label)
    ctx_clean = _maybe_reshape_rtl((footage_context or "OFFICIAL FOOTAGE").strip().upper())
    attr_clean = _maybe_reshape_rtl((attribution_line or "").strip())
    if compact_context_badge:
        badge_font, _ = _fit_font_for_lines(
            [ctx_clean], start_size=sc(18), min_size=sc(14), max_width_px=sc(380), assets_fonts_dir=assets_fonts_dir
        )
        c_box = badge_font.getbbox(ctx_clean)
        cw = max(sc(60), int(c_box[2] - c_box[0]))
        ch = max(sc(16), int(c_box[3] - c_box[1]))
        bx1 = safe_x + safe_max_w
        bx0 = max(pill_x1 + sc(24), bx1 - cw - sc(28))
        by0 = pill_y0 + sc(2)
        by1 = by0 + ch + sc(16)
        ctx_up = ctx_clean.upper()
        if "SYNTHETIC" in ctx_up or "AI-GENERATED" in ctx_up:
            b_fill, b_outline, b_txt = (48, 30, 8, 225), (255, 196, 61, 240), (255, 214, 96, 255)
        elif "DEMO" in ctx_up or "ANNOUNCEMENT" in ctx_up or "RUMOR" in ctx_up:
            b_fill, b_outline, b_txt = (14, 34, 48, 225), (94, 212, 255, 235), (156, 232, 255, 255)
        else:
            b_fill, b_outline, b_txt = CARD_FILL, CARD_BORDER, TEXT_LIGHT_GRAY + (245,)
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=sc(8), fill=b_fill, outline=b_outline, width=sc(2))
        draw.text((bx0 + sc(14), by0 + sc(7)), ctx_clean, font=badge_font, fill=b_txt)
        if attr_clean:
            attr_font, _ = _fit_font_for_lines(
                [attr_clean], sc(19), sc(14), safe_max_w, assets_fonts_dir=assets_fonts_dir, bold=False
            )
            draw.text((safe_x, sc(1104)), attr_clean, font=attr_font, fill=TEXT_MUTED + (215,))
    else:
        ctx_font, _ = _fit_font_for_lines([ctx_clean], sc(24), sc(18), safe_max_w, assets_fonts_dir=assets_fonts_dir)
        draw.text((safe_x, sc(1106)), ctx_clean, font=ctx_font, fill=TEXT_WHITE + (245,))
        if attr_clean:
            attr_font, _ = _fit_font_for_lines(
                [attr_clean], sc(20), sc(15), safe_max_w, assets_fonts_dir=assets_fonts_dir, bold=False
            )
            draw.text((safe_x, sc(1140)), attr_clean, font=attr_font, fill=TEXT_MUTED + (230,))

    # 5. Compact Fact Card below captions (omitted when redundant in clean 3-layer mode)
    card_x0, card_x1 = safe_x, safe_x + safe_max_w
    card_y0, card_y1 = sc(1445), sc(1545)
    raw_card = (fact_card or "").strip()
    should_draw_card = True
    if show_fact_card_only_when_distinct:
        if not raw_card:
            should_draw_card = False
        else:
            head_tokens = {w.lower() for w in f"{headline_line1} {headline_line2}".split() if len(w) > 2}
            card_tokens = {w.lower() for w in raw_card.split() if len(w) > 2}
            if head_tokens and card_tokens:
                overlap = len(head_tokens & card_tokens) / max(len(card_tokens), 1)
                if overlap >= 0.60:
                    should_draw_card = False

    if should_draw_card:
        draw.rounded_rectangle(
            [card_x0, card_y0, card_x1, card_y1],
            radius=sc(16),
            fill=CARD_FILL,
            outline=CARD_BORDER,
            width=sc(2),
        )
        card_txt = _maybe_reshape_rtl(raw_card or "Verified editorial breakdown")
        card_font, card_sz = _fit_font_for_lines(
            [card_txt], start_size=sc(28), min_size=sc(19), max_width_px=safe_max_w - sc(52), assets_fonts_dir=assets_fonts_dir
        )
        c_bbox = card_font.getbbox(card_txt)
        ch = max(sc(16), int(c_bbox[3] - c_bbox[1]))
        text_y = card_y0 + max(sc(18), (card_y1 - card_y0 - ch) // 2 - sc(3))
        draw.text((card_x0 + sc(26), text_y), card_txt, font=card_font, fill=TEXT_WHITE + (255,))

    # 6. Section counter & Acid-Lime progress bar
    total_safe = max(1, int(total_scenes))
    idx_safe = max(1, min(total_safe, int(scene_index)))
    cat_clean = _maybe_reshape_rtl((category_label or "TECH EXPLAINED").strip().upper())
    ind_txt = f"{idx_safe:02d} / {total_safe:02d}  •  {cat_clean}"
    ind_font = resolve_bold_font(assets_fonts_dir=assets_fonts_dir, size=sc(22), regular=False)
    draw.text((safe_x, sc(1586)), ind_txt, font=ind_font, fill=TEXT_LIGHT_GRAY + (245,))

    bar_y0, bar_y1 = sc(1636), sc(1642)
    draw.rounded_rectangle([card_x0, bar_y0, card_x1, bar_y1], radius=sc(3), fill=TRACK_BG)
    fill_w = max(sc(16), int(round((card_x1 - card_x0) * (idx_safe / total_safe))))
    draw.rounded_rectangle([card_x0, bar_y0, card_x0 + fill_w, bar_y1], radius=sc(3), fill=accent_rgb + (255,))

    canvas.save(out_path, format="PNG")
    return out_path
