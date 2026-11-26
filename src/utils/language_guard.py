from __future__ import annotations

import logging
import re
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_GERMAN_STOPWORDS: set[str] = {
    "der", "die", "das", "und", "ist", "ein", "eine", "einer", "einem", "einen",
    "mit", "auf", "von", "sich", "nicht", "werden", "wird", "aus", "auch", "als",
    "dem", "den", "des", "im", "am", "zum", "zur", "jahr", "film", "actionfilm",
    "gleichen", "namens", "basiert", "staffel", "folge", "serie", "serien", "regie",
    "über", "nach", "bei", "für", "wie", "oder", "aber", "dass", "sind", "war",
}
_FR_ES_IT_PT_STOPWORDS: set[str] = {
    "le", "la", "les", "une", "est", "dans", "pour", "avec", "sur", "qui", "sont",
    "el", "los", "las", "una", "del", "con", "por", "para", "que", "es", "como",
    "il", "gli", "alla", "degli", "che", "sono", "della", "nella",
    "os", "uma", "com", "não", "mais", "dos", "das",
}
_NON_EN_STOPWORDS = _GERMAN_STOPWORDS | _FR_ES_IT_PT_STOPWORDS

_NON_EN_DIACRITIC_RE = re.compile(r"[äöüßÄÖÜàâæçéèêëîïôœùûüÿñáíóú¿¡ãõ]")
_NON_LATIN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF\u0600-\u06FF\u0590-\u05FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0900-\u097F]"
)
_SRT_TIMING_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


def is_english_text(text: str) -> bool:
    """Return True when text appears to be English without foreign prose or non-Latin scripts."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return False
    if _NON_EN_DIACRITIC_RE.search(clean):
        return False
    if _NON_LATIN_SCRIPT_RE.search(clean):
        return False
    tokens = re.findall(r"[a-zA-ZäöüßÄÖÜ]+", clean.lower())
    if not tokens:
        return True
    non_en_hits = sum(1 for tok in tokens if tok in _NON_EN_STOPWORDS)
    if non_en_hits >= 2 and (non_en_hits / max(len(tokens), 1)) >= 0.14:
        return False
    return True


def translate_to_english(text: str, timeout: float = 8.0) -> str | None:
    """Translate non-English text into English using a $0 key-free endpoint; return None on failure."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return ""
    if is_english_text(clean):
        return clean
    try:
        resp = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "auto", "tl": "en", "dt": "t", "q": clean},
            headers={"User-Agent": "Mozilla/5.0 ViralStudioPro/1.0"},
            timeout=timeout,
        )
        if not resp.ok:
            return None
        data = resp.json()
        if not isinstance(data, list) or not data or not isinstance(data[0], list):
            return None
        translated = "".join(str(seg[0]) for seg in data[0] if isinstance(seg, list) and seg and seg[0]).strip()
        translated = re.sub(r"\s+", " ", translated).strip()
        if translated and is_english_text(translated):
            return translated
    except Exception as exc:
        logger.debug("Auto-translation skipped safely: %s", exc)
    return None


def ensure_english_text(text: str, fallback: str = "", allow_translate: bool = True) -> str:
    """Ensure text is English: keep if English, translate non-English at $0 cost, else return fallback."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return fallback
    if is_english_text(clean):
        return clean
    if allow_translate:
        translated = translate_to_english(clean)
        if translated:
            return translated
    return fallback


def is_subtitle_file_english(srt_path: Path) -> tuple[bool, list[str]]:
    """Inspect non-timestamp cues in an .srt file and return (all_english, offending_cues)."""
    if not srt_path.exists():
        return True, []
    content = srt_path.read_text(encoding="utf-8", errors="replace")
    offending: list[str] = []
    current_cue_lines: list[str] = []

    def _check_cue() -> None:
        if not current_cue_lines:
            return
        cue_text = " ".join(current_cue_lines).strip()
        if cue_text and not is_english_text(cue_text):
            offending.append(cue_text)
        current_cue_lines.clear()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            _check_cue()
            continue
        if line.isdigit() or _SRT_TIMING_RE.match(line):
            continue
        current_cue_lines.append(line)
    _check_cue()
    return len(offending) == 0, offending
