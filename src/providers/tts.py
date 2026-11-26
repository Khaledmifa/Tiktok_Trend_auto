from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

from src.config import AppConfig
from src.models import WordTiming

logger = logging.getLogger(__name__)


class BaseTTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str, output_audio_path: Path) -> list[WordTiming]:
        """Generate voiceover audio and return word-level timing cues."""


def _estimate_word_timings(text: str, total_duration_sec: float) -> list[WordTiming]:
    """Fallback proportional word timing estimator if TTS boundary events are absent."""
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []
    weights = [max(len(re.sub(r"[^\w]", "", w)), 2) for w in words]
    total_weight = sum(weights) or len(words)
    cursor = 0.0
    timings: list[WordTiming] = []
    for word, weight in zip(words, weights):
        dur = (weight / total_weight) * total_duration_sec
        timings.append(WordTiming(text=word, start_sec=cursor, end_sec=cursor + dur))
        cursor += dur
    return timings


class EdgeTTSProvider(BaseTTSProvider):
    """100% Free Neural TTS via Microsoft Edge Read-Aloud service (edge-tts)."""

    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    async def _run_async(self, text: str, output_audio_path: Path) -> list[WordTiming]:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            boundary="WordBoundary",
        )
        timings: list[WordTiming] = []
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)

        with output_audio_path.open("wb") as audio_file:
            async for chunk in communicate.stream():
                chunk_type = chunk.get("type")
                if chunk_type == "audio":
                    audio_file.write(chunk["data"])
                elif chunk_type in ("WordBoundary", "SentenceBoundary"):
                    # edge-tts offsets and durations are in 100-nanosecond ticks
                    start_sec = float(chunk["offset"]) / 10_000_000.0
                    dur_sec = float(chunk.get("duration", 2_500_000)) / 10_000_000.0
                    chunk_text = str(chunk.get("text", "")).strip()
                    if not chunk_text:
                        continue
                    if chunk_type == "SentenceBoundary" and " " in chunk_text:
                        sub_words = chunk_text.split()
                        per_word = dur_sec / max(len(sub_words), 1)
                        for idx, w in enumerate(sub_words):
                            w_start = start_sec + idx * per_word
                            timings.append(
                                WordTiming(text=w, start_sec=w_start, end_sec=w_start + per_word)
                            )
                    else:
                        timings.append(
                            WordTiming(
                                text=chunk_text,
                                start_sec=start_sec,
                                end_sec=start_sec + max(dur_sec, 0.12),
                            )
                        )
        return timings

    def synthesize(self, text: str, output_audio_path: Path) -> list[WordTiming]:
        timings = asyncio.run(self._run_async(text, output_audio_path))
        if not timings:
            logger.warning("No boundary events emitted by edge-tts; using proportional timing fallback.")
            words = len(text.split())
            est_dur = max(words / 2.4, 1.0)
            timings = _estimate_word_timings(text, est_dur)
        return timings


def build_tts_provider(config: AppConfig, language: str) -> BaseTTSProvider:
    """Factory creating the configured free TTS provider."""
    provider_name = config.tts.get("provider", "edge_tts").lower()
    voices = config.tts.get("voices", {})
    default_voice = "ar-EG-ShakirNeural" if language == "ar" else "en-US-AndrewNeural"
    voice = str(voices.get(language, default_voice) or default_voice).strip()
    if language == "en" and "multilingual" in voice.lower():
        voice = re.sub(r"multilingual", "", voice, flags=re.IGNORECASE)
        if not voice.endswith("Neural"):
            voice = "en-US-AndrewNeural"
    rate = config.tts.get("rate", "+4%")
    pitch = config.tts.get("pitch", "+0Hz")

    if provider_name == "edge_tts":
        return EdgeTTSProvider(voice=voice, rate=rate, pitch=pitch)

    raise ValueError(f"Unsupported TTS provider '{provider_name}' in config.yaml")
