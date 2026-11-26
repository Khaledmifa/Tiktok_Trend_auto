from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path


def _write_wav_mono(path: Path, samples: list[float], sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for s in samples:
            clamped = max(-0.98, min(0.98, s))
            frames.extend(struct.pack("<h", int(clamped * 32767)))
        wf.writeframes(bytes(frames))
    return path


def generate_dark_ambient_bed(
    path: Path,
    duration_sec: float,
    sample_rate: int = 44100,
    seamless_loop: bool = False,
) -> Path:
    """Synthesize a moody, cinematic sub-bass & warm pad bed track (100% free/offline)."""
    total_samples = max(1, int(duration_sec * sample_rate))
    actual_dur = total_samples / sample_rate
    samples: list[float] = [0.0] * total_samples
    # Minor chord drone frequencies (D1, A1, F2, C3) for suspenseful TikTok storytelling
    freqs = (36.71, 55.00, 87.31, 130.81)
    amps = (0.36, 0.28, 0.16, 0.08)

    if seamless_loop:
        # Snap LFO, pulse, and harmonic frequencies to integer cycles so end joins start cleanly
        lfo_hz = max(1, round(0.14 * actual_dur)) / actual_dur
        pulse_hz = max(1, round(1.916 * actual_dur)) / actual_dur
        loop_freqs = tuple(max(1, round(f * actual_dur)) / actual_dur for f in freqs)
    else:
        lfo_hz = 0.14
        pulse_hz = 1.916
        loop_freqs = freqs

    edge_taper = 0.04
    for i in range(total_samples):
        t = i / sample_rate
        lfo = 0.72 + 0.28 * math.sin(2.0 * math.pi * lfo_hz * t)
        pulse = 0.85 + 0.15 * math.sin(2.0 * math.pi * pulse_hz * t)
        val = 0.0
        for idx, (f, a) in enumerate(zip(loop_freqs, amps)):
            detune = (
                1.0
                if seamless_loop
                else (1.0 + (0.0025 if idx % 2 == 0 else -0.002))
            )
            val += a * math.sin(2.0 * math.pi * f * detune * t)

        fade = 1.0
        if seamless_loop:
            if t < edge_taper:
                fade = t / edge_taper
            elif t > actual_dur - edge_taper:
                fade = max(0.0, (actual_dur - t) / edge_taper)
        else:
            if t < 0.8:
                fade = t / 0.8
            elif t > duration_sec - 1.2:
                fade = max(0.0, (duration_sec - t) / 1.2)

        samples[i] = val * lfo * pulse * fade * 0.55

    return _write_wav_mono(path, samples, sample_rate=sample_rate)


def generate_bass_hit_sfx(path: Path, sample_rate: int = 44100) -> Path:
    """Generate a punchy 0.85s cinematic sub-drop impact for hooks & twists."""
    dur = 0.85
    total = int(dur * sample_rate)
    samples: list[float] = []
    phase = 0.0
    for i in range(total):
        t = i / sample_rate
        freq = 115.0 * math.exp(-4.2 * t) + 34.0
        phase += 2.0 * math.pi * freq / sample_rate
        env = math.exp(-4.5 * t)
        attack = min(1.0, t / 0.008)
        samples.append(math.sin(phase) * env * attack * 0.85)
    return _write_wav_mono(path, samples, sample_rate=sample_rate)


def generate_whoosh_sfx(path: Path, sample_rate: int = 44100) -> Path:
    """Generate a smooth 0.38s filtered-noise air whoosh for rapid scene cuts."""
    dur = 0.38
    total = int(dur * sample_rate)
    rng = random.Random(42)
    samples: list[float] = []
    prev = 0.0
    for i in range(total):
        t = i / sample_rate
        # Bell envelope peaking at 45% of duration
        env = math.sin(math.pi * (t / dur)) ** 1.8
        # Sweeping simple low-pass alpha for air-rush contour
        alpha = 0.04 + 0.22 * math.sin(math.pi * (t / dur))
        white = rng.uniform(-1.0, 1.0)
        prev = prev + alpha * (white - prev)
        samples.append(prev * env * 0.65)
    return _write_wav_mono(path, samples, sample_rate=sample_rate)
