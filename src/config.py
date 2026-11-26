from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    """Centralized configuration loaded from config.yaml and environment variables."""

    root_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def project(self) -> dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.raw.get("pipeline", {})

    @property
    def models(self) -> dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def sound_design(self) -> dict[str, Any]:
        return self.raw.get("sound_design", {})

    @property
    def video(self) -> dict[str, Any]:
        return self.raw.get("video", {})

    @property
    def research(self) -> dict[str, Any]:
        return self.raw.get("research", {})

    @property
    def llm(self) -> dict[str, Any]:
        return self.raw.get("llm", {})

    @property
    def tts(self) -> dict[str, Any]:
        return self.raw.get("tts", {})

    @property
    def visuals(self) -> dict[str, Any]:
        return self.raw.get("visuals", {})

    @property
    def subtitles(self) -> dict[str, Any]:
        return self.raw.get("subtitles", {})

    @property
    def editorial_hud(self) -> dict[str, Any]:
        return self.raw.get("editorial_hud", {})

    @property
    def qa_and_publish(self) -> dict[str, Any]:
        return self.raw.get("qa_and_publish", {})

    @property
    def telegram(self) -> dict[str, Any]:
        return self.raw.get("telegram", {})

    @property
    def output_dir(self) -> Path:
        return self.tiktok_dir

    @property
    def tiktok_dir(self) -> Path:
        tt_cfg = self.qa_and_publish.get("tiktok", {})
        export_dir_name = str(
            self.project.get("runs_dir")
            or tt_cfg.get("export_dir")
            or "tiktok"
        )
        path = self.root_dir / export_dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_dir(self) -> Path:
        path = self.root_dir / self.project.get("data_dir", "data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def assets_dir(self) -> Path:
        path = self.root_dir / self.project.get("assets_dir", "assets")
        (path / "music").mkdir(parents=True, exist_ok=True)
        (path / "fonts").mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        path = self.root_dir / self.project.get("logs_dir", "logs")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def env(key: str, default: str = "") -> str:
        return os.getenv(key, default).strip()


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load .env and config.yaml into an AppConfig instance regardless of CWD."""
    cand = Path(config_path)
    if not cand.is_absolute() and not cand.exists():
        root_cand = PROJECT_ROOT / cand
        if root_cand.exists():
            cand = root_cand
    resolved = cand.resolve()
    root_dir = resolved.parent
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)

    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return AppConfig(root_dir=root_dir, raw=raw)
