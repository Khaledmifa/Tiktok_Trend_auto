from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ViralOpportunity:
    """Content opportunity discovered by AI Model 1 (Research)."""

    topic: str
    unique_angle: str
    target_audience: str
    hook_pattern: str
    narrative_structure: str
    viral_triggers: list[str]
    keywords: list[str]
    source_signals: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ResearchReport:
    """Deliverable produced by AI Model 1 (Research AI)."""

    run_id: str
    niche: str
    created_at: str
    analyzed_patterns: dict[str, Any]
    opportunities: list[ViralOpportunity]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScriptScene:
    """Single visual beat in the 65-85s script."""

    scene_number: int
    estimated_duration_sec: float
    voiceover_text: str
    visual_query: str
    visual_ai_prompt: str
    video_motion_prompt: str = "zoom_in"
    on_screen_focus: str = ""
    kicker: str = ""
    headline_line1: str = ""
    headline_line2: str = ""
    fact_card: str = ""
    layout_mode: str = "auto"  # "auto" | "wide_panel" | "fullscreen_portrait"
    footage_context: str = ""


@dataclass
class ScriptPlan:
    """Deliverable produced by AI Model 2 (Script AI)."""

    run_id: str
    topic: str
    unique_angle: str
    language: str
    opening_hook_3s: str
    closing_cta: str
    full_voiceover_text: str
    total_estimated_duration_sec: float
    word_count: int
    scenes: list[ScriptScene]
    category_label: str = "TECH EXPLAINED"
    footage_context_label: str = "OFFICIAL FOOTAGE"
    attribution_line: str = "Official trailers & media | Commentary & analysis"
    fact_sources: list[str] = field(default_factory=list)
    video_sources: list[str] = field(default_factory=list)
    loop_bridge_phrase: str = ""
    loop_joined_preview: str = ""
    loop_cut_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedPhoto:
    """Deliverable per scene from AI Model 3 (Photo Generation AI)."""

    scene_number: int
    image_path: str
    provider: str
    prompt_used: str
    attribution: str


@dataclass
class GeneratedVideoClip:
    """Deliverable per scene from AI Model 4 (Video Generation / Motion AI)."""

    scene_number: int
    clip_path: str
    duration_sec: float
    provider: str
    motion_style: str


@dataclass
class WordTiming:
    """Timestamped word cue for kinetic subtitle generation."""

    text: str
    start_sec: float
    end_sec: float


@dataclass
class MediaManifest:
    """Deliverable produced by AI Model 5 (Montage Agent)."""

    run_id: str
    audio_path: str
    srt_path: str
    ass_path: str
    final_video_path: str
    actual_duration_sec: float
    resolution: str
    fps: int
    visual_sources_used: list[dict[str, str]]
    has_background_music: bool
    cover_path: str = ""
    audio_metrics: dict[str, Any] = field(default_factory=dict)
    loop_qa_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneAuditItem:
    """Per-scene evaluation from AI Model 6 (Vision Reviewer)."""

    scene_number: int
    matches_script: bool
    score: int
    issue: str = ""
    fixed_image_prompt: str = ""
    fixed_motion_prompt: str = ""


@dataclass
class ReviewerVerdict:
    """Structured review & feedback-routing decision from AI Model 6."""

    verdict: str  # "PASS" | "REVISION_REQUIRED"
    responsible_agent: str  # "none" | "research" | "script" | "photo" | "video" | "montage"
    error_diagnosis: str
    flagged_scenes: list[int] = field(default_factory=list)
    scene_audits: list[SceneAuditItem] = field(default_factory=list)
    montage_fixes: dict[str, Any] = field(default_factory=dict)
    revision_round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QACheckResult:
    """Automated technical + AI reviewer verification."""

    passed: bool
    duration_ok: bool
    audio_present: bool
    captions_synced: bool
    originality_ok: bool
    max_past_similarity: float
    ai_review_verdict: str = "PASS"
    issues: list[str] = field(default_factory=list)


@dataclass
class PlatformMetadata:
    """Tailored publishing metadata for TikTok Creator Rewards."""

    tiktok_title: str
    tiktok_caption: str
    tiktok_description: str
    tiktok_hashtags: list[str]
    tiktok_disclose_ai_content: bool


@dataclass
class ReviewPackage:
    """Final deliverable before the Human-in-the-Loop Gate."""

    run_id: str
    status: str  # READY_FOR_REVIEW | QA_FAILED | APPROVED_AND_PACKAGED
    created_at: str
    topic: str
    unique_angle: str
    video_path: str
    srt_path: str
    duration_sec: float
    qa: QACheckResult
    metadata: PlatformMetadata
    review_markdown_path: str
    reviewer_verdict: ReviewerVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
