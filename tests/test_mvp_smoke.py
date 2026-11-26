from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.agents.montage_agent import MontageAgent
from src.agents.photo_agent import PhotoGeneratorAgent
from src.agents.publisher_agent import QAPublisherAgent
from src.agents.research_agent import TrendResearchAgent
from src.agents.reviewer_agent import AIReviewerAgent
from src.agents.script_agent import ScriptArchitectAgent
from src.agents.video_agent import VideoGeneratorAgent
from src.config import AppConfig, load_config
from src.models import (
    GeneratedPhoto,
    GeneratedVideoClip,
    MediaManifest,
    ScriptPlan,
    ScriptScene,
    ViralOpportunity,
    WordTiming,
)
from src.providers.llm import LocalOllamaProvider
from src.providers.sound_design import (
    generate_bass_hit_sfx,
    generate_dark_ambient_bed,
    generate_whoosh_sfx,
)
from src.state.memory_store import MemoryStore, jaccard_similarity


class TestSixModelPipeline(unittest.TestCase):
    def test_jaccard_similarity_detects_duplicates(self) -> None:
        a = "In August 1977 a radio telescope printed six characters called the Wow signal"
        b = "In August 1977 a radio telescope printed six characters called the Wow signal again"
        c = "Roman concrete self heals cracks using hot mixed lime clasts underwater"
        self.assertGreater(jaccard_similarity(a, b), 0.7)
        self.assertLess(jaccard_similarity(a, c), 0.2)

    def test_memory_store_lifecycle_and_dedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.record_run_start("run_001", "Science Mysteries")
            store.update_run_script(
                "run_001",
                topic="The 1977 Wow! Signal Mystery",
                hook="In August 1977, a printer spat out six characters.",
                script_text="In August 1977 a radio telescope detected a 72 second anomaly.",
            )
            store.update_run_review_ready(
                "run_001",
                video_path="output/run_001/final_9x16.mp4",
                duration_sec=74.5,
                status="READY_FOR_REVIEW",
            )
            run = store.get_run("run_001")
            self.assertIsNotNone(run)
            assert run is not None
            self.assertEqual(run["status"], "READY_FOR_REVIEW")
            self.assertTrue((Path(tmp) / "history_log.json").exists())

    def test_sound_design_generates_valid_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            bed = generate_dark_ambient_bed(tdir / "bed.wav", duration_sec=2.0)
            hit = generate_bass_hit_sfx(tdir / "hit.wav")
            whoosh = generate_whoosh_sfx(tdir / "whoosh.wav")
            self.assertGreater(bed.stat().st_size, 10000)
            self.assertGreater(hit.stat().st_size, 10000)
            self.assertGreater(whoosh.stat().st_size, 5000)

    def test_kinetic_ass_captions_and_none_margin_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"width": 1080, "height": 1920},
                    "subtitles": {
                        "font_name": "Arial Black",
                        "font_size": 74,
                        "uppercase": True,
                        "active_word_highlight": True,
                        "primary_color": "&H00FFFFFF",
                        "highlight_color": "&H0000EAFF",
                        "power_color": "&H003B4DFF",
                        "outline_color": "&H00000000",
                        "margin_v": 520,
                        "words_per_caption_chunk": 2,
                    },
                },
            )
            agent = MontageAgent(cfg)
            timings = [
                WordTiming("Secret", 0.0, 0.4),
                WordTiming("signal", 0.4, 0.8),
                WordTiming("broke", 0.8, 1.2),
                WordTiming("silence", 1.2, 1.7),
            ]
            srt_path = tdir / "captions.srt"
            ass_path = tdir / "captions.ass"
            agent.build_captions(timings, 2.0, srt_path, ass_path, margin_v_override=None)

            ass_text = ass_path.read_text(encoding="utf-8")
            self.assertIn("Arial Black", ass_text)
            self.assertIn("\\c&H003B4DFF", ass_text)
            self.assertIn("\\c&H0000EAFF", ass_text)
            self.assertIn("\\fscx114", ass_text)
            self.assertIn(",520,1", ass_text)

    def test_montage_assemble_none_overrides_and_44k_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"width": 1080, "height": 1920, "fps": 30},
                    "sound_design": {"enabled": True, "bed_volume": 0.14, "sfx_volume": 0.30},
                    "subtitles": {"enabled": True, "burn_in": True, "margin_v": 520},
                },
            )
            img_path = tdir / "sample.jpg"
            Image.new("RGB", (1080, 1920), (30, 40, 60)).save(img_path)
            photo = GeneratedPhoto(1, str(img_path), "test", "prompt", "test")
            video_agent = VideoGeneratorAgent(cfg)
            clips_dir = tdir / "clips"
            clips_dir.mkdir()
            clip = video_agent._render_ai_motion_clip(photo, "zoom_in", 1.5, clips_dir / "clip_01.mp4")
            self.assertTrue(Path(clip.clip_path).exists())
            self.assertGreater(Path(clip.clip_path).stat().st_size, 10000)

            audio_path = generate_dark_ambient_bed(tdir / "voiceover.wav", duration_sec=1.4)
            montage = MontageAgent(cfg)
            manifest = montage.assemble(
                run_id="test_run",
                audio_path=audio_path,
                timings=[WordTiming("Hello", 0.0, 0.7), WordTiming("world", 0.7, 1.3)],
                total_duration_sec=1.5,
                photos=[photo],
                clips=[clip],
                out_dir=tdir,
                montage_overrides={"margin_v": None, "bed_volume": None},
            )
            self.assertTrue(Path(manifest.final_video_path).exists())
            self.assertGreater(manifest.actual_duration_sec, 1.0)

    def test_photo_circuit_breaker_and_reuse_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            reuse_dir = tdir / "custom_photos"
            reuse_dir.mkdir()
            sample_img = Image.new("RGB", (720, 1280))
            sample_img.putdata([((x * 3) % 255, (y * 5) % 255, (x + y) % 255) for y in range(1280) for x in range(720)])
            sample_img.save(reuse_dir / "scene_01_photo.png")

            cfg = AppConfig(
                root_dir=tdir,
                raw={"models": {"photo_provider_priority": ["offline_card"]}},
            )
            agent = PhotoGeneratorAgent(cfg)
            agent._disabled_providers.add("ddgs_web_image")
            out_photo = agent._generate_single(2, "test prompt", tdir / "out")
            self.assertEqual(out_photo.provider, "offline_atmosphere")

            plan = ScriptPlan(
                run_id="r_reuse",
                topic="Test",
                unique_angle="Angle",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Test",
                total_estimated_duration_sec=3.0,
                word_count=1,
                scenes=[ScriptScene(1, 3.0, "Scene 1", "q", "prompt")],
            )
            reused = agent.run(plan, tdir / "run_out", reuse_photos_dir=reuse_dir)
            self.assertEqual(reused[0].provider, "preexisting_custom_ai")
            self.assertTrue(Path(reused[0].image_path).exists())

    def test_llm_provider_fallback_deduplication(self) -> None:
        provider = LocalOllamaProvider(
            model="qwen2.5:7b",
            fallback_models=["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "llama3.1:8b"],
        )
        self.assertEqual(
            provider._ordered_models(),
            ["qwen2.5:7b", "llama3.1:8b", "mistral:7b"],
        )

    def test_reviewer_routes_offline_photo_back_to_photo_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"hard_monetization_floor_sec": 61},
                    "models": {"photo_provider_priority": ["offline_card"]},
                },
            )
            reviewer = AIReviewerAgent(cfg)
            plan = ScriptPlan(
                run_id="r_test",
                topic="Acoustic Kitty",
                unique_angle="Spy cat",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Test",
                total_estimated_duration_sec=65.0,
                word_count=10,
                scenes=[
                    ScriptScene(1, 3.0, "CIA spy cat", "spy cat", "1966 spy cat"),
                    ScriptScene(2, 3.0, "Soviet embassy", "embassy", "1966 embassy"),
                ],
            )
            manifest = MediaManifest(
                run_id="r_test",
                audio_path="a.mp3",
                srt_path="c.srt",
                ass_path="c.ass",
                final_video_path="v.mp4",
                actual_duration_sec=66.0,
                resolution="1080x1920",
                fps=30,
                visual_sources_used=[],
                has_background_music=True,
            )
            photos = [
                GeneratedPhoto(1, "p1.jpg", "ddgs_web_image", "1966 spy cat", "Web"),
                GeneratedPhoto(2, "p2.jpg", "offline_atmosphere", "1966 embassy", "Offline"),
            ]
            verdict = reviewer._heuristic_fallback_review(plan, manifest, photos, revision_round=0)
            self.assertEqual(verdict.verdict, "REVISION_REQUIRED")
            self.assertEqual(verdict.responsible_agent, "photo")
            self.assertEqual(verdict.flagged_scenes, [2])

            # Scene 2 passes when backed by live B-roll clip
            live_clip = GeneratedVideoClip(2, "c2.mp4", 6.0, "preexisting_custom_video", "live_broll")
            v_live = reviewer._heuristic_fallback_review(
                plan, manifest, photos, revision_round=0, clips=[live_clip]
            )
            self.assertEqual(v_live.verdict, "PASS")

            photo_agent = PhotoGeneratorAgent(cfg)
            updated = photo_agent.regenerate_scenes(
                photos, {2: "1966 Soviet embassy stone gate"}, tdir, revision_round=1
            )
            self.assertEqual(len(updated), 2)
            self.assertTrue(Path(updated[1].image_path).exists())

    def test_research_agent_honors_live_trend_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(root_dir=tdir, raw={"llm": {"provider": "offline"}})
            store = MemoryStore(tdir)
            agent = TrendResearchAgent(cfg, store)
            out = agent._offline_opportunities(
                niche="Gaming & Pop Culture",
                signals=[],
                past_topics=[],
                trend_topic="Grand Theft Auto VI Launches",
            )
            opps = out["opportunities"]
            self.assertTrue(len(opps) >= 1)
            self.assertEqual(opps[0]["topic"], "Grand Theft Auto VI Launches")
            self.assertEqual(opps[0]["score"], 0.98)

    def test_project_config_tiktok_defaults_aligned(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
        raw_text = cfg_path.read_text(encoding="utf-8")
        cfg = load_config(cfg_path)
        self.assertEqual(cfg.project.get("platform"), "tiktok")
        self.assertEqual(cfg.project.get("runs_dir"), "tiktok")
        self.assertEqual(cfg.output_dir, cfg.tiktok_dir)
        self.assertNotIn("youtube", cfg.qa_and_publish)
        self.assertNotIn("gemini", raw_text.lower())
        self.assertNotIn("pollinations", raw_text.lower())
        self.assertNotIn("dark psychology", raw_text.lower())
        self.assertEqual(cfg.pipeline.get("default_scenes"), 10)
        self.assertEqual(cfg.pipeline.get("max_scenes"), 12)
        self.assertEqual(cfg.pipeline.get("target_words_en"), [155, 185])
        self.assertEqual(cfg.pipeline.get("max_visual_prompt_words"), 15)
        self.assertIs(cfg.pipeline.get("auto_harvest_ytdlp"), True)
        self.assertEqual(cfg.tts.get("rate"), "+4%")
        self.assertEqual(cfg.video.get("hard_monetization_floor_sec"), 61)
        self.assertIs(cfg.qa_and_publish.get("require_human_approval"), True)
        self.assertEqual(cfg.qa_and_publish.get("tiktok", {}).get("export_dir"), "tiktok")

        env_ex = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("GEMINI_API_KEY", env_ex)
        self.assertNotIn("POLLINATIONS_API_KEY", env_ex)
        self.assertNotIn("PEXELS_API_KEY", env_ex)

    def test_auto_top_trend_ranks_and_filters_copyright_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "qa_and_publish": {"max_similarity_threshold": 0.55},
                    "research": {
                        "blocked_copyright_keywords": ["official trailer", "leaked clip", "full movie"]
                    },
                },
            )
            store = MemoryStore(tdir)
            store.record_run_start("run_prev", "Tech & AI")
            store.update_run_script(
                "run_prev",
                topic="Quantum Chip Breakthrough Explained",
                hook="Scientists just built a quantum chip.",
                script_text="Scientists built a quantum chip that solves errors fast.",
            )

            agent = TrendResearchAgent(cfg, store, skip_external_llm=True)
            signals = [
                {
                    "source": "google_trends_rss",
                    "title": "Official Trailer Leaked Clip Full Movie",
                    "summary": "movie scene leak",
                    "approx_traffic": 2_000_000,
                },
                {
                    "source": "google_trends_rss",
                    "title": "Quantum Chip Breakthrough Explained",
                    "summary": "quantum chip solves errors",
                    "approx_traffic": 900_000,
                },
                {
                    "source": "google_trends_rss",
                    "title": "Sleep Debt Neuroscience Secret",
                    "summary": "why brain crashes after poor sleep",
                    "approx_traffic": 50_000,
                },
                {
                    "source": "google_trends_rss",
                    "title": "Solar Storm Grid Blackout Warning",
                    "summary": "explained why geomagnetic flare matters",
                    "approx_traffic": 500_000,
                },
            ]
            out_dir = tdir / "out"
            discovery = agent.discover_top_tiktok_trend(signals, out_dir)
            self.assertEqual(
                discovery["selected_top_trend"]["topic"],
                "Solar Storm Grid Blackout Warning",
            )
            self.assertEqual(len(discovery["filtered_out_copyright"]), 1)
            self.assertEqual(len(discovery["filtered_out_memory_duplicate"]), 1)
            self.assertTrue((out_dir / "trend_discovery.json").exists())

    def test_script_agent_load_from_json_and_10_scene_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(root_dir=tdir, raw={"project": {"language": "en"}})
            store = MemoryStore(tdir)
            store.record_run_start("run_10s", "Science")
            agent = ScriptArchitectAgent(cfg, store, skip_external_llm=True)

            raw_in = {
                "topic": "Why Planes Avoid Flying Over Tibet",
                "unique_angle": "High-altitude oxygen physics and emergency descent terrain.",
                "scenes": [
                    {
                        "voiceover_text": f"Sentence number {i} explains why high altitude mountain peaks force planes to reroute safely.",
                        "visual_ai_prompt": f"Commercial jet flying over snowy mountain ridge scene {i}, dark cinematic",
                    }
                    for i in range(1, 11)
                ],
            }
            in_path = tdir / "input_script.json"
            in_path.write_text(json.dumps(raw_in), encoding="utf-8")
            out_dir = tdir / "out"

            plan = agent.load_from_json("run_10s", in_path, out_dir)
            self.assertEqual(len(plan.scenes), 10)
            self.assertEqual(plan.scenes[0].scene_number, 1)
            self.assertTrue(bool(plan.scenes[0].video_motion_prompt))
            self.assertTrue((out_dir / "script_plan.json").exists())
            self.assertTrue((out_dir / "script_plan.md").exists())

            fb = agent._build_topic_script(
                ViralOpportunity(
                    "The Rings of Power Season 3 — Sauron Unmasked",
                    "Major production reveal and Sauron plot shift in Season 3.",
                    "Adults",
                    "Hook",
                    "Arc",
                    [],
                    ["rings of power season 3"],
                ),
                "en",
                facts=["Amazon confirmed Rings of Power Season 3 enters a darker forge-of-the-ring storyline."],
            )
            self.assertEqual(len(fb["scenes"]), 10)
            joined = " ".join(s["voiceover_text"] for s in fb["scenes"]).lower()
            self.assertIn("rings of power", joined)
            for forbidden in ("1977", "wow", "telescope", "acoustic kitty", "zeigarnik"):
                self.assertNotIn(forbidden, joined)

    def test_photo_inplace_reuse_and_short_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            out_dir = tdir / "run_x"
            photos_dir = out_dir / "photos"
            photos_dir.mkdir(parents=True)

            img1 = Image.new("RGB", (720, 1280))
            img1.putdata([((x * 3) % 255, (y * 5) % 255, (x + y) % 255) for y in range(1280) for x in range(720)])
            img1.save(photos_dir / "scene_01_photo.jpg")
            img2 = Image.new("RGB", (800, 1400))
            img2.putdata([((x * 7) % 255, (y * 3) % 255, (x * y) % 255) for y in range(1400) for x in range(800)])
            img2.save(photos_dir / "scene_02.png")

            cfg = AppConfig(root_dir=tdir, raw={"models": {"photo_provider_priority": ["offline_card"]}})
            agent = PhotoGeneratorAgent(cfg)
            plan = ScriptPlan(
                run_id="run_x",
                topic="Inplace Reuse",
                unique_angle="Angle",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="One two",
                total_estimated_duration_sec=10.0,
                word_count=2,
                scenes=[
                    ScriptScene(1, 5.0, "Scene 1", "q1", "prompt 1"),
                    ScriptScene(2, 5.0, "Scene 2", "q2", "prompt 2"),
                ],
            )
            res = agent.run(plan, out_dir, reuse_photos_dir=photos_dir)
            self.assertEqual([p.provider for p in res], ["preexisting_custom_ai", "preexisting_custom_ai"])
            with Image.open(res[1].image_path) as out_im:
                self.assertEqual(out_im.size, (1080, 1920))

    def test_tiktok_only_publisher_and_frames_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"width": 1080, "height": 1920, "fps": 30, "hard_monetization_floor_sec": 61, "max_duration_sec": 85},
                    "sound_design": {"enabled": False},
                    "subtitles": {"enabled": True, "burn_in": True, "margin_v": 520},
                    "qa_and_publish": {
                        "require_human_approval": True,
                        "ai_generated_disclosure": True,
                        "max_similarity_threshold": 0.55,
                        "tiktok": {"export_bundle": True, "disclose_ai_content": True},
                    },
                },
            )
            store = MemoryStore(tdir)
            store.record_run_start("run_tt", "Tech")
            store.update_run_script("run_tt", "TikTok Only Topic", "Hook", "Full script")

            out_dir = tdir / "output" / "run_tt"
            (out_dir / "clips").mkdir(parents=True)
            img_path = out_dir / "s1.jpg"
            Image.new("RGB", (1080, 1920), (25, 35, 55)).save(img_path)
            photo = GeneratedPhoto(1, str(img_path), "preexisting_custom_ai", "dark server room", "AI")
            clip = VideoGeneratorAgent(cfg)._render_ai_motion_clip(photo, "zoom_in", 2.0, out_dir / "clips" / "clip_01.mp4")

            audio_path = generate_dark_ambient_bed(out_dir / "voiceover.wav", duration_sec=2.0)
            montage = MontageAgent(cfg)
            manifest = montage.assemble(
                run_id="run_tt",
                audio_path=audio_path,
                timings=[WordTiming("Hello", 0.0, 0.8), WordTiming("TikTok", 0.8, 1.8)],
                total_duration_sec=2.0,
                photos=[photo],
                clips=[clip],
                out_dir=out_dir,
            )
            manifest.actual_duration_sec = 65.0

            plan = ScriptPlan(
                run_id="run_tt",
                topic="TikTok Only Topic",
                unique_angle="Clean TikTok angle",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Hello TikTok",
                total_estimated_duration_sec=65.0,
                word_count=2,
                scenes=[ScriptScene(1, 65.0, "Hello TikTok", "server", "dark server room")],
            )
            reviewer = AIReviewerAgent(cfg, skip_external_llm=True)
            verdict = reviewer.review(plan, manifest, [photo], [clip], out_dir, revision_round=0)
            self.assertEqual(verdict.verdict, "PASS")
            frames_manifest = out_dir / "review_frames" / "frames_manifest.json"
            self.assertTrue(frames_manifest.exists())

            pub = QAPublisherAgent(cfg, store, skip_external_llm=True)
            pkg = pub.prepare_review_package(
                run_id="run_tt",
                plan=plan,
                manifest=manifest,
                out_dir=out_dir,
                reviewer_verdict=verdict,
                metadata_override={
                    "tiktok_title": "Custom TikTok Title",
                    "tiktok_caption": "Custom Caption",
                    "tiktok_description": "Custom Desc",
                    "tiktok_hashtags": ["tiktok", "tech"],
                },
            )
            self.assertEqual(pkg.status, "READY_FOR_REVIEW")
            self.assertEqual(pkg.metadata.tiktok_title, "Custom TikTok Title")
            self.assertTrue(pkg.metadata.tiktok_disclose_ai_content)
            md_text = Path(pkg.review_markdown_path).read_text(encoding="utf-8")
            self.assertNotIn("YouTube Shorts", md_text)

            export_res = pub.approve_and_publish("run_tt")
            self.assertEqual(export_res["platform"], "tiktok")
            bundle_dir = Path(export_res["tiktok_bundle_dir"])
            self.assertEqual(bundle_dir, tdir / "tiktok" / "run_tt")
            self.assertTrue((bundle_dir / "final_9x16.mp4").exists())
            details = bundle_dir / "tiktok_post_details.txt"
            self.assertTrue(details.exists())
            self.assertIn("Custom TikTok Title", details.read_text(encoding="utf-8"))

    def test_photo_upscales_vertical_poster_to_full_width_and_reviewer_checks_dims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            poster = Image.new("RGB", (647, 823), (220, 40, 40))
            fitted = PhotoGeneratorAgent._fit_to_vertical_9x16(poster)
            self.assertEqual(fitted.size, (1080, 1920))
            # Center horizontal strip must have full-width bright red pixels (not 647px centered)
            left_px = fitted.getpixel((5, 960))
            right_px = fitted.getpixel((1074, 960))
            self.assertGreater(left_px[0], 180)
            self.assertGreater(right_px[0], 180)

            cfg = AppConfig(
                root_dir=tdir,
                raw={"video": {"width": 1080, "height": 1920, "hard_monetization_floor_sec": 61}},
            )
            reviewer = AIReviewerAgent(cfg, skip_external_llm=True)
            plan = ScriptPlan(
                run_id="r_dims",
                topic="Dims Test",
                unique_angle="Angle",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Test",
                total_estimated_duration_sec=65.0,
                word_count=5,
                scenes=[ScriptScene(1, 65.0, "Test", "q", "prompt")],
            )
            bad_manifest = MediaManifest(
                run_id="r_dims",
                audio_path="a.mp3",
                srt_path="c.srt",
                ass_path="c.ass",
                final_video_path="v.mp4",
                actual_duration_sec=65.0,
                resolution="720x1280",
                fps=30,
                visual_sources_used=[],
                has_background_music=True,
            )
            photos = [GeneratedPhoto(1, "p1.jpg", "preexisting_custom_ai", "prompt", "AI")]
            v_res = reviewer._heuristic_fallback_review(plan, bad_manifest, photos, revision_round=0)
            self.assertEqual(v_res.verdict, "REVISION_REQUIRED")
            self.assertEqual(v_res.responsible_agent, "montage")

            ok_manifest = MediaManifest(
                run_id="r_dims",
                audio_path="a.mp3",
                srt_path="c.srt",
                ass_path="c.ass",
                final_video_path="v.mp4",
                actual_duration_sec=65.0,
                resolution="1080x1920",
                fps=30,
                visual_sources_used=[],
                has_background_music=True,
            )
            bad_frame = tdir / "bad_frame.jpg"
            Image.new("RGB", (640, 480), (20, 20, 20)).save(bad_frame)
            v_frame = reviewer._heuristic_fallback_review(
                plan, ok_manifest, photos, revision_round=0, frame_pairs=[(1, bad_frame)]
            )
            self.assertEqual(v_frame.verdict, "REVISION_REQUIRED")
            self.assertEqual(v_frame.responsible_agent, "photo")
            self.assertEqual(v_frame.flagged_scenes, [1])

    def test_unified_tiktok_folder_media_intake_and_same_folder_export(self) -> None:
        import subprocess
        from src.utils.ffmpeg_tools import resolve_ffmpeg_executable

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "pipeline": {"media_intake_mode": "tiktok_folder_and_chat", "auto_reuse_tiktok_media_dir": True},
                    "models": {"photo_provider_priority": ["offline_card"]},
                    "video": {"width": 1080, "height": 1920, "fps": 30, "hard_monetization_floor_sec": 61, "max_duration_sec": 85},
                    "sound_design": {"enabled": False},
                    "subtitles": {"enabled": True, "burn_in": True, "margin_v": 520},
                    "qa_and_publish": {
                        "require_human_approval": True,
                        "ai_generated_disclosure": True,
                        "max_similarity_threshold": 0.55,
                        "tiktok": {"export_bundle": True, "export_dir": "tiktok", "disclose_ai_content": True},
                    },
                },
            )
            run_id = "run_uni"
            tiktok_run_dir = cfg.tiktok_dir / run_id
            tiktok_run_dir.mkdir(parents=True, exist_ok=True)
            out_dir = cfg.output_dir / run_id
            out_dir.mkdir(parents=True, exist_ok=True)

            # Place user image scene_01.jpg and user video scene_02.mp4 inside tiktok/<run_id>/
            img1 = Image.new("RGB", (720, 1280))
            img1.putdata([((x * 5) % 255, (y * 3) % 255, 90) for y in range(1280) for x in range(720)])
            img1.save(tiktok_run_dir / "scene_01.jpg")

            ff = resolve_ffmpeg_executable()
            raw_clip = tiktok_run_dir / "scene_02.mp4"
            subprocess.run(
                [ff, "-y", "-f", "lavfi", "-i", "testsrc2=s=720x1280:d=2.0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw_clip)],
                check=True,
                capture_output=True,
            )

            store = MemoryStore(cfg.data_dir)
            store.record_run_start(run_id, "Tech")
            store.update_run_script(run_id, "Unified Folder Test", "Hook", "Full script")

            plan = ScriptPlan(
                run_id=run_id,
                topic="Unified Folder Test",
                unique_angle="Angle",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Scene one scene two",
                total_estimated_duration_sec=65.0,
                word_count=4,
                scenes=[
                    ScriptScene(1, 2.0, "Scene one", "q1", "p1"),
                    ScriptScene(2, 2.0, "Scene two", "q2", "p2"),
                ],
            )
            photos = PhotoGeneratorAgent(cfg).run(
                plan, out_dir, reuse_photos_dir=tiktok_run_dir, extra_reuse_dirs=[tiktok_run_dir]
            )
            self.assertEqual([p.provider for p in photos], ["preexisting_custom_ai", "offline_atmosphere"])

            clips = VideoGeneratorAgent(cfg).run(
                plan, photos, {1: 1.5, 2: 1.5}, out_dir, custom_clips_dir=[tiktok_run_dir]
            )
            self.assertEqual([c.provider for c in clips], ["cinematic_i2v_motion", "preexisting_custom_video"])

            audio_path = generate_dark_ambient_bed(out_dir / "voiceover.wav", duration_sec=3.0)
            manifest = MontageAgent(cfg).assemble(
                run_id=run_id,
                audio_path=audio_path,
                timings=[WordTiming("Scene", 0.0, 1.0), WordTiming("one", 1.0, 2.5)],
                total_duration_sec=3.0,
                photos=photos,
                clips=clips,
                out_dir=out_dir,
            )
            manifest.actual_duration_sec = 65.0

            pub = QAPublisherAgent(cfg, store, skip_external_llm=True)
            pkg = pub.prepare_review_package(run_id, plan, manifest, out_dir)
            self.assertEqual(pkg.status, "READY_FOR_REVIEW")
            res = pub.approve_and_publish(run_id)
            self.assertEqual(Path(res["tiktok_bundle_dir"]), tiktok_run_dir)
            self.assertTrue((tiktok_run_dir / "final_9x16.mp4").exists())
            self.assertTrue((tiktok_run_dir / "tiktok_post_details.txt").exists())
            self.assertTrue((tiktok_run_dir / "scene_01.jpg").exists())
            self.assertTrue((tiktok_run_dir / "scene_02.mp4").exists())

    def test_curated_trends_queue_and_topic_cues(self) -> None:
        repo_cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
        self.assertEqual(
            repo_cfg.research.get("topic_categories"),
            ["movies", "series", "games", "ai"],
        )
        self.assertIn("curated_trends_queue", repo_cfg.research.get("sources", []))

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            queue_file = tdir / "data" / "trends_queue.json"
            queue_file.parent.mkdir(parents=True, exist_ok=True)
            queue_file.write_text(
                json.dumps(
                    {
                        "updated_at": "2026-11-26",
                        "categories": ["movies", "series", "games", "ai"],
                        "trends": [
                            {"topic": "Completed Movie Trend", "category": "movies", "status": "DONE", "priority": 99},
                            {"topic": "Avengers Doomsday Infinity Vision", "category": "movies", "status": "PENDING", "priority": 95, "summary": "Cinema format shift"},
                            {"topic": "AI World Models in Game Engines", "category": "ai", "status": "PENDING", "priority": 90, "summary": "AI demo vs shipped"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            cfg = AppConfig(
                root_dir=tdir,
                raw={"research": {"trends_queue_file": "data/trends_queue.json"}},
            )
            agent = TrendResearchAgent(cfg, MemoryStore(tdir), skip_external_llm=True)
            fetched = agent._fetch_curated_trends_queue()
            self.assertEqual([s["title"] for s in fetched], ["Avengers Doomsday Infinity Vision", "AI World Models in Game Engines"])

            ai_score = agent._score_trend_signal(fetched[1])
            plain_score = agent._score_trend_signal({"source": "rss", "title": "General update bulletin", "summary": "misc note"})
            self.assertGreater(ai_score, plain_score)

    def test_viral_studio_pro_gate_publish_kit_and_multilang_config(self) -> None:
        from scripts.verify_video_gate import verify_video

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        self.assertEqual(repo_cfg.project.get("channel_name"), "Viral Studio Pro")
        self.assertIs(repo_cfg.pipeline.get("zero_cost_local_only"), True)
        self.assertIs(repo_cfg.project.get("enforce_english_only"), True)
        self.assertEqual(repo_cfg.project.get("supported_languages"), ["en"])
        self.assertEqual(repo_cfg.video.get("width"), 2160)
        self.assertEqual(repo_cfg.video.get("height"), 3840)
        self.assertEqual(repo_cfg.video.get("crf"), 16)
        self.assertIs(repo_cfg.telegram.get("send_as_document"), True)
        self.assertEqual(repo_cfg.tts.get("voices", {}).get("en"), "en-US-AndrewNeural")
        self.assertNotIn("de", repo_cfg.tts.get("voices", {}))
        self.assertNotIn("fr", repo_cfg.tts.get("voices", {}))
        self.assertIs(repo_cfg.qa_and_publish.get("tiktok", {}).get("generate_publish_kit"), True)
        self.assertIs(repo_cfg.qa_and_publish.get("tiktok", {}).get("allow_auto_public_upload"), False)

        for skill_name in (
            "tiktok-trend-hunter",
            "highres-clip-harvester",
            "viral-script-multilang",
            "gpu-caption-qa",
        ):
            self.assertTrue((repo_root / ".agent" / "skills" / skill_name / "SKILL.md").exists())
        self.assertTrue((repo_root / ".agents" / "hooks.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            missing = verify_video(Path(tmp) / "missing.mp4")
            self.assertFalse(missing["passed"])

        sample_mp4 = repo_root / "tiktok" / "run_20261125_181000" / "final_9x16.mp4"
        if sample_mp4.exists():
            res = verify_video(sample_mp4, min_sec=61.0)
            self.assertTrue(res["passed"])
            self.assertEqual((res["width"], res["height"]), (1080, 1920))
            self.assertTrue(res["has_audio"])

    def test_trend_history_manager_dedup_and_filtering(self) -> None:
        from scripts.trend_history_manager import process_candidates

        repo_root = Path(__file__).resolve().parent.parent
        self.assertTrue((repo_root / ".agents" / "daily_trend_hunter_automation.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            hist_path = tdir / "trends_history.json"
            log_path = tdir / "history_log.json"
            latest_md = tdir / "latest_trends.md"
            log_path.write_text(
                json.dumps(
                    [
                        {
                            "run_id": "run_old",
                            "topic": "Grand Theft Auto VI Launches",
                            "status": "APPROVED_AND_PACKAGED",
                            "created_at": "2026-11-25T00:00:00Z",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            candidates = [
                {"topic": "Grand Theft Auto VI Launches", "source": "google_trends", "category": "games"},
                {"topic": "Kids Baby Shark Nursery Rhyme", "source": "youtube_trending", "category": "kids"},
                {"topic": "Viral TikTok Dance Challenge Tutorial", "source": "tiktok_creative_center", "category": "dance"},
                {"topic": "UFC Fight Full Movie Leaked Clip", "source": "youtube_trending", "category": "sports"},
                {
                    "topic": "Blade Runner 2099 Prime Video Sci-Fi Series",
                    "source": "tiktok_creative_center",
                    "sources": ["tiktok_creative_center", "youtube_trending"],
                    "category": "series",
                    "summary": "Breakout sci-fi series explained with high-res trailer footage.",
                    "approx_traffic": 350000,
                },
            ]
            out = process_candidates(
                candidates=candidates,
                history_path=hist_path,
                pipeline_log_path=log_path,
                latest_md_path=latest_md,
            )
            self.assertEqual(len(out["rejected"]), 4)
            self.assertEqual(out["top_pick"]["topic"], "Blade Runner 2099 Prime Video Sci-Fi Series")
            self.assertTrue(hist_path.exists())
            self.assertTrue(latest_md.exists())
            self.assertIn("الترند #1 المختار", latest_md.read_text(encoding="utf-8"))

    def test_telegram_delivery_graceful_missing_env_and_payload_builder(self) -> None:
        import os
        import unittest.mock
        from scripts.send_to_telegram import (
            deliver_run_to_telegram,
            deliver_trends_to_telegram,
            mask_secret,
        )

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        self.assertIs(repo_cfg.telegram.get("enabled"), True)
        self.assertEqual(repo_cfg.telegram.get("transport"), "telethon_mtproto")
        self.assertEqual(repo_cfg.telegram.get("review_mode"), "streamable_preview")
        self.assertIs(repo_cfg.telegram.get("send_raw_on_review"), False)
        self.assertIs(repo_cfg.telegram.get("send_as_document"), True)
        self.assertEqual(repo_cfg.telegram.get("max_upload_mb"), 1950)
        self.assertEqual(repo_cfg.telegram.get("session_file"), "data/viral_studio_bot.session")

        gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore_text)
        self.assertIn("*.session*", gitignore_text)
        env_ex_text = (repo_root / ".env.example").read_text(encoding="utf-8")
        for key_name in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            self.assertIn(key_name, env_ex_text)

        self.assertEqual(
            mask_secret("https://api.telegram.org/bot123:ABC/sendMessage?h=xyz", "123:ABC", "xyz"),
            "https://api.telegram.org/bot***TELEGRAM_TOKEN***/sendMessage?h=***TELEGRAM_SECRET***",
        )

        sample_out = repo_root / "tiktok" / "run_20261125_181000"
        if (sample_out / "final_9x16.mp4").exists():
            dry = deliver_run_to_telegram(sample_out, config=repo_cfg, dry_run=True)
            self.assertTrue(dry["ok"])
            self.assertEqual(dry["transport"], "telethon_mtproto")
            self.assertEqual(dry["method"], "sendDocument")
            self.assertIs(dry["force_document"], True)
            self.assertIn("<pre>", dry["copy_message_html"])

            saved_env = {
                k: os.environ.pop(k, None)
                for k in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
            }
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    empty_env = Path(tmp) / ".env"
                    empty_env.write_text("TELEGRAM_BOT_TOKEN=123:ABC\nTELEGRAM_CHAT_ID=1015524262\n", encoding="utf-8")
                    grace = deliver_run_to_telegram(
                        sample_out, config=repo_cfg, dry_run=False, graceful=True, env_file=empty_env
                    )
                    self.assertFalse(grace["ok"])
                    self.assertEqual(grace["reason"], "missing_credentials")
                    self.assertIn("TELEGRAM_API_ID", grace["message_ar"])
                    self.assertIn("TELEGRAM_API_HASH", grace["message_ar"])
            finally:
                for k, v in saved_env.items():
                    if v is not None:
                        os.environ[k] = v

        trends_md_sample = repo_root / "data" / "latest_trends.md"
        if trends_md_sample.exists():
            trends_dry = deliver_trends_to_telegram(trends_md_sample, dry_run=True)
            self.assertTrue(trends_dry["ok"])
            self.assertEqual(trends_dry["transport"], "telethon_mtproto")

        # Telethon MTProto bypasses 48MB compression for >48MB masters
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            env_p = tdir / ".env"
            env_p.write_text("TELEGRAM_API_ID=123456\nTELEGRAM_API_HASH=secret_hash_val\n", encoding="utf-8")
            run_dir = tdir / "run_big_4k"
            run_dir.mkdir(parents=True, exist_ok=True)
            big_mp4 = run_dir / "final_9x16.mp4"
            big_mp4.write_bytes(b"\x00" * (49 * 1024 * 1024))
            (run_dir / "review_package.json").write_text(
                json.dumps({"run_id": "run_big_4k", "topic": "Big 4K Master", "metadata": {"tiktok_caption": "Cap"}}),
                encoding="utf-8",
            )
            big_cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "project": {"language": "en"},
                    "video": {"hard_monetization_floor_sec": 61},
                    "telegram": {"enabled": True, "send_as_document": True, "max_upload_mb": 1950},
                },
            )
            fake_gate = {"passed": True, "duration_sec": 65.0, "width": 2160, "height": 3840, "issues": []}
            with unittest.mock.patch("scripts.send_to_telegram.verify_video", return_value=fake_gate):
                dry_mt = deliver_run_to_telegram(run_dir, config=big_cfg, dry_run=True, env_file=env_p)
            self.assertTrue(dry_mt["ok"])
            self.assertEqual(dry_mt["transport"], "telethon_mtproto")
            self.assertEqual(dry_mt["method"], "sendDocument")
            self.assertIs(dry_mt["force_document"], True)
            self.assertEqual(Path(dry_mt["delivery_video"]), big_mp4)

    def test_telegram_bot_state_transitions_whitelist_and_48h_fallback(self) -> None:
        import asyncio
        from telethon.errors import RPCError
        from telethon.tl.types import ReplyInlineMarkup
        from scripts.telegram_bot import (
            BotStateStore,
            _delete_or_archive_card,
            format_trend_card_caption,
            is_authorized,
        )
        from scripts.trend_cards_pool import build_category_trend_pools

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")

        # 1. Hard whitelist enforcement
        self.assertTrue(is_authorized(1015524262, 1015524262, 1015524262))
        self.assertFalse(is_authorized(9999999, 1015524262, 1015524262))
        self.assertFalse(is_authorized(1015524262, 9999999, 1015524262))

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            pools = build_category_trend_pools(
                config=repo_cfg,
                thumbs_dir=tdir / "thumbs",
                fetch_remote_thumbs=False,
            )
            for req_cat in ("movies", "series", "games", "ai"):
                self.assertIn(req_cat, pools)
                self.assertGreaterEqual(len(pools[req_cat]["items"]), 3)
                first = pools[req_cat]["items"][0]
                self.assertTrue(Path(first["image_url_or_path"]).exists())
                cap = format_trend_card_caption(first)
                self.assertEqual(len(cap.splitlines()), 3)

            store = BotStateStore(tdir / "state.json")
            store.set_pools(pools)
            store.register_card(1001, "games", pool_index=0)
            store.register_card(1002, "ai", pool_index=0)

            # next_trend advances only that card's pool_index
            nxt, wrapped = store.advance_card_trend(1001, "games")
            self.assertIsNotNone(nxt)
            self.assertFalse(wrapped)
            self.assertEqual(store.get_card(1001)["pool_index"], 1)
            self.assertEqual(store.get_card(1002)["pool_index"], 0)

            # produce locks card & starts job
            job, status = store.start_job_for_card(1001, "games", daily_quota=2)
            self.assertEqual(status, "ok")
            assert job is not None
            store.mark_job_review_ready(job["job_id"], review_message_id=2001)

            # edit loop
            self.assertIsNotNone(store.enter_edit_feedback(job["job_id"]))
            rev_job = store.submit_edit_feedback("Make voice faster")
            self.assertEqual(rev_job["revision_round"], 1)
            store.mark_job_review_ready(job["job_id"], review_message_id=2002)

            # approve & publish
            store.approve_job(job["job_id"], publish_message_id=3001)
            hist_path = tdir / "trends_history.json"
            q_path = tdir / "trends_queue.json"
            q_path.write_text(json.dumps({"trends": [{"id": job["trend_id"], "topic": job["trend_title"], "status": "PENDING"}]}), encoding="utf-8")

            pub_job, already = store.mark_job_published(job["job_id"], history_path=hist_path, queue_path=q_path)
            self.assertFalse(already)
            self.assertEqual(pub_job["status"], "published")
            self.assertEqual(store.get_today_quota_used(), 1)
            hist_data = json.loads(hist_path.read_text(encoding="utf-8"))
            self.assertTrue(any(e.get("status") == "published" and e.get("topic") == job["trend_title"] for e in hist_data["entries"]))

            # 48h delete fallback test (Telethon RPCError + ReplyInlineMarkup clear)
            class _OldCardClient:
                def __init__(self) -> None:
                    self.edited_text = ""
                    self.edited_buttons = None

                async def delete_messages(self, entity: int, message_ids: list[int], revoke: bool = True) -> None:
                    raise RPCError(None, "MESSAGE_DELETE_FORBIDDEN")

                async def edit_message(self, entity: int, message: int, text: str = "", buttons=None) -> None:
                    self.edited_text = text
                    self.edited_buttons = buttons

            fake_client = _OldCardClient()
            outcome = asyncio.run(_delete_or_archive_card(fake_client, 1015524262, 1001))
            self.assertEqual(outcome, "archived")
            self.assertIn("تم النشر والأرشفة", fake_client.edited_text)
            self.assertIsInstance(fake_client.edited_buttons, ReplyInlineMarkup)
            self.assertEqual(fake_client.edited_buttons.rows, [])

    def test_four_fixed_pillars_zero_cross_category_overlap(self) -> None:
        from scripts.trend_cards_pool import build_category_trend_pools, count_cross_category_overlaps

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        cats = repo_cfg.telegram.get("categories", [])
        self.assertEqual(
            [(c["key"], c["label_ar"]) for c in cats],
            [
                ("movies", "أفلام"),
                ("series", "مسلسلات"),
                ("games", "ألعاب"),
                ("ai", "ذكاء اصطناعي"),
            ],
        )
        self.assertEqual(float(repo_cfg.telegram.get("card_send_delay_sec")), 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            q_path = tdir / "queue.json"
            h_path = tdir / "history.json"
            l_path = tdir / "log.json"
            h_path.write_text(
                json.dumps({"entries": [{"topic": "Already Published GTA VI Story", "status": "published"}]}),
                encoding="utf-8",
            )
            l_path.write_text("[]", encoding="utf-8")
            q_path.write_text(
                json.dumps(
                    {
                        "trends": [
                            {"id": "dup_1", "category": "movies", "topic": "OpenAI Sora Pre-Vis Cinema Deal", "priority": 99, "status": "PENDING"},
                            {"id": "dup_2", "category": "ai", "topic": "OpenAI Sora Video Model Update", "priority": 98, "status": "PENDING"},
                            {"id": "dup_3", "category": "games", "topic": "OpenAI NPC World Model Demo", "priority": 97, "status": "PENDING"},
                            {"id": "pub_1", "category": "games", "topic": "Already Published GTA VI Story", "priority": 99, "status": "PENDING"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pools = build_category_trend_pools(
                config=repo_cfg,
                queue_path=q_path,
                history_path=h_path,
                pipeline_log_path=l_path,
                thumbs_dir=tdir / "thumbs",
                fetch_remote_thumbs=False,
            )
            self.assertEqual(len(pools), 4)
            for k in ("movies", "series", "games", "ai"):
                self.assertGreaterEqual(len(pools[k]["items"]), 3)
            self.assertEqual(count_cross_category_overlaps(pools), 0)
            all_titles = [i["title"] for c in pools.values() for i in c["items"]]
            self.assertNotIn("Already Published GTA VI Story", all_titles)
            openai_hits = [t for t in all_titles if "openai" in t.lower()]
            self.assertLessEqual(len(openai_hits), 1)

    def test_telegram_bot_single_instance_lock_and_rotating_logger(self) -> None:
        import os
        from logging.handlers import RotatingFileHandler
        from scripts.telegram_bot import SingleInstanceLock, configure_logging, next_backoff_sec

        repo_root = Path(__file__).resolve().parent.parent
        for ps1 in (
            "install_bot_autostart.ps1",
            "uninstall_bot_autostart.ps1",
            "bot_service_status.ps1",
        ):
            self.assertTrue((repo_root / "scripts" / ps1).exists())
        self.assertIn("logs/", (repo_root / ".gitignore").read_text(encoding="utf-8"))

        # Exponential backoff with uptime reset
        self.assertEqual(next_backoff_sec(5.0, uptime_sec=10.0), 10.0)
        self.assertEqual(next_backoff_sec(80.0, uptime_sec=10.0), 120.0)
        self.assertEqual(next_backoff_sec(80.0, uptime_sec=75.0), 5.0)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            pid_path = tdir / "bot.pid"
            mutex_name = f"Local\\ViralStudioPro_TestLock_{os.getpid()}_{id(tdir)}"
            lock1 = SingleInstanceLock(pid_path=pid_path, mutex_name=mutex_name)
            lock2 = SingleInstanceLock(pid_path=pid_path, mutex_name=mutex_name)
            try:
                self.assertTrue(lock1.acquire())
                self.assertTrue(pid_path.exists())
                self.assertEqual(pid_path.read_text(encoding="utf-8").strip(), str(os.getpid()))
                self.assertFalse(lock2.acquire())
            finally:
                lock1.release()
            self.assertFalse(pid_path.exists())

            log_path = tdir / "logs" / "bot.log"
            test_logger = configure_logging(log_path=log_path, logger_name="test_rot_bot")
            try:
                rot_handlers = [h for h in test_logger.handlers if isinstance(h, RotatingFileHandler)]
                self.assertEqual(len(rot_handlers), 1)
                self.assertEqual(rot_handlers[0].maxBytes, 5 * 1024 * 1024)
                self.assertEqual(rot_handlers[0].backupCount, 3)
                test_logger.info("اختبار سجل عربي")
                rot_handlers[0].flush()
                self.assertIn("اختبار سجل عربي", log_path.read_text(encoding="utf-8"))
            finally:
                for h in list(test_logger.handlers):
                    test_logger.removeHandler(h)
                    h.close()

    def test_portability_no_hardcoded_paths_and_daily_quota_state(self) -> None:
        import os
        import re
        from scripts.check_daily_quota import get_daily_quota_status, record_daily_video
        from src.utils.ffmpeg_tools import detect_h264_encoder, resolve_ffprobe_executable

        repo_root = Path(__file__).resolve().parent.parent
        scan_files = [
            repo_root / "run_pipeline.py",
            repo_root / "config.yaml",
            repo_root / ".agents" / "hooks.json",
            repo_root / "start_telegram_bot.bat",
            *sorted((repo_root / "src").rglob("*.py")),
            *sorted((repo_root / "scripts").glob("*.py")),
        ]
        bad_pat = re.compile(r"d:\\\\?money|c:\\\\", re.I)
        hits: list[str] = []
        for fp in scan_files:
            txt = fp.read_text(encoding="utf-8", errors="replace")
            if bad_pat.search(txt):
                hits.append(str(fp.relative_to(repo_root)))
        self.assertEqual(hits, [], f"Hardcoded drive paths found: {hits}")

        # Verify load_config works cleanly from a completely different working directory
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                cfg = load_config("config.yaml")
                self.assertEqual(cfg.root_dir, repo_root)
                self.assertEqual(cfg.logs_dir, repo_root / "logs")
            finally:
                os.chdir(old_cwd)

            q_state = Path(tmp) / "daily_run_state.json"
            s0 = get_daily_quota_status(state_path=q_state, config=cfg)
            self.assertEqual(s0["used"], 0)
            self.assertTrue(s0["allowed"])
            s1 = record_daily_video("run_p1", "Test Topic", state_path=q_state, config=cfg)
            self.assertEqual(s1["used"], 1)
            s2 = record_daily_video("run_p2", "Test Topic 2", state_path=q_state, config=cfg)
            self.assertFalse(s2["allowed"])

        enc = detect_h264_encoder(prefer_nvenc=True)
        self.assertIn(enc, ("h264_nvenc", "libx264"))
        self.assertTrue(bool(resolve_ffprobe_executable()))
        self.assertTrue((repo_root / "MIGRATION_GUIDE.md").exists())
        req_txt = (repo_root / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("python-telegram-bot", req_txt)
        self.assertIn("telethon", req_txt)

    def test_auto_harvester_helpers_and_bot_menu_cleanup_and_lists(self) -> None:
        import asyncio
        from scripts.auto_media_harvester import (
            BLOCKED_WATERMARK_DOMAINS,
            compute_segment_starts,
            parse_cropdetect,
        )
        from scripts.telegram_bot import (
            BotStateStore,
            collect_published_trends_list,
            collect_unpublished_runs,
            extract_pipeline_failure_reason,
            format_dashboard_text,
            main_menu_keyboard,
        )

        self.assertIn("gettyimages", BLOCKED_WATERMARK_DOMAINS)
        self.assertEqual(
            parse_cropdetect("[cropdetect] x:0 y:140 crop=1920:800:0:140\n[cropdetect] crop=1920:800:0:140"),
            "crop=1920:800:0:140",
        )
        starts = compute_segment_starts(duration_sec=120.0, count=5, seg_len=8.0)
        self.assertEqual(len(starts), 5)
        self.assertTrue(all(s >= 3.0 for s in starts))

        kb = main_menu_keyboard()
        cb_data = [
            btn.data.decode("utf-8") if isinstance(btn.data, (bytes, bytearray)) else str(btn.data)
            for row in kb
            for btn in row
        ]
        self.assertEqual(cb_data, ["menu:trends", "menu:status", "menu:published", "menu:unpublished"])
        self.assertIn("1 / 2", format_dashboard_text(1, 2))

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            tiktok_dir = tdir / "tiktok"
            pub_run = tiktok_dir / "run_pub_1"
            pub_run.mkdir(parents=True)
            (pub_run / "final_9x16.mp4").write_bytes(b"0" * 5000)
            (pub_run / "review_package.json").write_text(
                json.dumps({"run_id": "run_pub_1", "topic": "GTA VI Launch", "status": "READY_FOR_REVIEW", "duration_sec": 70.0, "metadata": {"tiktok_caption": "GTA", "tiktok_hashtags": ["gta6"]}}),
                encoding="utf-8",
            )

            unpub_run = tiktok_dir / "run_unpub_1"
            unpub_run.mkdir(parents=True)
            (unpub_run / "final_9x16.mp4").write_bytes(b"0" * 5000)
            (unpub_run / "review_package.json").write_text(
                json.dumps({"run_id": "run_unpub_1", "topic": "Rings of Power S3", "status": "READY_FOR_REVIEW", "duration_sec": 72.0, "metadata": {"tiktok_caption": "Rings", "tiktok_hashtags": ["lotr"]}}),
                encoding="utf-8",
            )

            failed_run = tiktok_dir / "run_fail_1"
            failed_run.mkdir(parents=True)
            (failed_run / "final_9x16.mp4").write_bytes(b"0" * 5000)
            (failed_run / "review_package.json").write_text(
                json.dumps({"run_id": "run_fail_1", "topic": "Broken", "status": "QA_FAILED", "qa": {"issues": ["Scene 3 offline"]}, "reviewer_verdict": {"error_diagnosis": "Scenes [3] failed"}}),
                encoding="utf-8",
            )
            reason = extract_pipeline_failure_reason(failed_run, {"issues": []}, "HUMAN-IN-THE-LOOP GATE ACTIVE")
            self.assertIn("Scene 3 offline", reason)
            self.assertNotIn("HUMAN-IN-THE-LOOP", reason)

            hist_path = tdir / "trends_history.json"
            hist_path.write_text(
                json.dumps({"entries": [{"topic": "GTA VI Launch", "run_id": "run_pub_1", "status": "published", "timestamp": "2026-11-25T00:00:00Z"}]}),
                encoding="utf-8",
            )
            log_path = tdir / "history_log.json"
            log_path.write_text("[]", encoding="utf-8")

            store = BotStateStore(tdir / "state.json")
            store.track_message(101)
            store.track_message(102)
            store.set_dashboard_message(103)
            pub_list = collect_published_trends_list(store.state, history_path=hist_path, pipeline_log_path=log_path)
            self.assertEqual([x["title"] for x in pub_list], ["GTA VI Launch"])

            unpub_list = collect_unpublished_runs(store.state, tiktok_dir, history_path=hist_path, pipeline_log_path=log_path)
            self.assertEqual([x["run_id"] for x in unpub_list], ["run_unpub_1"])
            self.assertIn("<pre>", unpub_list[0]["copy_html"])

            deleted_ids: list[int] = []
            class _CleanupClient:
                async def delete_messages(self, entity: int, message_ids: list[int], revoke: bool = True) -> None:
                    deleted_ids.extend(message_ids)

                async def send_message(self, *args, **kwargs):
                    class _M:
                        id = 999
                    return _M()

            client_obj = _CleanupClient()
            all_ids = store.collect_all_tracked_ids()
            self.assertEqual(all_ids, [101, 102, 103])
            for mid in reversed(all_ids):
                asyncio.run(client_obj.delete_messages(1015524262, [mid], revoke=True))
            store.clear_tracked_messages()
            self.assertEqual(set(deleted_ids), {101, 102, 103})
            self.assertEqual(store.collect_all_tracked_ids(), [])

    def test_editorial_hud_pixel_captions_word_alignment_and_48k_master(self) -> None:
        import subprocess
        from src.providers.editorial_hud import (
            measure_text_width_px,
            render_scene_hud_overlay,
            wrap_caption_balanced,
        )
        from src.utils.ffmpeg_tools import measure_audio_metrics, resolve_ffmpeg_executable

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        self.assertIs(repo_cfg.editorial_hud.get("enabled"), True)
        self.assertEqual(repo_cfg.editorial_hud.get("accent_color"), "#CDFF43")
        self.assertEqual(repo_cfg.tts.get("voices", {}).get("en"), "en-US-AndrewNeural")
        self.assertEqual(repo_cfg.sound_design.get("sample_rate"), 48000)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            # 1. Verify Editorial HUD PNG overlay has 1080x1920 size and Acid-Lime accents
            hud_png = render_scene_hud_overlay(
                tdir / "hud_01.png",
                brand_label="VIRAL STUDIO PRO",
                accent_hex="#CDFF43",
                kicker="TWO EVENTS",
                headline_line1="GOLDEN WEEK",
                headline_line2="NOV 28",
                footage_context="CURRENT SEASON FOOTAGE",
                attribution_line="Official trailers & media | Commentary & analysis",
                fact_card="Chapter 7 finale: DEC 5",
                scene_index=2,
                total_scenes=12,
                category_label="GAMING EXPLAINED",
            )
            with Image.open(hud_png) as him:
                self.assertEqual(him.size, (1080, 1920))
                self.assertEqual(him.mode, "RGBA")
                # Top brand pill & bottom progress bar both contain Acid-Lime (#CDFF43)
                pill_px = him.getpixel((90, 170))
                bar_px = him.getpixel((110, 1639))
                self.assertGreater(pill_px[1], 240)
                self.assertGreater(bar_px[1], 240)

            # 2. Verify pixel-measured 2-line balanced caption wrapping
            long_phrase = ["Crossbones", "Isle", "eighty-player", "island"]
            groups = wrap_caption_balanced(long_phrase, font_size=57, max_width_px=790)
            self.assertEqual(len(groups), 2)
            for grp in groups:
                line_str = " ".join(long_phrase[i] for i in grp)
                self.assertLessEqual(measure_text_width_px(line_str, font_size=57), 790)

            # 3. Verify MontageAgent editorial captions, word alignment, and 48kHz limiter master + cover.jpg
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"width": 1080, "height": 1920, "fps": 30, "crf": 18, "hard_monetization_floor_sec": 3},
                    "editorial_hud": dict(repo_cfg.editorial_hud),
                    "sound_design": dict(repo_cfg.sound_design),
                    "subtitles": dict(repo_cfg.subtitles),
                },
            )
            plan = ScriptPlan(
                run_id="r_editorial",
                topic="Fortnite Golden Week vs Chapter 7 Finale",
                unique_angle="Clear separation between lead-up week and live event.",
                language="en",
                opening_hook_3s="Two dates you should not confuse.",
                closing_cta="Follow for the next breakdown.",
                full_voiceover_text="Two dates you should not confuse. Save November twenty eighth.",
                total_estimated_duration_sec=4.0,
                word_count=10,
                category_label="GAMING EXPLAINED",
                scenes=[
                    ScriptScene(
                        1, 2.0, "Two dates you should not confuse.", "fortnite battle",
                        "fortnite battle royale", kicker="TWO DATES",
                        headline_line1="TWO DATES.", headline_line2="ONE DIFFERENCE.",
                        fact_card="Golden Week starts Nov 28", layout_mode="wide_panel",
                    ),
                    ScriptScene(
                        2, 2.0, "Save November twenty eighth.", "fortnite island",
                        "fortnite island sunset", kicker="TIMELINE",
                        headline_line1="GOLDEN WEEK", headline_line2="NOV 28",
                        fact_card="Chapter 7 finale: DEC 5", layout_mode="fullscreen_portrait",
                    ),
                ],
            )
            ff = resolve_ffmpeg_executable()
            raw_broll = tdir / "broll_4k.mp4"
            subprocess.run(
                [ff, "-y", "-f", "lavfi", "-i", "testsrc2=s=1920x1080:d=3.0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw_broll)],
                check=True, capture_output=True,
            )
            clips_in = tdir / "custom_clips"
            clips_in.mkdir()
            for name in ("scene_01.mp4", "scene_02.mp4"):
                (clips_in / name).write_bytes(raw_broll.read_bytes())

            photos = [
                GeneratedPhoto(1, str(tdir / "p1.jpg"), "preexisting_custom_ai", "q1", "Test"),
                GeneratedPhoto(2, str(tdir / "p2.jpg"), "preexisting_custom_ai", "q2", "Test"),
            ]
            for p in photos:
                Image.new("RGB", (1080, 1920), (20, 30, 50)).save(p.image_path)

            timings = [
                WordTiming("Two", 0.1, 0.4),
                WordTiming("dates", 0.4, 0.7),
                WordTiming("you", 0.7, 0.9),
                WordTiming("should", 0.9, 1.2),
                WordTiming("not", 1.2, 1.5),
                WordTiming("confuse.", 1.5, 1.9),
                WordTiming("Save", 2.1, 2.5),
                WordTiming("November", 2.5, 3.0),
                WordTiming("twenty", 3.0, 3.4),
                WordTiming("eighth.", 3.4, 3.8),
            ]
            montage = MontageAgent(cfg)
            scene_durs, timeline_rows = montage._align_scenes_to_word_timings(plan, timings, 4.2)
            self.assertEqual(len(timeline_rows), 2)
            self.assertAlmostEqual(sum(scene_durs.values()), 4.2, places=2)

            clips = VideoGeneratorAgent(cfg).run(
                plan, photos, scene_durs, tdir, custom_clips_dir=[clips_in]
            )
            self.assertEqual(len(clips), 2)
            self.assertTrue((tdir / "hud" / "hud_01.png").exists())

            vo_wav = generate_dark_ambient_bed(tdir / "vo.wav", duration_sec=4.0, sample_rate=48000)
            manifest = montage.assemble(
                run_id="r_editorial",
                audio_path=vo_wav,
                timings=timings,
                total_duration_sec=4.2,
                photos=photos,
                clips=clips,
                out_dir=tdir,
            )
            ass_text = Path(manifest.ass_path).read_text(encoding="utf-8")
            self.assertIn("&H0043FFCD", ass_text)
            self.assertIn("\\pos(505,1300)", ass_text)
            self.assertTrue(Path(manifest.cover_path).exists())
            metrics = measure_audio_metrics(ff, Path(manifest.final_video_path))
            self.assertTrue(metrics.get("measured"))
            self.assertEqual(metrics.get("clipped_samples"), 0)

    def test_distinct_topics_not_falsely_blocked_by_concurrent_or_failed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {"width": 1080, "height": 1920, "fps": 30, "hard_monetization_floor_sec": 61, "max_duration_sec": 85},
                    "qa_and_publish": {"max_similarity_threshold": 0.55, "tiktok": {"export_dir": "tiktok"}},
                },
            )
            store = MemoryStore(tdir / "data")
            script_agent = ScriptArchitectAgent(cfg, store, skip_external_llm=True)
            pub = QAPublisherAgent(cfg, store, skip_external_llm=True)

            # Long unique_angle must never truncate mid-word
            opp_a = ViralOpportunity(
                "The Altruists (Netflix) — Crypto Exchange Collapse",
                "Deep-dive breakdown of The Altruists Netflix dramatized collapse of FTX and why the real numbers behind this unprecedented exchange scandal shocked investors worldwide.",
                "Adults", "Hook", "Arc", [], ["altruists", "netflix"],
            )
            raw_a = script_agent._build_topic_script(opp_a, "en", facts=[])
            self.assertNotIn("worldwi.", raw_a["scenes"][4]["voiceover_text"])

            opp_b = ViralOpportunity(
                "Blade Runner 2099 (Prime Video) — Sci-Fi Event",
                "Why Blade Runner 2099 brings Michelle Yeoh into a high-stakes sequel series.",
                "Adults", "Hook", "Arc", [], ["blade runner 2099", "series"],
            )
            raw_b = script_agent._build_topic_script(opp_b, "en", facts=[])

            # Seed run A as READY_FOR_REVIEW and a failed QA_FAILED run C for topic B
            store.record_run_start("run_a", "movies")
            plan_a = script_agent._build_plan_from_raw("run_a", opp_a.topic, opp_a.unique_angle, "en", raw_a, tdir / "out_a")
            store.update_run_review_ready("run_a", "v_a.mp4", 72.0, "READY_FOR_REVIEW")

            store.record_run_start("run_c_failed", "series")
            script_agent._build_plan_from_raw("run_c_failed", opp_b.topic, opp_b.unique_angle, "en", raw_b, tdir / "out_c")
            store.update_run_review_ready("run_c_failed", "v_c.mp4", 72.0, "QA_FAILED")

            # Run B (distinct topic) must NOT be blocked by run A's shared fallback phrasing or run C's QA_FAILED row
            store.record_run_start("run_b", "series")
            plan_b = script_agent._build_plan_from_raw("run_b", opp_b.topic, opp_b.unique_angle, "en", raw_b, tdir / "out_b")
            dummy_vid = tdir / "out_b" / "final_9x16.mp4"
            dummy_srt = tdir / "out_b" / "captions.srt"
            dummy_ass = tdir / "out_b" / "captions.ass"
            dummy_vid.write_bytes(b"0" * 15000)
            dummy_srt.write_text("1\n00:00:00,000 --> 00:00:02,000\nTest\n", encoding="utf-8")
            dummy_ass.write_text("[Script Info]\n" + "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Test\n", encoding="utf-8")

            # Mock manifest + ffprobe audio check via _run_qa check on originality_ok directly
            manifest_b = MediaManifest("run_b", "a.mp3", str(dummy_srt), str(dummy_ass), str(dummy_vid), 72.0, "1080x1920", 30, [], True)
            qa_b = pub._run_qa(plan_b, manifest_b)
            self.assertTrue(qa_b.originality_ok, f"Unexpected similarity block: {qa_b.max_past_similarity}")

            # Duplicate of topic A against READY_FOR_REVIEW run A MUST still be blocked
            store.record_run_start("run_a_dup", "movies")
            plan_a_dup = script_agent._build_plan_from_raw("run_a_dup", opp_a.topic, opp_a.unique_angle, "en", raw_a, tdir / "out_a2")
            qa_dup = pub._run_qa(plan_a_dup, manifest_b)
            self.assertFalse(qa_dup.originality_ok)

            # delete_runs cleans rows cleanly
            self.assertEqual(store.delete_runs(["run_c_failed"]), 1)
            self.assertIsNone(store.get_run("run_c_failed"))

    def test_seamless_loop_bed_sentence_captions_and_loop_qa_preview(self) -> None:
        import struct
        import subprocess
        import wave
        from src.utils.ffmpeg_tools import resolve_ffmpeg_executable

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            # 1. Seamless loop audio bed must stay active at tail (no 1.2s fade-to-silence)
            bed_path = generate_dark_ambient_bed(
                tdir / "loop_bed.wav", duration_sec=3.0, sample_rate=48000, seamless_loop=True
            )
            with wave.open(str(bed_path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = struct.unpack(f"<{len(raw) // 2}h", raw)
            # Last 0.25s before final 0.04s edge taper must have non-trivial energy
            tail_slice = samples[-int(0.25 * 48000) : -int(0.05 * 48000)]
            self.assertGreater(max(abs(s) for s in tail_slice), 500)

            # 2. Sentence-boundary caption splitting keeps distinct sentences separate
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {
                        "width": 1080, "height": 1920, "fps": 30, "crf": 18,
                        "hard_monetization_floor_sec": 3, "seamless_loop": True,
                    },
                    "sound_design": {"enabled": True, "sample_rate": 48000, "seamless_loop": True},
                    "subtitles": {
                        "enabled": True, "burn_in": True, "font_name": "Arial",
                        "font_size": 57, "uppercase": False, "words_per_caption_chunk": 4,
                        "break_on_sentence_boundary": True, "use_fixed_position": True,
                        "caption_center_x": 505, "caption_center_y": 1300,
                    },
                },
            )
            montage = MontageAgent(cfg)
            timings = [
                WordTiming("First", 0.0, 0.3),
                WordTiming("part.", 0.3, 0.6),
                WordTiming("Second", 0.7, 1.0),
                WordTiming("line.", 1.0, 1.4),
            ]
            srt_p = tdir / "c.srt"
            ass_p = tdir / "c.ass"
            montage.build_captions(timings, 1.5, srt_p, ass_p)
            srt_txt = srt_p.read_text(encoding="utf-8")
            self.assertIn("First part.", srt_txt)
            self.assertIn("Second line.", srt_txt)
            self.assertNotIn("part. Second", srt_txt)

            # 3. ScriptArchitect builds topic-adapted seamless loop fields without forcing "And that's why..."
            store = MemoryStore(tdir / "data")
            sa = ScriptArchitectAgent(cfg, store, skip_external_llm=True)
            opp = ViralOpportunity(
                "Avengers: Doomsday Infinity Vision Format",
                "Disney created a 4K laser certification badge after IMAX exclusivity.",
                "Adults", "Hook", "Arc", [], ["avengers", "infinity vision"],
            )
            raw_script = sa._build_topic_script(opp, "en", facts=[])
            self.assertTrue(bool(raw_script["loop_bridge_phrase"]))
            self.assertNotIn("And that's why", raw_script["loop_bridge_phrase"])
            store.record_run_start("run_loop", "movies")
            plan = sa._build_plan_from_raw(
                "run_loop", opp.topic, opp.unique_angle, "en", raw_script, tdir / "out_plan"
            )
            self.assertTrue(plan.loop_joined_preview.startswith(plan.loop_bridge_phrase))

            # 4. Assemble produces loop_qa_transition.mp4 and publish_kit includes Seamless Loop section
            ff = resolve_ffmpeg_executable()
            clip_file = tdir / "clips" / "clip_01.mp4"
            clip_file.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [ff, "-y", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:d=2.0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_file)],
                check=True, capture_output=True,
            )
            img_p = tdir / "p1.jpg"
            Image.new("RGB", (1080, 1920), (20, 30, 50)).save(img_p)
            photo = GeneratedPhoto(1, str(img_p), "preexisting_custom_ai", "p", "Test")
            clip = GeneratedVideoClip(1, str(clip_file), 2.0, "preexisting_custom_video", "live_broll")
            manifest = montage.assemble(
                run_id="run_loop",
                audio_path=bed_path,
                timings=timings,
                total_duration_sec=2.0,
                photos=[photo],
                clips=[clip],
                out_dir=tdir,
            )
            self.assertTrue(bool(manifest.loop_qa_path))
            self.assertTrue(Path(manifest.loop_qa_path).exists())
            pub = QAPublisherAgent(cfg, store, skip_external_llm=True)
            pkg = pub.prepare_review_package("run_loop", plan, manifest, tdir)
            kit_txt = (tdir / "publish_kit.md").read_text(encoding="utf-8")
            self.assertIn("Seamless Loop Check", kit_txt)
            self.assertIn(plan.loop_bridge_phrase, kit_txt)

    def test_trend_sources_registry_live_parser_hardware_exclusion_and_clean_hud(self) -> None:
        from scripts.fetch_live_trends import (
            classify_ai_claim_status,
            fetch_live_trends,
            load_trend_sources,
            parse_feed_xml,
        )
        from src.providers.editorial_hud import render_scene_hud_overlay

        repo_root = Path(__file__).resolve().parent.parent
        reg = load_trend_sources(repo_root / "data" / "trend_sources.json")
        pillars = reg.get("pillars", {})
        self.assertEqual(set(pillars.keys()), {"movies", "series", "games", "ai"})
        total_feeds = sum(
            len(v.get("rss_feeds", [])) + len(v.get("public_hubs", []))
            for v in pillars.values()
        ) + len(reg.get("global_signals", []))
        self.assertGreaterEqual(total_feeds, 35)
        handles = {r["handle"] for r in reg.get("reference_accounts", [])}
        self.assertEqual(handles, {"@ign", "@straw_hat_goofy", "@billyfx"})
        self.assertNotIn("carterpcs", json.dumps(reg).lower())

        auto_cfg = (repo_root / ".agents" / "daily_trend_hunter_automation.json").read_text(encoding="utf-8")
        self.assertIn("fetch_live_trends.py", auto_cfg)
        self.assertIn("--send-cards-once", auto_cfg)

        rss_sample = b"<rss><channel><item><title>Avengers Doomsday Infinity Vision</title><description>Cinema laser standard</description><link>https://variety.com/a</link></item></channel></rss>"
        atom_sample = b"<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>AI World Model Demo</title><summary>New interactive demo</summary><link href='https://deepmind.google/b'/></entry></feed>"
        self.assertEqual(len(parse_feed_xml(rss_sample, "Variety", "movies")), 1)
        self.assertEqual(len(parse_feed_xml(atom_sample, "DeepMind", "ai")), 1)
        self.assertEqual(classify_ai_claim_status("New interactive world model demo preview"), "ANNOUNCEMENT / DEMO")
        self.assertEqual(classify_ai_claim_status("Synthetic AI-generated trailer concept"), "SYNTHETIC / AI-GENERATED")

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            offline_items = [
                {"title": "Best Laptop Unboxing and Hardware Review 2026", "summary": "Shopping guide", "category": "ai", "source": "Test"},
                {"title": "Avengers: Doomsday Infinity Vision 4K Laser Specs", "summary": "Why Disney built a rival cinema badge", "url": "https://variety.com/x", "category": "movies", "source": "Variety Film"},
                {"title": "AI World Model Demo vs Shipped Game Engines", "summary": "Interactive demo preview breakdown", "url": "https://deepmind.google/y", "category": "ai", "source": "DeepMind"},
            ]
            out = fetch_live_trends(
                sources_path=repo_root / "data" / "trend_sources.json",
                history_path=tdir / "trends_history.json",
                pipeline_log_path=tdir / "history_log.json",
                queue_path=tdir / "trends_queue.json",
                latest_md_path=tdir / "latest_trends.md",
                offline_items=offline_items,
            )
            self.assertEqual(out["rejected_count"], 1)
            self.assertEqual(len(out["selected_by_pillar"]["movies"]), 1)
            self.assertEqual(len(out["selected_by_pillar"]["ai"]), 1)
            self.assertEqual(out["selected_by_pillar"]["ai"][0]["ai_claim_status"], "ANNOUNCEMENT / DEMO")
            self.assertTrue((tdir / "latest_trends.md").exists())

            hud_clean = render_scene_hud_overlay(
                tdir / "hud_clean.png",
                kicker="AI VS GAMES",
                headline_line1="SHIPPED VS DEMO",
                headline_line2="CHECK THE LIMITS",
                footage_context="SYNTHETIC / AI-GENERATED",
                fact_card="Shipped vs demo check limits",  # redundant with headline -> should be omitted
                compact_context_badge=True,
                show_fact_card_only_when_distinct=True,
            )
            with Image.open(hud_clean) as im:
                self.assertEqual(im.size, (1080, 1920))
                # Fact card border at (82, 1490) should remain transparent/unbordered when redundant card is omitted
                self.assertLess(im.getpixel((82, 1490))[3], 180)

            # 4K HUD & verify_video 2160x3840 support check
            import subprocess
            from scripts.verify_video_gate import verify_video
            from src.utils.ffmpeg_tools import resolve_ffmpeg_executable

            hud_4k = render_scene_hud_overlay(
                tdir / "hud_4k.png",
                width=2160,
                height=3840,
                kicker="4K TEST",
                headline_line1="NATIVE ULTRA HD",
                headline_line2="SHARP TEXT",
                compact_context_badge=True,
                show_fact_card_only_when_distinct=True,
            )
            with Image.open(hud_4k) as him4k:
                self.assertEqual(him4k.size, (2160, 3840))
                self.assertEqual(him4k.getpixel((180, 340))[:3], (205, 255, 67))

            mp4_4k = tdir / "sample_4k.mp4"
            ff = resolve_ffmpeg_executable()
            subprocess.run(
                [
                    ff, "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=2160x3840:r=30:d=63",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=63",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(mp4_4k),
                ],
                check=True,
                capture_output=True,
            )
            gate_4k = verify_video(mp4_4k, min_sec=61.0)
            self.assertTrue(gate_4k["passed"])
            self.assertEqual((gate_4k["width"], gate_4k["height"]), (2160, 3840))

    def test_telegram_bot_keeps_dashboard_first_and_clears_cards_on_button_press(self) -> None:
        import asyncio
        import unittest.mock
        from types import SimpleNamespace
        from scripts.send_to_telegram import TelegramCredentials
        from scripts.telegram_bot import InteractiveTikTokBot
        from scripts.trend_cards_pool import build_category_trend_pools

        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            state_path = tdir / "bot_state.json"
            bot_runner = InteractiveTikTokBot(
                config_path=repo_root / "config.yaml",
                state_path=state_path,
                creds=TelegramCredentials(12345, "hash_abc", "123:tok", 1015524262),
            )

            pools = build_category_trend_pools(
                config=bot_runner.config,
                thumbs_dir=tdir / "thumbs",
                fetch_remote_thumbs=False,
            )
            bot_runner.store.set_pools(pools)
            bot_runner.store.set_dashboard_message(100)
            for cid, cat in ((101, "movies"), (102, "series"), (103, "games"), (104, "ai")):
                bot_runner.store.register_card(cid, cat, pool_index=0)
            bot_runner.store.track_message(105)

            sent_log: list[dict[str, object]] = []
            deleted_log: list[int] = []
            edited_log: list[int] = []
            next_id = 200

            class _FakeClient:
                async def edit_message(self, entity: int, message: int, text: str = "", buttons=None, link_preview: bool = False):
                    edited_log.append(int(message))

                async def delete_messages(self, entity: int, message_ids: list[int], revoke: bool = True):
                    deleted_log.extend(int(m) for m in message_ids)

                async def send_message(self, entity: int, text: str, buttons=None, parse_mode=None, link_preview: bool = False):
                    nonlocal next_id
                    next_id += 1
                    sent_log.append({"id": next_id, "text": text})
                    return SimpleNamespace(id=next_id)

            class _FakeCallbackEvent:
                def __init__(self, data: bytes, message_id: int, client: _FakeClient) -> None:
                    self.chat_id = 1015524262
                    self.sender_id = 1015524262
                    self.data = data
                    self.message_id = message_id
                    self.client = client

                async def answer(self, text: str = "", alert: bool = False) -> None:
                    return None

            fake_client = _FakeClient()

            async def _exercise() -> None:
                # 1. Selecting a trend on card #101 deletes ALL cards (#101..#105), preserves Dashboard #100, and sends production start message
                with unittest.mock.patch.object(
                    bot_runner, "_run_pipeline_and_deliver_review", new=unittest.mock.AsyncMock(return_value=None)
                ):
                    ev_prod = _FakeCallbackEvent(b"prod:movies", 101, fake_client)
                    await bot_runner.on_callback(ev_prod)
                    await asyncio.sleep(0)

                self.assertEqual(set(deleted_log), {101, 102, 103, 104, 105})
                self.assertNotIn(100, deleted_log)
                self.assertIn(100, edited_log)
                self.assertIn("⏳ بدأ الإنتاج الآلي للترند", str(sent_log[-1]["text"]))
                self.assertEqual(bot_runner.store.state["dashboard_message_id"], 100)
                self.assertEqual(bot_runner.store.state["cards"], {})

                # 2. Pressing menu:status deletes the previous action message (#201) while keeping Dashboard #100
                deleted_log.clear()
                ev_status = _FakeCallbackEvent(b"menu:status", 100, fake_client)
                await bot_runner.on_callback(ev_status)
                self.assertEqual(deleted_log, [201])
                self.assertIn("العمليات الحالية", str(sent_log[-1]["text"]))
                self.assertEqual(bot_runner.store.collect_all_tracked_ids(), [100, 202])

            asyncio.run(_exercise())

    def test_english_fact_filter_category_label_and_non_english_sanitization(self) -> None:
        from src.agents.script_agent import (
            ScriptArchitectAgent,
            _clean_complete_sentence,
            _infer_category_label,
            _is_english_text,
            _is_topic_relevant_fact,
        )

        # 1. German/non-English rejection vs clean English acceptance
        self.assertFalse(
            _is_english_text(
                "Blade ist ein Horror - Actionfilm aus dem Jahr 1998, der auf den Marvel -Comics gleichen Namens basiert."
            )
        )
        self.assertTrue(
            _is_english_text(
                "Set fifty years after Blade Runner 2049, Michelle Yeoh stars as aging Replicant Olwen on Prime Video."
            )
        )
        self.assertTrue(_is_english_text("Los Angeles has inverted"))

        # 2. Topic entity relevance rejects 1998 Marvel "Blade" for "Blade Runner 2099 (Prime Video)"
        topic = "Blade Runner 2099 (Prime Video) — The Hidden Trailer Clue About Michelle Yeoh"
        self.assertFalse(
            _is_topic_relevant_fact(
                "Blade is an American superhero film based on the Marvel Comics character.",
                topic,
            )
        )
        self.assertTrue(
            _is_topic_relevant_fact(
                "Blade Runner 2099 is set fifty years after the 2017 film and streams on Prime Video.",
                topic,
            )
        )

        # 3. Complete-sentence cleaner strips date prefixes and rejects dangling fragments
        self.assertIsNone(
            _clean_complete_sentence(
                "Aug 21, 1998 · Blade: Directed by Stephen Norrington. Character of the..."
            )
        )
        cleaned = _clean_complete_sentence(
            "Nov 25, 2026 · Blade Runner 2099 stars Michelle Yeoh and Hunter Schafer across eight episodes on Prime Video."
        )
        self.assertIsNotNone(cleaned)
        self.assertTrue(str(cleaned).startswith("Blade Runner 2099 stars Michelle Yeoh"))

        # 4. Category inference ignores multi-pillar niche comma list ("Movies, Series, Games, AI Trends")
        self.assertEqual(
            _infer_category_label(topic, niche="Movies, Series, Games, AI Trends"),
            "SERIES EXPLAINED",
        )
        self.assertEqual(
            _infer_category_label("OpenAI Unveils Real-Time Video World Model Demo", niche="Movies, Series, Games, AI Trends"),
            "AI EXPLAINED",
        )

        # 5. _build_plan_from_raw translates foreign resources into English when translator succeeds,
        #    and falls back to clean English when translation is unavailable.
        import unittest.mock
        from scripts.verify_video_gate import verify_video
        from src.agents.publisher_agent import QAPublisherAgent
        from src.utils.ffmpeg_tools import resolve_ffmpeg_executable

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "project": {
                        "language": "en",
                        "enforce_english_only": True,
                        "niche": "Movies, Series, Games, AI Trends",
                    },
                    "pipeline": {},
                    "models": {},
                    "editorial_hud": {},
                    "video": {"hard_monetization_floor_sec": 61, "max_duration_sec": 85},
                    "qa_and_publish": {"max_similarity_threshold": 0.55},
                },
            )
            store = MemoryStore(cfg.data_dir)
            store.record_run_start("run_en_guard", "series")
            agent = ScriptArchitectAgent(cfg, store, skip_external_llm=True)
            raw_script = {
                "topic": topic,
                "language": "de",
                "unique_angle": "Replicant power reversal",
                "opening_hook_3s": "Blade Runner 2099 flips the franchise rule.",
                "closing_cta": "Who wins in 2099, because from frame one,",
                "scenes": [
                    {
                        "scene_number": 1,
                        "estimated_duration_sec": 35.0,
                        "voiceover_text": "Blade Runner 2099 flips the franchise rule.",
                        "visual_query": "Blade Runner 2099",
                        "fact_card": "Prime Video limited series",
                    },
                    {
                        "scene_number": 2,
                        "estimated_duration_sec": 35.0,
                        "voiceover_text": "Blade ist ein Horror - Actionfilm aus dem Jahr 1998, der auf den Marvel -Comics gleichen Namens basiert.",
                        "visual_query": "Blade Runner 2099 scene",
                        "fact_card": "Ein Film aus dem Jahr 1998",
                    },
                ],
            }
            with unittest.mock.patch(
                "src.utils.language_guard.translate_to_english",
                side_effect=lambda txt, timeout=8.0: (
                    "Blade Runner 2099 stars Michelle Yeoh in an inverted future Los Angeles."
                    if "Horror" in txt
                    else "A 1998 film release"
                ),
            ):
                plan_tr = agent._build_plan_from_raw(
                    run_id="run_en_guard",
                    topic=topic,
                    unique_angle="Replicant power reversal",
                    language="en",
                    raw_script=raw_script,
                    out_dir=tdir / "tiktok" / "run_en_guard",
                )
            self.assertEqual(plan_tr.category_label, "SERIES EXPLAINED")
            self.assertEqual(
                plan_tr.scenes[1].voiceover_text,
                "Blade Runner 2099 stars Michelle Yeoh in an inverted future Los Angeles.",
            )
            self.assertEqual(plan_tr.scenes[1].fact_card, "A 1998 film release")

            # Fallback when translation returns None
            with unittest.mock.patch("src.utils.language_guard.translate_to_english", return_value=None):
                plan_fb = agent._build_plan_from_raw(
                    run_id="run_en_guard",
                    topic=topic,
                    unique_angle="Replicant power reversal",
                    language="en",
                    raw_script=raw_script,
                    out_dir=tdir / "tiktok" / "run_en_guard",
                )
            self.assertTrue(_is_english_text(plan_fb.scenes[1].voiceover_text))
            self.assertTrue(_is_english_text(plan_fb.scenes[1].fact_card))

            # 6. QA Publisher and verify_video gate reject runs whose captions.srt contains German
            run_dir_bad = tdir / "tiktok" / "run_bad_srt"
            run_dir_bad.mkdir(parents=True, exist_ok=True)
            bad_mp4 = run_dir_bad / "final_9x16.mp4"
            ff = resolve_ffmpeg_executable()
            import subprocess
            subprocess.run(
                [
                    ff, "-y",
                    "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=64",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=64",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(bad_mp4),
                ],
                check=True,
                capture_output=True,
            )
            bad_srt = run_dir_bad / "captions.srt"
            bad_ass = run_dir_bad / "captions.ass"
            bad_srt.write_text(
                "1\n00:00:01,000 --> 00:00:04,000\nBlade ist ein Horror Actionfilm aus dem Jahr 1998\n",
                encoding="utf-8",
            )
            bad_ass.write_text("[Script Info]\nTitle: test\n" + ("Dialogue: x\n" * 4), encoding="utf-8")
            gate_res = verify_video(bad_mp4, min_sec=61.0)
            self.assertFalse(gate_res["passed"])
            self.assertTrue(any("Non-English subtitle text" in issue for issue in gate_res["issues"]))

            pub = QAPublisherAgent(cfg, store, skip_external_llm=True)
            bad_manifest = MediaManifest(
                run_id="run_en_guard",
                audio_path=str(bad_mp4),
                srt_path=str(bad_srt),
                ass_path=str(bad_ass),
                final_video_path=str(bad_mp4),
                actual_duration_sec=64.0,
                resolution="1080x1920",
                fps=30,
                visual_sources_used=[],
                has_background_music=True,
            )
            qa_res = pub._run_qa(plan_fb, bad_manifest)
            self.assertFalse(qa_res.passed)
            self.assertTrue(any("Non-English text detected" in issue for issue in qa_res.issues))

    def test_monolingual_tts_voice_and_real_revision_engine(self) -> None:
        from src.providers.tts import build_tts_provider
        from src.utils.revision_engine import (
            apply_revision_feedback,
            parse_scene_numbers,
            parse_time_ranges,
        )

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        self.assertEqual(repo_cfg.tts.get("voices", {}).get("en"), "en-US-AndrewNeural")
        self.assertNotIn("Multilingual", str(repo_cfg.tts.get("voices", {}).get("en")))

        # Safeguard strips 'Multilingual' even if passed manually for English
        multi_cfg = AppConfig(
            root_dir=repo_root,
            raw={"tts": {"provider": "edge_tts", "voices": {"en": "en-US-AndrewMultilingualNeural"}}},
        )
        tts_prov = build_tts_provider(multi_cfg, "en")
        self.assertEqual(tts_prov.voice, "en-US-AndrewNeural")

        self.assertEqual(
            parse_time_ranges("Delete the scenes from 15 seconds to 21 seconds and remerge"),
            [(15.0, 21.0)],
        )
        self.assertEqual(parse_time_ranges("Between 00:15 and 00:22"), [(15.0, 22.0)])
        self.assertEqual(parse_time_ranges("من الثانية ١٥ الى ٢١"), [(15.0, 21.0)])
        self.assertEqual(parse_scene_numbers("Delete scene 3 and المشهد 4"), [3, 4])

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            run_dir = tdir / "tiktok" / "run_rev"
            photos_dir = run_dir / "photos"
            photos_dir.mkdir(parents=True)

            scenes = [
                {
                    "scene_number": i,
                    "estimated_duration_sec": 7.5,
                    "voiceover_text": f"Scene {i} reveals how Blade Runner 2099 reshapes Los Angeles.",
                    "visual_query": f"blade runner scene {i}",
                    "visual_ai_prompt": f"blade runner scene {i}",
                }
                for i in range(1, 11)
            ]
            (run_dir / "script_plan.json").write_text(
                json.dumps({"run_id": "run_rev", "topic": "Blade Runner 2099", "scenes": scenes}),
                encoding="utf-8",
            )
            timings = [
                {
                    "scene_number": i,
                    "start_sec": round((i - 1) * 7.5, 2),
                    "end_sec": round(i * 7.5, 2),
                    "duration_sec": 7.5,
                }
                for i in range(1, 11)
            ]
            (run_dir / "scene_timings.json").write_text(json.dumps(timings), encoding="utf-8")
            (run_dir / "voiceover.mp3").write_bytes(b"stale")
            for i in range(1, 11):
                (run_dir / f"scene_{i:02d}.mp4").write_bytes(f"clip{i}".encode())
                (photos_dir / f"scene_{i:02d}_photo.jpg").write_bytes(f"photo{i}".encode())

            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "project": {"language": "en"},
                    "tts": {"provider": "edge_tts", "voices": {"en": "en-US-AndrewMultilingualNeural"}},
                    "video": {"hard_monetization_floor_sec": 61},
                },
            )
            res = apply_revision_feedback(
                "Do not use German language, delete the scenes from 15 seconds to 21 seconds and remerge",
                run_dir,
                cfg,
            )
            self.assertTrue(res["matched"])
            self.assertEqual(cfg.tts["voices"]["en"], "en-US-AndrewNeural")
            self.assertFalse((run_dir / "voiceover.mp3").exists())

            updated_plan = json.loads((run_dir / "script_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(len(updated_plan["scenes"]), 9)
            self.assertEqual([s["scene_number"] for s in updated_plan["scenes"]], list(range(1, 10)))
            # Old scene 4 shifted to scene 3, old scene 10 removed
            self.assertEqual((run_dir / "scene_03.mp4").read_bytes(), b"clip4")
            self.assertFalse((run_dir / "scene_10.mp4").exists())
            self.assertTrue((run_dir / "revision_applied.json").exists())

    def test_edit_escalation_after_three_rounds_and_ticket_queue(self) -> None:
        from scripts.telegram_bot import BotStateStore
        from src.utils.revision_engine import (
            compute_file_sha256,
            create_ai_escalation_ticket,
            resolve_ai_escalation_ticket,
        )

        repo_root = Path(__file__).resolve().parent.parent
        repo_cfg = load_config(repo_root / "config.yaml")
        self.assertEqual(int(repo_cfg.telegram.get("max_auto_edit_rounds", 0)), 3)
        self.assertEqual(
            repo_cfg.telegram.get("ai_escalation_queue_file"),
            "data/ai_escalation_queue.json",
        )

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            f1 = tdir / "v.mp4"
            f1.write_bytes(b"video_v1")
            h1 = compute_file_sha256(f1)
            self.assertEqual(h1, compute_file_sha256(f1))
            f1.write_bytes(b"video_v2")
            self.assertNotEqual(h1, compute_file_sha256(f1))

            store = BotStateStore(tdir / "state.json")
            store.set_pools(
                {
                    "series": {
                        "category_ar": "مسلسلات",
                        "items": [{"id": "s1", "title": "Blade Runner 2099", "sourcing_keywords": ["br2099"]}],
                    }
                }
            )
            store.register_card(501, "series", pool_index=0)
            job, status = store.start_job_for_card(501, "series", daily_quota=2)
            self.assertEqual(status, "ok")
            assert job is not None
            job_id = job["job_id"]
            max_rounds = int(repo_cfg.telegram.get("max_auto_edit_rounds", 3))

            for r in range(1, 5):
                store.enter_edit_feedback(job_id)
                updated_job = store.submit_edit_feedback(f"Edit attempt #{r}")
                assert updated_job is not None
                if r <= 3:
                    self.assertLessEqual(updated_job["revision_round"], max_rounds)
                    store.mark_job_review_ready(job_id, review_message_id=600 + r)
                else:
                    self.assertGreater(updated_job["revision_round"], max_rounds)
                    run_dir = tdir / "tiktok" / updated_job["run_id"]
                    queue_path = tdir / "data" / "ai_escalation_queue.json"
                    ticket = create_ai_escalation_ticket(
                        updated_job,
                        run_dir,
                        queue_path,
                        diagnosis_ar="تجاوز حد التعديلات التلقائية",
                    )
                    store.mark_job_escalated(job_id, ticket["ticket_id"], reason=ticket["local_ai_diagnosis_ar"])

            esc_job = store.state["jobs"][job_id]
            self.assertEqual(esc_job["status"], "escalated_to_ai")
            self.assertEqual(len(esc_job["edit_history"]), 4)
            queue_data = json.loads((tdir / "data" / "ai_escalation_queue.json").read_text(encoding="utf-8"))
            self.assertEqual(len(queue_data["tickets"]), 1)
            self.assertEqual(queue_data["tickets"][0]["status"], "OPEN")
            self.assertTrue((tdir / "tiktok" / esc_job["run_id"] / "AI_ESCALATION_TICKET.md").exists())

            resolve_ai_escalation_ticket(esc_job["run_id"], tdir / "data" / "ai_escalation_queue.json")
            queue_after = json.loads((tdir / "data" / "ai_escalation_queue.json").read_text(encoding="utf-8"))
            self.assertEqual(queue_after["tickets"][0]["status"], "RESOLVED")

    def test_monetization_floor_padding_exceeds_61s_strictly(self) -> None:
        import unittest.mock

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg = AppConfig(
                root_dir=tdir,
                raw={
                    "video": {
                        "hard_monetization_floor_sec": 61,
                        "seamless_loop": True,
                        "loop_tail_pad_sec": 0.08,
                    },
                    "tts": {"provider": "edge_tts", "voices": {"en": "en-US-AndrewNeural"}},
                },
            )
            plan = ScriptPlan(
                run_id="run_pad",
                topic="Padding Floor Test",
                unique_angle="Strict >61s check",
                language="en",
                opening_hook_3s="Hook",
                closing_cta="CTA",
                full_voiceover_text="Short script ending under sixty seconds.",
                total_estimated_duration_sec=58.0,
                word_count=6,
                scenes=[
                    ScriptScene(i, 5.8, f"Beat {i}", "q", "p")
                    for i in range(1, 11)
                ],
            )

            class _FakeTTS:
                def synthesize(self, text: str, output_audio_path: Path):
                    output_audio_path.write_bytes(b"FAKE_MP3")
                    return [WordTiming(f"w{i}", float(i * 5), float(i * 5 + 4)) for i in range(10)]

            agent = MontageAgent(cfg)
            with (
                unittest.mock.patch("src.agents.montage_agent.resolve_ffmpeg_executable", return_value="ffmpeg"),
                unittest.mock.patch("src.agents.montage_agent.build_tts_provider", return_value=_FakeTTS()),
                unittest.mock.patch("src.agents.montage_agent.probe_media_duration", return_value=58.4),
            ):
                _, _, total_dur, _ = agent.prepare_audio_and_scene_timings(plan, tdir / "out")

            self.assertGreater(total_dur, 61.0)
            self.assertGreaterEqual(total_dur, 62.5)

    def test_telethon_mtproto_review_approve_delivery_progress_and_missing_creds(self) -> None:
        import asyncio
        import logging
        import os
        import unittest.mock
        from types import SimpleNamespace
        from telethon.tl.types import DocumentAttributeVideo, ReplyInlineMarkup
        from scripts.send_to_telegram import (
            MissingTelegramCredentialsError,
            TelegramCredentials,
            load_telethon_credentials,
            make_upload_progress_callback,
            should_send_raw_on_review,
        )
        from scripts.telegram_bot import InteractiveTikTokBot

        repo_root = Path(__file__).resolve().parent.parent
        # 1. Zero legacy python-telegram-bot imports in active bot/sender scripts
        for script_rel in ("scripts/telegram_bot.py", "scripts/send_to_telegram.py"):
            src_txt = (repo_root / script_rel).read_text(encoding="utf-8")
            self.assertNotIn("from telegram", src_txt)
            self.assertNotIn("python-telegram-bot", src_txt)
            self.assertIn("telethon", src_txt)

        # 2. Missing TELEGRAM_API_ID / TELEGRAM_API_HASH fails gracefully with clear Arabic prompt
        saved_env = {
            k: os.environ.pop(k, None)
            for k in ("TELEGRAM_API_ID", "TELEGRAM_API_HASH", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tdir = Path(tmp)
                partial_env = tdir / ".env"
                partial_env.write_text("TELEGRAM_BOT_TOKEN=123:ABC\nTELEGRAM_CHAT_ID=1015524262\n", encoding="utf-8")
                with self.assertRaises(MissingTelegramCredentialsError) as ctx:
                    load_telethon_credentials(partial_env)
                err_txt = str(ctx.exception)
                self.assertIn("TELEGRAM_API_ID", err_txt)
                self.assertIn("TELEGRAM_API_HASH", err_txt)
                self.assertIn("my.telegram.org", err_txt)

                full_env = tdir / ".env.full"
                full_env.write_text(
                    "TELEGRAM_API_ID=987654\nTELEGRAM_API_HASH=abc123hash\nTELEGRAM_BOT_TOKEN=123:ABC\nTELEGRAM_CHAT_ID=1015524262\n",
                    encoding="utf-8",
                )
                creds = load_telethon_credentials(full_env)
                self.assertEqual(creds.api_id, 987654)
                self.assertEqual(creds.api_hash, "abc123hash")
                self.assertEqual(creds.chat_id, 1015524262)
        finally:
            for k, v in saved_env.items():
                if v is not None:
                    os.environ[k] = v

        # 3. Review mode toggle behavior
        self.assertFalse(should_send_raw_on_review({"review_mode": "streamable_preview", "send_raw_on_review": False}))
        self.assertTrue(should_send_raw_on_review({"review_mode": "streamable_preview", "send_raw_on_review": True}))
        self.assertTrue(should_send_raw_on_review({"review_mode": "raw_document", "send_raw_on_review": False}))

        # 4. Large (>50 MB) upload progress logging
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            small_f = tdir / "small.mp4"
            small_f.write_bytes(b"0" * 1024)
            self.assertIsNone(make_upload_progress_callback(small_f, threshold_mb=50.0))

            big_f = tdir / "big_master.mp4"
            with big_f.open("wb") as fh:
                fh.seek(55 * 1024 * 1024 - 1)
                fh.write(b"\0")

            logged_lines: list[str] = []

            class _MemHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    logged_lines.append(self.format(record))

            test_log = logging.getLogger("test_mtproto_progress")
            test_log.setLevel(logging.INFO)
            test_log.propagate = False
            mh = _MemHandler()
            test_log.addHandler(mh)
            try:
                cb = make_upload_progress_callback(big_f, target_logger=test_log, threshold_mb=50.0, step_percent=25)
                assert cb is not None
                total_b = big_f.stat().st_size
                cb(total_b // 2, total_b)
                cb(total_b, total_b)
            finally:
                test_log.removeHandler(mh)
            self.assertTrue(any("50%" in ln and "big_master.mp4" in ln for ln in logged_lines))
            self.assertTrue(any("100%" in ln and "big_master.mp4" in ln for ln in logged_lines))

        # 5. Streamable review -> Approve (uncompressed force_document=True master + publish_kit.md + copy block) & unauthorized ignore
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            cfg_p = tdir / "config.yaml"
            cfg_p.write_text(
                (repo_root / "config.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            run_id = "run_mtproto_test"
            run_dir = tdir / "tiktok" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            master_mp4 = run_dir / "final_9x16.mp4"
            master_mp4.write_bytes(b"FAKE_RAW_4K_MASTER" * 512)
            (run_dir / "review_package.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "topic": "GTA VI Trailer Physics",
                        "duration_sec": 72.4,
                        "metadata": {
                            "tiktok_title": "GTA VI Physics",
                            "tiktok_caption": "Did you catch this detail?",
                            "tiktok_hashtags": ["gtavi", "gaming"],
                            "tiktok_disclose_ai_content": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            bot_runner = InteractiveTikTokBot(
                config_path=cfg_p,
                state_path=tdir / "state.json",
                creds=TelegramCredentials(12345, "hash_abc", "123:tok", 1015524262),
            )
            bot_runner.store.set_dashboard_message(100)
            job = {
                "job_id": "job_101_0",
                "run_id": run_id,
                "card_message_id": 101,
                "category": "games",
                "category_ar": "ألعاب",
                "trend_id": "g1",
                "trend_title": "GTA VI Trailer Physics",
                "sourcing_keywords": ["gta 6"],
                "status": "producing",
            }
            bot_runner.store.state["jobs"][job["job_id"]] = job
            bot_runner.store.save()

            sent_files: list[dict[str, object]] = []
            sent_texts: list[dict[str, object]] = []
            edited_calls: list[dict[str, object]] = []
            next_id = 300

            class _FakeMTProtoClient:
                async def send_file(
                    self,
                    entity: int,
                    file_path: str,
                    *,
                    caption: str = "",
                    force_document: bool = False,
                    supports_streaming: bool = False,
                    attributes=None,
                    thumb=None,
                    buttons=None,
                    progress_callback=None,
                ):
                    nonlocal next_id
                    next_id += 1
                    sent_files.append(
                        {
                            "id": next_id,
                            "file": Path(file_path).name,
                            "caption": caption,
                            "force_document": force_document,
                            "supports_streaming": supports_streaming,
                            "attributes": attributes,
                            "buttons": buttons,
                        }
                    )
                    return SimpleNamespace(id=next_id)

                async def send_message(self, entity: int, text: str, buttons=None, parse_mode=None, link_preview: bool = False):
                    nonlocal next_id
                    next_id += 1
                    sent_texts.append({"id": next_id, "text": text, "buttons": buttons, "parse_mode": parse_mode})
                    return SimpleNamespace(id=next_id)

                async def edit_message(self, entity: int, message: int, text: str = None, buttons=None, link_preview: bool = False, file=None):
                    edited_calls.append({"id": int(message), "text": text, "buttons": buttons})

                async def delete_messages(self, entity: int, message_ids: list[int], revoke: bool = True):
                    return None

            class _FakeEv:
                def __init__(self, chat_id: int, sender_id: int, data: bytes, message_id: int, client) -> None:
                    self.chat_id = chat_id
                    self.sender_id = sender_id
                    self.data = data
                    self.message_id = message_id
                    self.client = client
                    self.answered = False

                async def answer(self, text: str = "", alert: bool = False) -> None:
                    self.answered = True

            fake_client = _FakeMTProtoClient()
            fake_gate = {"passed": True, "duration_sec": 72.4, "width": 2160, "height": 3840, "issues": []}

            async def _run_flow() -> None:
                # Unauthorized callback must be silently ignored
                unauth_ev = _FakeEv(9999999, 9999999, b"appr:job_101_0", 301, fake_client)
                await bot_runner.on_callback(unauth_ev)
                self.assertFalse(unauth_ev.answered)
                self.assertEqual(sent_files, [])

                # Review delivery uses streamable playable video (force_document=False)
                with (
                    unittest.mock.patch("scripts.telegram_bot._run_subprocess_async", return_value=(0, "ok")),
                    unittest.mock.patch("scripts.telegram_bot.verify_video", return_value=fake_gate),
                ):
                    ok = await bot_runner._run_pipeline_and_deliver_review(fake_client, job)
                self.assertTrue(ok)
                self.assertEqual(len(sent_files), 1)
                self.assertEqual(sent_files[0]["file"], "final_9x16.mp4")
                self.assertIs(sent_files[0]["force_document"], False)
                self.assertIs(sent_files[0]["supports_streaming"], True)
                self.assertIsInstance(sent_files[0]["attributes"][0], DocumentAttributeVideo)
                review_mid = int(sent_files[0]["id"])

                # Approve triggers uncompressed master (force_document=True) + publish_kit.md + copy block
                appr_ev = _FakeEv(1015524262, 1015524262, b"appr:job_101_0", review_mid, fake_client)
                with unittest.mock.patch("scripts.telegram_bot._run_subprocess_async", return_value=(0, "ok")):
                    await bot_runner.on_callback(appr_ev)
                self.assertTrue(appr_ev.answered)

            asyncio.run(_run_flow())

            # Review buttons cleared via ReplyInlineMarkup(rows=[])
            cleared_edits = [e for e in edited_calls if e["id"] == 301 and isinstance(e["buttons"], ReplyInlineMarkup)]
            self.assertTrue(cleared_edits)
            self.assertEqual(cleared_edits[0]["buttons"].rows, [])

            # Uploaded raw master + publish_kit.md
            self.assertEqual(len(sent_files), 3)
            self.assertEqual(sent_files[1]["file"], "final_9x16.mp4")
            self.assertIs(sent_files[1]["force_document"], True)
            self.assertIs(sent_files[1]["supports_streaming"], False)
            self.assertEqual(sent_files[2]["file"], "publish_kit.md")
            self.assertIs(sent_files[2]["force_document"], True)

            # Copyable caption + تم النشر button
            self.assertIn("<pre>", str(sent_texts[-1]["text"]))
            pub_btn = sent_texts[-1]["buttons"][0][0]
            self.assertEqual(pub_btn.data, b"pub:job_101_0")


if __name__ == "__main__":
    unittest.main()
