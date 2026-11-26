---
name: viral-script-multilang
description: >-
  Authors high-retention 65-85s TikTok Creator Rewards scripts strictly in English
  (10-12 scenes, 3s curiosity-gap hook, ~35s editorial take, seamless loop ending)
  matched to free edge-tts neural voices, auto-translating any non-English source
  material into English first.
---

# Viral Script Blueprint (`65–85s` Creator Rewards — Strictly English)

## Strict English-Only & Auto-Translation Rule
- **100% English Videos Only:** Every spoken word, burned-in caption, HUD headline/fact card/badge, and TikTok publish kit (`publish_kit.md`) MUST be in English (`en` only, voice `en-US-AndrewMultilingualNeural`).
- **Auto-Translate Foreign Sources:** Even when source articles, RSS items, or web snippets arrive in German, Arabic, French, or another language, translate them into natural English via `src/utils/language_guard.py` before scripting or narration.
- **Arabic Chat Only:** Communicate with the user in clear Arabic in chat, never inside generated videos.

## Retention & Format Rules
- **Monetization Floor:** Rendered runtime MUST exceed **61s** (target **65–85s**).
- **Word Counts Across 10–12 Scenes:**
  - `en`: **155–185 words** -> Voice: `en-US-AndrewMultilingualNeural`
- **Visual Prompt Budget:** Keep every `visual_ai_prompt` strictly `<= 15` words (set `video_motion_prompt: "live_broll"` when real clips exist).

## Narrative Blueprint (Editorial + Seamless Loop Hybrid)
1. **0–3s (Sharp Framing Hook):** Concrete distinction, verified date/number, or clear editorial framing hook (avoid vague clickbait), written *together* with the closing connector.
2. **3–35s (Fact-Grounded Escalation):** Crisp authoritative sentences (`8–14` words each) separating confirmed facts from noise.
3. **~35s (Editorial Take / Mechanism):** Reveal the core design, mechanism, or turning point ("My take? ...").
4. **60–85s (Complete Payoff + CTA Before Seamless Loop Bridge):** Deliver the complete answer before the ending, place any follow/save invitation naturally *before* the final connecting phrase, and end on a topic-adapted bridge clause that flows grammatically and logically back into the opening line on replay (never force `"And that's why..."`).

## Editorial HUD & AI Coverage Rules
- Populate `category_label` (`MOVIES EXPLAINED`, `SERIES EXPLAINED`, `GAMING EXPLAINED`, `AI EXPLAINED`), `footage_context_label`, `attribution_line`, and per-scene `kicker`, `headline_line1`, `headline_line2`, `fact_card` (only on concrete-metric scenes so screens stay <= 3 text layers), and alternating `layout_mode` (`wide_panel` / `fullscreen_portrait`).
- For `ai` stories: explicitly distinguish `RELEASED FEATURE` vs `ANNOUNCEMENT / DEMO` vs `RUMOR / SPECULATION`, and tag any synthetic/generated visuals with `SYNTHETIC / AI-GENERATED` in `footage_context`.
- Exclude hardware reviews, unboxings, shopping recommendations, and physical product tests.
- `ScriptArchitectAgent._build_plan_from_raw` auto-derives clean defaults if any field is omitted.

## Single-Language Output Convention
- All runs write strictly in English to `tiktok/<run_id>/`.
- Never call paid translation or external LLM APIs; validate via `ScriptArchitectAgent.load_from_json(..., skip_external_llm=True)`.
