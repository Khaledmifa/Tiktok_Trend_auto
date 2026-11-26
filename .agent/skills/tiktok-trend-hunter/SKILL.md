---
name: tiktok-trend-hunter
description: >-
  Discovers and selects the #1 viable global TikTok trend at $0 cost using Antigravity's
  built-in Browser (TikTok Creative Center, Google Trends, YouTube Trending), web search,
  local trends_history.json deduplication, and automated content/copyright filtering.
  Use when hunting new TikTok topics or running --auto-top-trend discovery in d:\Money.
---

# TikTok Trend Hunter (`$0` Free Browser Extension)

## Primary Public Browser Targets (No Login Walls, No Paid APIs)
Invoke Antigravity's built-in `browser` subagent strictly with **`Model: "flash"`** (lightweight/fast):
1. **TikTok Creative Center:** `https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en` (with optional `US`/`GB`/`DE`/`FR` region filter when multi-language runs are requested).
2. **Google Trends Trending Now:** `https://trends.google.com/trending?geo=US` (or `DE`/`FR` for localized runs).
3. **YouTube Trending:** `https://www.youtube.com/feed/trending` to cross-verify momentum and ensure high-resolution real footage (`1080p`) is available for editing.

## Token-Saving & Anti-Block Browser Rules
- Cap navigation to **2–3 pages maximum** per run.
- Extract **DOM text and tables directly** rather than taking excessive screenshots or endless scrolling.
- If any page triggers a CAPTCHA, anti-bot verification, or login wall, **skip it immediately** (never attempt to click through or bypass it) and move cleanly to the next public target or `search_web`.

## Zero-Token Filtering, Scoring & Persistent History
Pass raw candidates (JSON) into `d:\Money\scripts\trend_history_manager.py`:
```powershell
python scripts/trend_history_manager.py --input <candidates.json>
```
This zero-token script automatically:
1. Maintains `d:\Money\data\trends_history.json` (seeding already-produced runs from `data/history_log.json` and logging every `discovered`, `rejected`, and `selected` topic with timestamp).
2. Filters out:
   - Previously covered/produced topics (`similarity >= 0.50` or substring match).
   - Kids' content (`kids`, `toddler`, `cocomelon`, `nursery rhyme`, `baby shark`, etc.).
   - Pure dance / lip-sync trends with no narrative (`dance challenge`, `lip sync`, `choreography`, etc.).
   - High-copyright celebrity/live-broadcast traps (`full movie`, `leaked clip`, `music video`, `nfl/nba highlights`, `ufc fight`, `celebrity gossip`, `deepfake`, `reupload`).
3. Scores remaining topics for **65–85s analytical/storytelling TikTok** viability (**Viral Studio Pro**, adult audience `18–45`) and picks the **#1 strongest trend + 2 backups**.
4. Writes the daily summary strictly in clear Arabic to `d:\Money\output\latest_trends.md` (with selected angle, 3s hook idea, and English clip-sourcing keywords).
5. Stops at human review — **never** triggers heavy rendering, large video downloads, or public uploads automatically.

## 4 Connected Content Pillars & Live Sources (`data/trend_sources.json`)
Every `/trends` run or daily 9:00 AM scan MUST re-scan `d:\Money\data\trend_sources.json` via `scripts/fetch_live_trends.py`, enforcing strict cross-pillar entity deduplication and excluding hardware reviews, unboxings, shopping recommendations, physical product tests, kids/dance traps, and everything in `d:\Money\data\trends_history.json`:
1. **`أفلام` (`movies`):** Timely movie news, hidden details, trailer analysis, and original interpretations (Sources: Variety Film, Deadline Film, THR Movies, Collider, IndieWire, Screen Rant, Box Office Mojo, Rotten Tomatoes, IMDb).
2. **`مسلسلات` (`series`):** Episode details, evidence-grounded theories, upcoming releases, and cultural conversations (Sources: Variety TV, Deadline TV, THR TV, TVLine, EW TV, Vulture TV, Netflix Top 10, IMDb TV).
3. **`ألعاب` (`games`):** Major updates, reveals, gameplay changes, and stories behind trending games (Sources: IGN, Eurogamer, PC Gamer, Kotaku, Polygon, GameSpot, VGC, PlayStation Blog, Steam News, SteamDB).
4. **`ذكاء اصطناعي` (`ai`):** Noteworthy releases, surprising capabilities, creative applications, and impact on entertainment & gaming (Sources: TechCrunch AI, The Verge AI, Ars Technica, MIT Tech Review AI, VentureBeat AI, OpenAI, DeepMind, Anthropic, Hugging Face, arXiv, Simon Willison).

**Reference Accounts:** `@ign`, `@straw_hat_goofy`, `@billyfx` (hardware/product review accounts excluded).

## Interactive Telegram Trend-Card Pool & Daily Refresh (`$0` Free)
- To re-scan all sources in `data/trend_sources.json`, update `data/trends_queue.json` & `data/latest_trends.md`, and push fresh 4-pillar photo cards to Telegram:
  ```powershell
  python scripts/fetch_live_trends.py
  python scripts/trend_cards_pool.py --dry-run
  python scripts/telegram_bot.py --send-cards-once
  ```
