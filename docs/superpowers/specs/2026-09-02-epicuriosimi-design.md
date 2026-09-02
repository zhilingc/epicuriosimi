# epicuriosimi — Design

Date: 2026-09-02
Status: Approved

## Summary

A daily ingredient-guessing game (Semantle-style) named **epicuriosimi**. Each day a
target ingredient is chosen from the epicure vocabulary; the player guesses ingredients
and sees, for each guess, three similarity scores against the target — one per epicure
model — plus a cold/tepid/warm/hot temperature per score. Guessing the exact target wins,
with confetti and a popup showing the guess count.

Similarity engine: the three epicure embedding models (static word2vec-style tables,
shared 1,790-ingredient vocab, 300 dims each):

- [Kaikaku/epicure-cooc](https://huggingface.co/Kaikaku/epicure-cooc) — how well the guess *pairs with* the target
- [Kaikaku/epicure-chem](https://huggingface.co/Kaikaku/epicure-chem) — how much the guess *shares the target's flavor profile*
- [Kaikaku/epicure-core](https://huggingface.co/Kaikaku/epicure-core) — how *similar* the guess is to the target

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Architecture | Static site + precomputed daily score JSON (no backend) |
| Hosting | Local only for now (`python -m http.server` or similar) |
| Target pool | Full 1,790-ingredient vocabulary |
| Temperature mapping | Rank-based bands: hot = top 10, warm = top 100, tepid = top 500, cold = rest (rank never displayed) |
| Guess input | Autocomplete dropdown over the fixed vocab; invalid guesses impossible |
| Stack | Vanilla JS frontend (no build step) + Python precompute script |

Accepted trade-off: the day's full score table (including the target) ships to the
browser, so the answer is discoverable in devtools. Fine for a local/for-fun game.

## Architecture

Two parts:

```
semancook/
  docs/                       # goal + this spec
  precompute/
    generate_daily.py         # CLI: downloads/caches models, writes daily JSON
    epicure_daily/            # importable module with the actual logic (tested)
    tests/                    # pytest
    requirements.txt          # numpy, safetensors, huggingface_hub, pytest
  site/
    index.html
    app.js
    style.css
    lib/confetti.js           # vendored canvas-confetti (works offline)
    vocab.json                # sorted ingredient list for autocomplete
    data/YYYY-MM-DD.json      # one file per puzzle day
```

### Precompute (Python)

`python precompute/generate_daily.py [--date YYYY-MM-DD] [--days N]`

1. **Model fetch**: `huggingface_hub` snapshot-downloads `embeddings.safetensors` +
   `vocab.json` for each of the three models into a local cache dir (gitignored).
   Validates all three vocabs are identical; fails loudly if not.
2. **Target selection**: deterministic —
   `int(SHA256("epicuriosimi:" + date)) % 1790` indexing into the sorted vocab.
   Same date always yields the same target; no state; future days pre-generable.
3. **Scoring**: L2-normalize embeddings (per the model cards), cosine similarity of
   the target against all 1,790 ingredients, per model.
4. **Bands**: rank each ingredient per measurement (target excluded from ranking);
   rank ≤ 10 → hot, ≤ 100 → warm, ≤ 500 → tepid, else cold.
5. **Output** `site/data/<date>.json` (~100 KB):
   - `date`, `puzzle_number` (1-based: 2026-09-02 is puzzle #1), `target` (vocab index)
   - per-ingredient arrays (indexed by vocab position): three scores (cosine, rounded
     to 4 decimals) and three band codes.

Also writes `site/vocab.json` (sorted vocab list) if missing or stale.

### Frontend (vanilla JS)

- On load: fetch `vocab.json` + today's `data/<local-date>.json`. Missing data file →
  friendly message telling the user to run the generate script.
- **Input**: textbox with autocomplete dropdown; substring match on display names
  (underscores shown as spaces); arrow keys + Enter (Enter with no selection submits
  the top match). Duplicate guesses refused with a notice.
- **Tables**: three, side by side on desktop, stacked on narrow screens —
  "Pairs well with", "Shares flavor profile", "Similar to". Each lists all guesses
  sorted descending by that measurement's score. Columns: ingredient, score
  (cosine × 100, 2 decimals), temperature badge (cold=blue, tepid=teal, warm=orange,
  hot=red). Most recent guess highlighted in each table. No rank column.
- **Persistence**: guesses in `localStorage` keyed by date; refresh keeps progress
  (wrapped in try/catch; game still works without storage).
- **Win**: guess index === target index → confetti burst (vendored canvas-confetti)
  and a dismissible popup: "You got it — *ingredient* in N guesses!" Input disabled
  after winning; tables remain visible.

## Testing

All meaningful logic lives in Python and is pytest-covered:

- deterministic target for a fixed date (golden value)
- band boundaries exactly at ranks 10 / 100 / 500; target excluded from ranks
- embeddings L2-normalized before cosine; target's own score = 1.0
- vocab-mismatch failure path
- output JSON shape (keys, lengths, value ranges)

The JS layer is thin display + autocomplete; verified manually by loading the page
against a generated fixture day.

## Out of scope (YAGNI)

- Deployment/hosting, cron automation (revisit when going beyond local)
- Hint systems, streaks, share buttons, stats
- Fuzzy/free-text guess matching
- Hiding the answer from devtools
