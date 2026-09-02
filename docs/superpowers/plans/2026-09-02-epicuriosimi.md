# epicuriosimi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local static daily ingredient-guessing game (Semantle-style) scored by the three epicure embedding models, with a Python precompute step generating per-day JSON.

**Architecture:** A Python package (`precompute/epicure_daily/`) downloads the three epicure models from HuggingFace, picks a deterministic daily target, and writes `site/data/YYYY-MM-DD.json` containing every ingredient's three cosine scores and temperature bands. A vanilla-JS static site (`site/`) loads that JSON and runs the whole game client-side. No backend, no build step.

**Tech Stack:** Python 3 (numpy, safetensors, huggingface_hub, pytest), vanilla HTML/CSS/JS, vendored canvas-confetti.

**Spec:** `docs/superpowers/specs/2026-09-02-epicuriosimi-design.md`

## Global Constraints

- Game name is lowercase everywhere: **epicuriosimi**.
- Models: `Kaikaku/epicure-cooc`, `Kaikaku/epicure-chem`, `Kaikaku/epicure-core`. All share one vocab: 1,790 ingredients, indices 0..1789, alphabetical order. Embeddings are 1790×300 F32 under safetensors key `"embeddings"` (verified against the real repos during planning).
- Target selection: `int(SHA256("epicuriosimi:" + YYYY-MM-DD hex digest), 16) % 1790` into the vocab-index-ordered word list. Golden value: date `2026-09-02` → index `1006` → `merguez` (verified against the real vocab).
- Temperature bands from per-measurement rank (target excluded): rank ≤ 10 → hot(3), ≤ 100 → warm(2), ≤ 500 → tepid(1), else cold(0). Rank is never displayed.
- Puzzle numbering is 1-based: 2026-09-02 is puzzle #1.
- Scores stored as cosine rounded to 4 decimals; displayed as cosine × 100 with 2 decimals.
- Embeddings are L2-normalized before cosine (per model cards).
- Frontend: no framework, no build step, no CDN at runtime (confetti is vendored).
- All Python commands below assume the venv: `precompute/.venv/bin/python` / `precompute/.venv/bin/pytest`. Run everything from the repo root (`/Users/zchen/Documents/etc/semancook`).
- Use `rg`, not `grep`, for any searching.

---

### Task 1: Precompute scaffolding + model loading

**Files:**
- Create: `.gitignore`
- Create: `precompute/requirements.txt`
- Create: `precompute/conftest.py`
- Create: `precompute/epicure_daily/__init__.py`
- Create: `precompute/epicure_daily/models.py`
- Test: `precompute/tests/test_models.py`

**Interfaces:**
- Produces: `models.MODELS: dict[str, str]` (key → HF repo id); `models.load_model_files(model_dir: Path) -> tuple[list[str], np.ndarray]`; `models.load_all(model_dirs: dict[str, Path]) -> tuple[list[str], dict[str, np.ndarray]]` (words list + `{"cooc"|"chem"|"core": (N,300) float32}`); `models.download_models(cache_dir: Path) -> dict[str, Path]`.

- [ ] **Step 1: Scaffolding files**

`.gitignore`:

```
__pycache__/
.pytest_cache/
precompute/.venv/
precompute/.model_cache/
site/data/
site/vocab.json
```

`precompute/requirements.txt`:

```
numpy>=1.24
safetensors>=0.4
huggingface_hub>=0.20
pytest>=8
```

`precompute/conftest.py` (makes `epicure_daily` importable when running pytest from repo root):

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

`precompute/epicure_daily/__init__.py`: empty file.

- [ ] **Step 2: Create venv and install deps**

Run:
```bash
python3 -m venv precompute/.venv
precompute/.venv/bin/pip install -r precompute/requirements.txt
```
Expected: installs succeed; `precompute/.venv/bin/pytest --version` prints a version.

- [ ] **Step 3: Write the failing tests**

`precompute/tests/test_models.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from epicure_daily import models


def make_model_dir(tmp_path: Path, name: str, words: list[str], embeddings: np.ndarray,
                   vocab_override: dict | None = None) -> Path:
    d = tmp_path / name
    d.mkdir()
    vocab = vocab_override if vocab_override is not None else {w: i for i, w in enumerate(words)}
    (d / "vocab.json").write_text(json.dumps(vocab))
    save_file({"embeddings": embeddings.astype(np.float32)}, str(d / "embeddings.safetensors"))
    return d


WORDS = ["apple", "basil", "carrot"]
EMB = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0


def test_load_model_files_roundtrip(tmp_path):
    d = make_model_dir(tmp_path, "m", WORDS, EMB)
    words, emb = models.load_model_files(d)
    assert words == WORDS
    assert emb.shape == (3, 4)
    assert emb.dtype == np.float32
    np.testing.assert_array_equal(emb, EMB)


def test_load_model_files_respects_vocab_index_order(tmp_path):
    # vocab dict insertion order differs from index order; words must follow indices
    vocab = {"carrot": 2, "apple": 0, "basil": 1}
    d = make_model_dir(tmp_path, "m", WORDS, EMB, vocab_override=vocab)
    words, _ = models.load_model_files(d)
    assert words == ["apple", "basil", "carrot"]


def test_vocab_values_must_be_contiguous(tmp_path):
    d = make_model_dir(tmp_path, "m", WORDS, EMB, vocab_override={"apple": 0, "basil": 2, "carrot": 3})
    with pytest.raises(ValueError, match="0..N-1"):
        models.load_model_files(d)


def test_embedding_rows_must_match_vocab(tmp_path):
    d = make_model_dir(tmp_path, "m", WORDS, np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="vocab size"):
        models.load_model_files(d)


def test_load_all_returns_all_keys(tmp_path):
    dirs = {k: make_model_dir(tmp_path, k, WORDS, EMB * (i + 1))
            for i, k in enumerate(["cooc", "chem", "core"])}
    words, embeddings = models.load_all(dirs)
    assert words == WORDS
    assert set(embeddings) == {"cooc", "chem", "core"}
    np.testing.assert_array_equal(embeddings["chem"], EMB * 2)


def test_load_all_rejects_vocab_mismatch(tmp_path):
    dirs = {
        "cooc": make_model_dir(tmp_path, "cooc", WORDS, EMB),
        "chem": make_model_dir(tmp_path, "chem", ["apple", "basil", "cumin"], EMB),
    }
    with pytest.raises(ValueError, match="vocab mismatch"):
        models.load_all(dirs)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `precompute/.venv/bin/pytest precompute/tests/test_models.py -v`
Expected: FAIL — `AttributeError` / import errors (`models` has no functions yet).

- [ ] **Step 5: Write the implementation**

`precompute/epicure_daily/models.py`:

```python
"""Download and load the three epicure embedding models."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

MODELS = {
    "cooc": "Kaikaku/epicure-cooc",
    "chem": "Kaikaku/epicure-chem",
    "core": "Kaikaku/epicure-core",
}

TENSOR_KEY = "embeddings"


def load_model_files(model_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load one model dir containing vocab.json + embeddings.safetensors."""
    vocab = json.loads((model_dir / "vocab.json").read_text())
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError(f"{model_dir}: vocab values are not exactly 0..N-1")
    words = sorted(vocab, key=vocab.__getitem__)
    embeddings = load_file(str(model_dir / "embeddings.safetensors"))[TENSOR_KEY]
    if embeddings.shape[0] != len(words):
        raise ValueError(
            f"{model_dir}: embeddings rows {embeddings.shape[0]} != vocab size {len(words)}"
        )
    return words, embeddings.astype(np.float32)


def load_all(model_dirs: dict[str, Path]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load every model; require identical vocabs. Returns (words, {key: embeddings})."""
    words = None
    embeddings = {}
    for key, model_dir in model_dirs.items():
        w, emb = load_model_files(model_dir)
        if words is None:
            words = w
        elif w != words:
            raise ValueError(f"vocab mismatch: {key} differs from previously loaded model")
        embeddings[key] = emb
    return words, embeddings


def download_models(cache_dir: Path) -> dict[str, Path]:
    """Snapshot-download vocab + embeddings for each model. Returns {key: local dir}."""
    from huggingface_hub import snapshot_download

    return {
        key: Path(
            snapshot_download(
                repo,
                allow_patterns=["vocab.json", "embeddings.safetensors"],
                cache_dir=str(cache_dir),
            )
        )
        for key, repo in MODELS.items()
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `precompute/.venv/bin/pytest precompute/tests/test_models.py -v`
Expected: 6 passed. (`download_models` is exercised in Task 3's integration step, not unit-tested.)

- [ ] **Step 7: Commit**

```bash
git add .gitignore precompute/
git commit -m "feat: precompute scaffolding and epicure model loading"
```

---

### Task 2: Puzzle logic (target, scores, bands)

**Files:**
- Create: `precompute/epicure_daily/puzzle.py`
- Test: `precompute/tests/test_puzzle.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure numpy/stdlib).
- Produces: `puzzle.LAUNCH_DATE: datetime.date`; `puzzle.target_index(date_str: str, vocab_size: int) -> int`; `puzzle.puzzle_number(d: date) -> int`; `puzzle.cosine_scores(embeddings: np.ndarray, target_idx: int) -> np.ndarray`; `puzzle.band_codes(scores: np.ndarray, target_idx: int) -> np.ndarray` (int codes 0=cold 1=tepid 2=warm 3=hot); `puzzle.build_puzzle(d: date, embeddings: dict[str, np.ndarray]) -> dict` with keys `date`, `puzzle_number`, `target`, `scores` (`{key: [float×N]}`), `bands` (`{key: [int×N]}`).

- [ ] **Step 1: Write the failing tests**

`precompute/tests/test_puzzle.py`:

```python
from datetime import date

import numpy as np

from epicure_daily import puzzle


def test_target_index_golden():
    # Verified against the real vocab during planning: 2026-09-02 -> merguez (index 1006)
    assert puzzle.target_index("2026-09-02", 1790) == 1006


def test_target_index_deterministic_and_date_sensitive():
    a = puzzle.target_index("2026-09-03", 1790)
    assert a == puzzle.target_index("2026-09-03", 1790)
    assert 0 <= a < 1790
    days = {puzzle.target_index(f"2026-10-{d:02d}", 1790) for d in range(1, 11)}
    assert len(days) > 1  # not constant across dates


def test_puzzle_number_is_one_based_from_launch():
    assert puzzle.puzzle_number(date(2026, 9, 2)) == 1
    assert puzzle.puzzle_number(date(2026, 9, 11)) == 10


def test_cosine_scores_normalizes_before_dot():
    emb = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    s = puzzle.cosine_scores(emb, 0)
    np.testing.assert_allclose(s, [1.0, 0.0, np.sqrt(0.5)], atol=1e-6)


def test_band_codes_boundaries():
    # 600 items, scores strictly descending by index, target = 0 (the top score).
    # Non-target item i then has rank i exactly.
    scores = 1.0 - np.arange(600, dtype=np.float64) * 1e-3
    codes = puzzle.band_codes(scores, 0)
    assert codes[0] == 3     # target itself counts as hot
    assert codes[10] == 3    # rank 10 -> hot
    assert codes[11] == 2    # rank 11 -> warm
    assert codes[100] == 2   # rank 100 -> warm
    assert codes[101] == 1   # rank 101 -> tepid
    assert codes[500] == 1   # rank 500 -> tepid
    assert codes[501] == 0   # rank 501 -> cold


def test_band_codes_excludes_target_from_ranking():
    # Target in the middle: items scoring above it still get ranks 1..k with no gap.
    scores = np.array([0.9, 0.8, 0.95, 0.7])
    codes = puzzle.band_codes(scores, 2)
    assert codes[2] == 3
    assert codes[0] == 3  # rank 1
    assert codes[1] == 3  # rank 2
    assert codes[3] == 3  # rank 3 (tiny vocab: everything is top-10)


def test_build_puzzle_shape_and_values():
    rng = np.random.default_rng(42)
    embeddings = {k: rng.normal(size=(20, 4)).astype(np.float32) for k in ["cooc", "chem", "core"]}
    p = puzzle.build_puzzle(date(2026, 9, 2), embeddings)
    assert p["date"] == "2026-09-02"
    assert p["puzzle_number"] == 1
    t = p["target"]
    assert 0 <= t < 20
    assert set(p["scores"]) == set(p["bands"]) == {"cooc", "chem", "core"}
    for key in embeddings:
        assert len(p["scores"][key]) == 20
        assert len(p["bands"][key]) == 20
        assert p["scores"][key][t] == 1.0  # self-cosine rounds to exactly 1.0
        assert p["bands"][key][t] == 3
        assert all(-1.0001 <= s <= 1.0001 for s in p["scores"][key])
        assert all(b in (0, 1, 2, 3) for b in p["bands"][key])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `precompute/.venv/bin/pytest precompute/tests/test_puzzle.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` (puzzle module doesn't exist).

- [ ] **Step 3: Write the implementation**

`precompute/epicure_daily/puzzle.py`:

```python
"""Daily target selection, cosine scoring, and temperature bands."""
import hashlib
from datetime import date

import numpy as np

LAUNCH_DATE = date(2026, 9, 2)

# Band codes: 0=cold, 1=tepid, 2=warm, 3=hot. Rank thresholds per spec.
_BAND_THRESHOLDS = ((10, 3), (100, 2), (500, 1))


def target_index(date_str: str, vocab_size: int) -> int:
    digest = hashlib.sha256(f"epicuriosimi:{date_str}".encode()).hexdigest()
    return int(digest, 16) % vocab_size


def puzzle_number(d: date) -> int:
    return (d - LAUNCH_DATE).days + 1


def cosine_scores(embeddings: np.ndarray, target_idx: int) -> np.ndarray:
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normed @ normed[target_idx]


def band_codes(scores: np.ndarray, target_idx: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    order = order[order != target_idx]
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    ranks[target_idx] = 0  # target ranks best -> hot
    codes = np.zeros(len(scores), dtype=np.int64)
    for threshold, code in reversed(_BAND_THRESHOLDS):
        codes[ranks <= threshold] = code
    return codes


def build_puzzle(d: date, embeddings: dict[str, np.ndarray]) -> dict:
    vocab_size = next(iter(embeddings.values())).shape[0]
    date_str = d.isoformat()
    t = target_index(date_str, vocab_size)
    scores = {}
    bands = {}
    for key, emb in embeddings.items():
        s = cosine_scores(emb, t)
        scores[key] = [round(float(x), 4) for x in s]
        bands[key] = band_codes(s, t).tolist()
    return {
        "date": date_str,
        "puzzle_number": puzzle_number(d),
        "target": t,
        "scores": scores,
        "bands": bands,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `precompute/.venv/bin/pytest precompute/tests/test_puzzle.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add precompute/epicure_daily/puzzle.py precompute/tests/test_puzzle.py
git commit -m "feat: puzzle logic - deterministic target, cosine scores, temperature bands"
```

---

### Task 3: CLI generator + real-model integration run

**Files:**
- Create: `precompute/generate_daily.py`
- Test: `precompute/tests/test_generate.py`

**Interfaces:**
- Consumes: `models.download_models`, `models.load_all` (Task 1); `puzzle.build_puzzle` (Task 2).
- Produces: `generate_daily.write_puzzles(start: date, days: int, words: list[str], embeddings: dict[str, np.ndarray], site_dir: Path, reveal: bool = False) -> list[Path]`; CLI `python precompute/generate_daily.py [--date YYYY-MM-DD] [--days N] [--reveal]`. Output files: `site/vocab.json` (JSON list of words, vocab-index order) and `site/data/<date>.json` (the `build_puzzle` dict).

- [ ] **Step 1: Write the failing tests**

`precompute/tests/test_generate.py`:

```python
import json
from datetime import date

import numpy as np

import generate_daily


def make_embeddings(n=20, dim=4):
    rng = np.random.default_rng(7)
    return {k: rng.normal(size=(n, dim)).astype(np.float32) for k in ["cooc", "chem", "core"]}


WORDS = [f"ing_{i:02d}" for i in range(20)]


def test_write_puzzles_creates_data_and_vocab(tmp_path):
    site = tmp_path / "site"
    written = generate_daily.write_puzzles(date(2026, 9, 2), 3, WORDS, make_embeddings(), site)
    assert [p.name for p in written] == ["2026-09-02.json", "2026-09-03.json", "2026-09-04.json"]
    assert json.loads((site / "vocab.json").read_text()) == WORDS
    first = json.loads(written[0].read_text())
    assert first["date"] == "2026-09-02"
    assert first["puzzle_number"] == 1
    assert len(first["scores"]["core"]) == 20


def test_write_puzzles_reveal_prints_target(tmp_path, capsys):
    generate_daily.write_puzzles(date(2026, 9, 2), 1, WORDS, make_embeddings(), tmp_path / "site",
                                 reveal=True)
    out = capsys.readouterr().out
    assert "target=ing_" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `precompute/.venv/bin/pytest precompute/tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_daily'`. (The conftest from Task 1 puts `precompute/` on `sys.path`, so once the file exists it imports.)

- [ ] **Step 3: Write the implementation**

`precompute/generate_daily.py`:

```python
"""Generate epicuriosimi daily puzzle JSON files.

Usage: python precompute/generate_daily.py [--date YYYY-MM-DD] [--days N] [--reveal]
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from epicure_daily.models import download_models, load_all
from epicure_daily.puzzle import build_puzzle

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
CACHE_DIR = REPO_ROOT / "precompute" / ".model_cache"


def write_puzzles(start: date, days: int, words: list[str],
                  embeddings: dict[str, np.ndarray], site_dir: Path,
                  reveal: bool = False) -> list[Path]:
    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "vocab.json").write_text(json.dumps(words))
    written = []
    for offset in range(days):
        d = start + timedelta(days=offset)
        puzzle = build_puzzle(d, embeddings)
        path = data_dir / f"{d.isoformat()}.json"
        path.write_text(json.dumps(puzzle))
        written.append(path)
        line = f"wrote {path} (puzzle #{puzzle['puzzle_number']})"
        if reveal:
            line += f" target={words[puzzle['target']]}"
        print(line)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate epicuriosimi daily puzzle files.")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="start date YYYY-MM-DD (default: today)")
    parser.add_argument("--days", type=int, default=1,
                        help="number of consecutive days to generate (default: 1)")
    parser.add_argument("--reveal", action="store_true",
                        help="print each day's target ingredient (spoiler!)")
    args = parser.parse_args()

    model_dirs = download_models(CACHE_DIR)
    words, embeddings = load_all(model_dirs)
    write_puzzles(date.fromisoformat(args.date), args.days, words, embeddings, SITE_DIR,
                  reveal=args.reveal)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `precompute/.venv/bin/pytest precompute/tests -v`
Expected: all tests pass (Tasks 1–3).

- [ ] **Step 5: Integration run against the real models**

Run: `precompute/.venv/bin/python precompute/generate_daily.py --date 2026-09-02 --days 1 --reveal`
Expected: first run downloads ~13 MB of model files into `precompute/.model_cache/`, then prints a `wrote .../site/data/2026-09-02.json (puzzle #1) target=merguez` line. Verify:

```bash
precompute/.venv/bin/python - <<'EOF'
import json
p = json.load(open("site/data/2026-09-02.json"))
assert p["target"] == 1006 and p["puzzle_number"] == 1
for k in ("cooc", "chem", "core"):
    assert len(p["scores"][k]) == 1790 and p["scores"][k][1006] == 1.0 and p["bands"][k][1006] == 3
vocab = json.load(open("site/vocab.json"))
assert len(vocab) == 1790 and vocab[1006] == "merguez"
print("integration OK")
EOF
```
Expected: `integration OK`.

- [ ] **Step 6: Commit**

```bash
git add precompute/generate_daily.py precompute/tests/test_generate.py
git commit -m "feat: daily puzzle generator CLI"
```

(`site/data/` and `site/vocab.json` are gitignored — generated artifacts.)

---

### Task 4: Site shell — HTML, CSS, vendored confetti

**Files:**
- Create: `site/index.html`
- Create: `site/style.css`
- Create: `site/lib/confetti.js` (vendored)

**Interfaces:**
- Produces: DOM ids consumed by Task 5's `app.js`: `puzzle-number`, `message`, `guess-form`, `guess-input`, `suggestions`, `notice`, `table-cooc`, `table-chem`, `table-core` (each with a `tbody`), `win-modal`, `win-text`, `win-close`. Global `confetti()` function from the vendored script. CSS classes consumed by app.js: `band band-cold|band-tepid|band-warm|band-hot`, `latest`, `active` (suggestion highlight), `num`.

- [ ] **Step 1: Vendor canvas-confetti**

```bash
mkdir -p site/lib
curl -sL https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js -o site/lib/confetti.js
rg -c "confetti" site/lib/confetti.js
```
Expected: file exists, >10 KB, rg count ≥ 1.

- [ ] **Step 2: Write index.html**

`site/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>epicuriosimi</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>epicuriosimi</h1>
    <p class="tagline">guess the daily ingredient <span id="puzzle-number"></span></p>
  </header>
  <main>
    <div id="message" class="message" hidden></div>
    <form id="guess-form" autocomplete="off">
      <div class="input-wrap">
        <input id="guess-input" type="text" placeholder="type an ingredient…"
               spellcheck="false" autofocus>
        <ul id="suggestions" hidden></ul>
      </div>
      <button type="submit">guess</button>
    </form>
    <p id="notice" class="notice" hidden></p>
    <div class="tables">
      <section>
        <h2>pairs well with</h2>
        <table id="table-cooc">
          <thead><tr><th>ingredient</th><th class="num">score</th><th>heat</th></tr></thead>
          <tbody></tbody>
        </table>
      </section>
      <section>
        <h2>shares flavor profile</h2>
        <table id="table-chem">
          <thead><tr><th>ingredient</th><th class="num">score</th><th>heat</th></tr></thead>
          <tbody></tbody>
        </table>
      </section>
      <section>
        <h2>similar to</h2>
        <table id="table-core">
          <thead><tr><th>ingredient</th><th class="num">score</th><th>heat</th></tr></thead>
          <tbody></tbody>
        </table>
      </section>
    </div>
  </main>
  <div id="win-modal" class="modal" hidden>
    <div class="modal-box">
      <h2>you got it! 🎉</h2>
      <p id="win-text"></p>
      <button id="win-close">close</button>
    </div>
  </div>
  <script src="lib/confetti.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Write style.css**

`site/style.css`:

```css
* { box-sizing: border-box; }
[hidden] { display: none !important; }

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #faf7f2;
  color: #2d2a26;
  min-height: 100vh;
}

header { text-align: center; padding: 2rem 1rem 0.5rem; }
h1 { margin: 0; font-size: 2.2rem; letter-spacing: 0.02em; }
.tagline { margin: 0.3rem 0 0; color: #8a8378; }

main { max-width: 1100px; margin: 0 auto; padding: 1rem; }

.message {
  background: #fff3cd;
  border: 1px solid #e6d9a8;
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem auto;
  max-width: 480px;
  text-align: center;
}

#guess-form {
  display: flex;
  gap: 0.5rem;
  max-width: 480px;
  margin: 1rem auto;
}

.input-wrap { position: relative; flex: 1; }

#guess-input {
  width: 100%;
  font-size: 1.1rem;
  padding: 0.6rem 0.8rem;
  border: 2px solid #d8d2c8;
  border-radius: 8px;
  background: #fff;
}
#guess-input:focus { outline: none; border-color: #b0663c; }
#guess-input:disabled { background: #eee9e1; }

#guess-form button {
  font-size: 1.05rem;
  padding: 0.6rem 1.2rem;
  border: none;
  border-radius: 8px;
  background: #b0663c;
  color: #fff;
  cursor: pointer;
}
#guess-form button:disabled { background: #cbb9ac; cursor: default; }

#suggestions {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  z-index: 10;
  margin: 2px 0 0;
  padding: 0;
  list-style: none;
  background: #fff;
  border: 1px solid #d8d2c8;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  overflow: hidden;
}
#suggestions li { padding: 0.5rem 0.8rem; cursor: pointer; }
#suggestions li.active, #suggestions li:hover { background: #f3e7db; }

.notice { text-align: center; color: #a05a2c; min-height: 1.2em; margin: 0.2rem 0; }

.tables {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-top: 1rem;
}
@media (max-width: 800px) { .tables { grid-template-columns: 1fr; } }

.tables section {
  background: #fff;
  border: 1px solid #e5dfd5;
  border-radius: 10px;
  padding: 0.8rem;
  overflow-x: auto;
}
.tables h2 { margin: 0 0 0.5rem; font-size: 1rem; color: #6f675c; }

table { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
th { text-align: left; color: #8a8378; font-weight: 600; padding: 0.3rem 0.4rem; }
td { padding: 0.35rem 0.4rem; border-top: 1px solid #f0ebe2; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.latest td { background: #fdf0e3; font-weight: 600; }

.band {
  display: inline-block;
  padding: 0.1rem 0.55rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #fff;
}
.band-cold  { background: #3b82f6; }
.band-tepid { background: #14b8a6; }
.band-warm  { background: #f97316; }
.band-hot   { background: #ef4444; }

.modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.modal-box {
  background: #fff;
  border-radius: 12px;
  padding: 2rem 2.5rem;
  text-align: center;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
}
.modal-box h2 { margin-top: 0; }
.modal-box button {
  margin-top: 1rem;
  padding: 0.5rem 1.5rem;
  border: none;
  border-radius: 8px;
  background: #b0663c;
  color: #fff;
  font-size: 1rem;
  cursor: pointer;
}
```

- [ ] **Step 4: Verify the shell renders**

Run: `python3 -m http.server -d site 8123 &` then open `http://localhost:8123` in a browser.
Expected: title, input, button, and three empty tables render; no 404 for `style.css` or `lib/confetti.js` in the server log (a 404 for `app.js` is expected — it's Task 5). Stop the server afterwards (`kill %1`).

- [ ] **Step 5: Commit**

```bash
git add site/index.html site/style.css site/lib/confetti.js
git commit -m "feat: static site shell with styles and vendored confetti"
```

---

### Task 5: Game logic (app.js)

**Files:**
- Create: `site/app.js`

**Interfaces:**
- Consumes: DOM ids/classes from Task 4; `site/vocab.json` + `site/data/<date>.json` from Task 3; global `confetti()`.
- Produces: the complete game. No exports.

- [ ] **Step 1: Write app.js**

`site/app.js`:

```javascript
"use strict";

const MEASURES = [
  { key: "cooc", tableId: "table-cooc" },
  { key: "chem", tableId: "table-chem" },
  { key: "core", tableId: "table-core" },
];
const BAND_NAMES = ["cold", "tepid", "warm", "hot"];
const MAX_SUGGESTIONS = 8;

let vocab = [];          // canonical names, vocab-index order (underscored)
let displayNames = [];   // underscores -> spaces
let puzzle = null;       // today's data file content
let guesses = [];        // vocab indices in guess order
let won = false;
let suggestionIndices = []; // vocab indices currently shown in the dropdown
let activeSuggestion = -1;  // position within suggestionIndices
let noticeTimer = null;

const input = () => document.getElementById("guess-input");
const storageKey = () => `epicuriosimi:${puzzle.date}`;

function todayStr() {
  const now = new Date();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${m}-${d}`;
}

function loadState() {
  try {
    const raw = localStorage.getItem(storageKey());
    if (!raw) return;
    const s = JSON.parse(raw);
    guesses = (s.guesses || []).filter(
      (i) => Number.isInteger(i) && i >= 0 && i < vocab.length
    );
    won = Boolean(s.won);
  } catch (e) {
    /* storage unavailable: play without persistence */
  }
}

function saveState() {
  try {
    localStorage.setItem(storageKey(), JSON.stringify({ guesses, won }));
  } catch (e) {
    /* ignore */
  }
}

function showNotice(text) {
  const el = document.getElementById("notice");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.hidden = true; }, 2500);
}

function updateSuggestions() {
  const q = input().value.trim().toLowerCase();
  suggestionIndices = [];
  if (q) {
    // prefix matches first, then substring matches
    for (let i = 0; i < displayNames.length && suggestionIndices.length < MAX_SUGGESTIONS; i++) {
      if (displayNames[i].startsWith(q)) suggestionIndices.push(i);
    }
    for (let i = 0; i < displayNames.length && suggestionIndices.length < MAX_SUGGESTIONS; i++) {
      if (!displayNames[i].startsWith(q) && displayNames[i].includes(q)) {
        suggestionIndices.push(i);
      }
    }
  }
  activeSuggestion = suggestionIndices.length ? 0 : -1;
  renderSuggestions();
}

function renderSuggestions() {
  const listEl = document.getElementById("suggestions");
  listEl.innerHTML = "";
  listEl.hidden = suggestionIndices.length === 0;
  suggestionIndices.forEach((vocabIdx, pos) => {
    const li = document.createElement("li");
    li.textContent = displayNames[vocabIdx];
    if (pos === activeSuggestion) li.classList.add("active");
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep input focus
      submitGuess(vocabIdx);
    });
    listEl.appendChild(li);
  });
}

function clearSuggestions() {
  suggestionIndices = [];
  activeSuggestion = -1;
  renderSuggestions();
}

function onKeyDown(e) {
  if (suggestionIndices.length === 0) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeSuggestion = (activeSuggestion + 1) % suggestionIndices.length;
    renderSuggestions();
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeSuggestion =
      (activeSuggestion - 1 + suggestionIndices.length) % suggestionIndices.length;
    renderSuggestions();
  } else if (e.key === "Escape") {
    clearSuggestions();
  }
}

function onSubmit(e) {
  e.preventDefault();
  if (won) return;
  let idx = -1;
  if (activeSuggestion >= 0) {
    idx = suggestionIndices[activeSuggestion];
  } else {
    idx = displayNames.indexOf(input().value.trim().toLowerCase());
  }
  if (idx < 0) {
    showNotice("pick an ingredient from the list");
    return;
  }
  submitGuess(idx);
}

function submitGuess(idx) {
  if (guesses.includes(idx)) {
    showNotice(`already guessed ${displayNames[idx]}`);
    input().value = "";
    clearSuggestions();
    return;
  }
  guesses.push(idx);
  if (idx === puzzle.target) won = true;
  saveState();
  input().value = "";
  clearSuggestions();
  renderTables();
  if (won) finishGame(true);
}

function renderTables() {
  const last = guesses[guesses.length - 1];
  for (const { key, tableId } of MEASURES) {
    const tbody = document.querySelector(`#${tableId} tbody`);
    tbody.innerHTML = "";
    const sorted = [...guesses].sort(
      (a, b) => puzzle.scores[key][b] - puzzle.scores[key][a]
    );
    for (const idx of sorted) {
      const tr = document.createElement("tr");
      if (idx === last) tr.classList.add("latest");
      const nameTd = document.createElement("td");
      nameTd.textContent = displayNames[idx];
      const scoreTd = document.createElement("td");
      scoreTd.className = "num";
      scoreTd.textContent = (puzzle.scores[key][idx] * 100).toFixed(2);
      const bandTd = document.createElement("td");
      const band = BAND_NAMES[puzzle.bands[key][idx]];
      const badge = document.createElement("span");
      badge.className = `band band-${band}`;
      badge.textContent = band;
      bandTd.appendChild(badge);
      tr.append(nameTd, scoreTd, bandTd);
      tbody.appendChild(tr);
    }
  }
}

function finishGame(celebrate) {
  input().disabled = true;
  document.querySelector("#guess-form button").disabled = true;
  clearSuggestions();
  if (!celebrate) return;
  confetti({ particleCount: 160, spread: 80, origin: { y: 0.6 } });
  setTimeout(() => confetti({ particleCount: 80, spread: 120, origin: { y: 0.4 } }), 300);
  const n = guesses.length;
  document.getElementById("win-text").textContent =
    `${displayNames[puzzle.target]} in ${n} ${n === 1 ? "guess" : "guesses"}!`;
  document.getElementById("win-modal").hidden = false;
}

async function init() {
  const dateStr = todayStr();
  try {
    const [vres, dres] = await Promise.all([
      fetch("vocab.json"),
      fetch(`data/${dateStr}.json`),
    ]);
    if (!vres.ok || !dres.ok) throw new Error("missing puzzle files");
    vocab = await vres.json();
    puzzle = await dres.json();
  } catch (e) {
    const msg = document.getElementById("message");
    msg.textContent =
      `no puzzle for today (${dateStr}) — run: python precompute/generate_daily.py`;
    msg.hidden = false;
    document.getElementById("guess-form").hidden = true;
    return;
  }
  displayNames = vocab.map((w) => w.replace(/_/g, " "));
  document.getElementById("puzzle-number").textContent = `— puzzle #${puzzle.puzzle_number}`;
  loadState();
  renderTables();
  if (won) finishGame(false);

  input().addEventListener("input", updateSuggestions);
  input().addEventListener("keydown", onKeyDown);
  input().addEventListener("blur", () => setTimeout(clearSuggestions, 150));
  document.getElementById("guess-form").addEventListener("submit", onSubmit);
  document.getElementById("win-close").addEventListener("click", () => {
    document.getElementById("win-modal").hidden = true;
  });
}

init();
```

- [ ] **Step 2: Generate today's puzzle and serve**

```bash
precompute/.venv/bin/python precompute/generate_daily.py --reveal
python3 -m http.server -d site 8123 &
```
Note the revealed target — needed for the win check below.

- [ ] **Step 3: Manual verification checklist**

Open `http://localhost:8123` and verify each of these:

1. Header shows "epicuriosimi" and today's puzzle number.
2. Typing `chick` shows a dropdown of matching ingredients with spaces (e.g. "chicken", "chicken broth"); prefix matches come before substring matches.
3. Arrow keys move the highlighted suggestion; Escape closes the dropdown; clicking a suggestion submits it.
4. Enter submits the highlighted suggestion; the guess appears in **all three** tables with a 0–100-style score (2 decimals) and a colored band badge (cold=blue, tepid=teal, warm=orange, hot=red).
5. Guess several ingredients: each table sorts independently, descending by its own score; the latest guess row is highlighted in all three.
6. Re-guessing the same ingredient shows "already guessed …" and adds no row.
7. Submitting gibberish text shows "pick an ingredient from the list".
8. Reload the page: guesses persist (localStorage).
9. Guess the revealed target: confetti fires and the popup shows "<target> in N guesses!"; input and button disable.
10. Reload after winning: tables and disabled input persist, no confetti replay.
11. Rename `site/data/<today>.json` temporarily: page shows the "no puzzle for today" message and hides the form. Rename it back.
12. Narrow the window below 800px: tables stack vertically.

Fix anything that fails before proceeding. Stop the server (`kill %1`).

- [ ] **Step 4: Commit**

```bash
git add site/app.js
git commit -m "feat: game logic - autocomplete, score tables, persistence, win flow"
```

---

### Task 6: README + full pass

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md**

````markdown
# epicuriosimi

A daily ingredient-guessing game. Guess the secret ingredient; every guess shows
three scores against the target, powered by the
[epicure](https://huggingface.co/Kaikaku/epicure-core) ingredient embeddings:

- **pairs well with** (epicure-cooc) — co-occurrence in recipes
- **shares flavor profile** (epicure-chem) — shared flavor compounds
- **similar to** (epicure-core) — overall similarity

Each score gets a temperature — cold → tepid → warm → hot — based on how close
the guess ranks to the target's nearest neighbors.

## Setup (once)

```bash
python3 -m venv precompute/.venv
precompute/.venv/bin/pip install -r precompute/requirements.txt
```

## Play

```bash
# generate today's puzzle (downloads ~13 MB of models on first run)
precompute/.venv/bin/python precompute/generate_daily.py

# serve the site
python3 -m http.server -d site 8123
```

Open http://localhost:8123 and guess.

Useful flags: `--date YYYY-MM-DD` (specific day), `--days N` (generate N days
ahead), `--reveal` (print the answer — spoiler!).

## Tests

```bash
precompute/.venv/bin/pytest precompute/tests -v
```
````

- [ ] **Step 2: Full verification pass**

```bash
precompute/.venv/bin/pytest precompute/tests -v
precompute/.venv/bin/python precompute/generate_daily.py --date 2026-09-02
```
Expected: all tests pass; generator reruns cleanly from cache (no re-download, overwrites the existing file without error).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, play, and test instructions"
```
