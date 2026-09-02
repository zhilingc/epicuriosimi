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
# generate today's puzzle (downloads ~6 MB of models on first run)
precompute/.venv/bin/python precompute/generate_daily.py

# serve the site
python3 -m http.server -d site 8123
```

Open http://localhost:8123 and guess.

Useful flags: `--date YYYY-MM-DD` (specific day), `--days N` (generate N consecutive days starting at `--date`), `--reveal` (print the answer — spoiler!).

## Tests

```bash
precompute/.venv/bin/pytest precompute/tests -v
```
