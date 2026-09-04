"""Generate epicuriosimi daily puzzle JSON files.

Usage: python precompute/generate_daily.py [--date YYYY-MM-DD] [--days N] [--reveal]
"""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from epicure_daily.models import download_models, load_all
from epicure_daily.puzzle import build_puzzle, load_targets

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT
PRECOMPUTE_DIR = REPO_ROOT / "precompute"
CACHE_DIR = PRECOMPUTE_DIR / ".model_cache"
TARGETS_FILE = PRECOMPUTE_DIR / "targets.txt"
NAME_EMBEDDINGS_FILE = PRECOMPUTE_DIR / "name_embeddings.npy"


def write_puzzles(start: date, days: int, words: list[str],
                  embeddings: dict[str, np.ndarray], name_embeddings: np.ndarray,
                  bag: list[str], site_dir: Path, reveal: bool = False) -> list[Path]:
    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "vocab.json").write_text(json.dumps(words))
    written = []
    for offset in range(days):
        d = start + timedelta(days=offset)
        puzzle = build_puzzle(d, embeddings, name_embeddings, bag, words)
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

    words, embeddings = load_all(download_models(CACHE_DIR))
    name_embeddings = np.load(NAME_EMBEDDINGS_FILE).astype(np.float32)
    if name_embeddings.shape[0] != len(words):
        raise SystemExit(
            f"{NAME_EMBEDDINGS_FILE} has {name_embeddings.shape[0]} rows but vocab has "
            f"{len(words)} — rerun precompute/embed_names.py"
        )
    write_puzzles(date.fromisoformat(args.date), args.days, words, embeddings,
                  name_embeddings, load_targets(TARGETS_FILE), SITE_DIR, reveal=args.reveal)


if __name__ == "__main__":
    main()
