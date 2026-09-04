"""Daily target selection, scoring, and temperature bands."""
import hashlib
import random
from datetime import date
from pathlib import Path

import numpy as np

LAUNCH_DATE = date(2026, 9, 2)

# Band codes: 0=cold, 1=tepid, 2=warm, 3=hot. Rank thresholds per spec.
_BAND_THRESHOLDS = ((10, 3), (100, 2), (500, 1))

# "similar to" = name semantics (sentence embedding) blended with epicure-core,
# both in percentile-rank space so neither model's cosine scale dominates.
NAME_WEIGHT = 0.7


def load_targets(path: Path) -> list[str]:
    lines = (ln.strip() for ln in path.read_text().splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def target_index(puzzle_no: int, bag: list[str], words: list[str]) -> int:
    """Draw from the bag without replacement; each full cycle gets its own shuffle."""
    for w in bag:
        if w not in words:
            raise ValueError(f"target {w!r} not in vocab")
    cycle, pos = divmod(puzzle_no - 1, len(bag))
    seed = int(hashlib.sha256(f"epicuriosimi:cycle:{cycle}".encode()).hexdigest(), 16)
    order = list(bag)
    random.Random(seed).shuffle(order)
    return words.index(order[pos])


def puzzle_number(d: date) -> int:
    return (d - LAUNCH_DATE).days + 1


def cosine_scores(embeddings: np.ndarray, target_idx: int) -> np.ndarray:
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return normed @ normed[target_idx]


def percentile_scores(scores: np.ndarray, target_idx: int) -> np.ndarray:
    """Map scores to closeness in [0, 1]: nearest non-target = 1, farthest = 0, target = 1."""
    others = np.delete(np.arange(len(scores)), target_idx)
    order = others[np.argsort(-scores[others], kind="stable")]
    pct = np.empty(len(scores))
    pct[order] = 1.0 - np.arange(len(others)) / max(len(others) - 1, 1)
    pct[target_idx] = 1.0
    return pct


def blended_scores(a: np.ndarray, b: np.ndarray, target_idx: int, weight_a: float) -> np.ndarray:
    return weight_a * percentile_scores(a, target_idx) + (1 - weight_a) * percentile_scores(b, target_idx)


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


def build_puzzle(d: date, embeddings: dict[str, np.ndarray], name_embeddings: np.ndarray,
                 bag: list[str], words: list[str]) -> dict:
    date_str = d.isoformat()
    n = puzzle_number(d)
    t = target_index(n, bag, words)
    scores = {}
    bands = {}
    for key, emb in embeddings.items():
        s = cosine_scores(emb, t)
        if key == "core":
            s = blended_scores(cosine_scores(name_embeddings, t), s, t, NAME_WEIGHT)
        scores[key] = [round(float(x), 4) for x in s]
        bands[key] = band_codes(s, t).tolist()
    return {
        "date": date_str,
        "puzzle_number": n,
        "target": t,
        "scores": scores,
        "bands": bands,
    }
