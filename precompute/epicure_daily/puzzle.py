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
