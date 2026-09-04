"""Regenerate name_embeddings.npy — run once locally when the vocab changes.

Encodes every ingredient name (underscores -> spaces) with a sentence-transformer
and stores L2-normalised float16 vectors in vocab-index order. The daily generator
only loads the .npy, so CI never needs torch.

Usage: precompute/.venv/bin/python precompute/embed_names.py
Requires: pip install sentence-transformers
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from epicure_daily.models import download_models, load_all

MODEL = "sentence-transformers/all-mpnet-base-v2"
HERE = Path(__file__).resolve().parent


def main() -> None:
    words, _ = load_all(download_models(HERE / ".model_cache"))
    names = [w.replace("_", " ") for w in words]
    vectors = SentenceTransformer(MODEL).encode(
        names, normalize_embeddings=True, batch_size=128, show_progress_bar=False
    )
    out = HERE / "name_embeddings.npy"
    np.save(out, vectors.astype(np.float16))
    print(f"wrote {out} {vectors.shape}")


if __name__ == "__main__":
    main()
