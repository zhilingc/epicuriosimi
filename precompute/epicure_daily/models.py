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
