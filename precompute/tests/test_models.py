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
