import json
from datetime import date

import numpy as np

import generate_daily
from epicure_daily.puzzle import load_targets

WORDS = [f"ing_{i:02d}" for i in range(20)]
BAG = ["ing_02", "ing_05", "ing_09"]


def make_embeddings(n=20, dim=4):
    rng = np.random.default_rng(7)
    return {k: rng.normal(size=(n, dim)).astype(np.float32) for k in ["cooc", "chem", "core"]}


def make_names(n=20, dim=6):
    return np.random.default_rng(3).normal(size=(n, dim)).astype(np.float32)


def test_write_puzzles_creates_data_and_vocab(tmp_path):
    site = tmp_path / "site"
    written = generate_daily.write_puzzles(date(2026, 9, 2), 3, WORDS, make_embeddings(),
                                           make_names(), BAG, site)
    assert [p.name for p in written] == ["2026-09-02.json", "2026-09-03.json", "2026-09-04.json"]
    assert json.loads((site / "vocab.json").read_text()) == WORDS
    first = json.loads(written[0].read_text())
    assert first["date"] == "2026-09-02"
    assert first["puzzle_number"] == 1
    assert len(first["scores"]["core"]) == 20
    assert WORDS[first["target"]] in BAG


def test_write_puzzles_reveal_prints_target(tmp_path, capsys):
    generate_daily.write_puzzles(date(2026, 9, 2), 1, WORDS, make_embeddings(), make_names(),
                                 BAG, tmp_path / "site", reveal=True)
    out = capsys.readouterr().out
    assert "target=ing_" in out


def test_shipped_targets_are_all_in_real_vocab():
    # guards the committed bag against typos; uses the vocab list shipped with the models
    bag = load_targets(generate_daily.TARGETS_FILE)
    assert len(bag) == 100
    assert len(set(bag)) == len(bag)
    vocab_file = generate_daily.SITE_DIR / "vocab.json"
    if vocab_file.exists():
        vocab = set(json.loads(vocab_file.read_text()))
        missing = [w for w in bag if w not in vocab]
        assert missing == []


def test_shipped_name_embeddings_match_vocab_size():
    emb = np.load(generate_daily.NAME_EMBEDDINGS_FILE)
    assert emb.shape[0] == 1790
    assert emb.dtype == np.float16
