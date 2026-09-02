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
