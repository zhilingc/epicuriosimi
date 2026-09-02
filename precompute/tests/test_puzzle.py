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
