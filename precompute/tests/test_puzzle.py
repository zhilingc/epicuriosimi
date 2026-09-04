from datetime import date

import numpy as np
import pytest

from epicure_daily import puzzle

WORDS = [f"w{i:03d}" for i in range(20)]
BAG = ["w003", "w007", "w011", "w015"]


def test_target_index_cycles_through_bag_without_repeats():
    n = len(BAG)
    first_cycle = [puzzle.target_index(p, BAG, WORDS) for p in range(1, n + 1)]
    assert sorted(first_cycle) == sorted(WORDS.index(w) for w in BAG)
    second_cycle = [puzzle.target_index(p, BAG, WORDS) for p in range(n + 1, 2 * n + 1)]
    assert sorted(second_cycle) == sorted(first_cycle)
    assert first_cycle != second_cycle or n == 1  # reshuffled per cycle (tiny bag may collide)


def test_target_index_deterministic():
    assert puzzle.target_index(5, BAG, WORDS) == puzzle.target_index(5, BAG, WORDS)


def test_target_index_rejects_unknown_bag_word():
    with pytest.raises(ValueError, match="not in vocab"):
        puzzle.target_index(1, ["nope"], WORDS)


def test_load_targets_reads_one_word_per_line(tmp_path):
    f = tmp_path / "targets.txt"
    f.write_text("garlic\n\n# comment\nonion \n")
    assert puzzle.load_targets(f) == ["garlic", "onion"]


def test_puzzle_number_is_one_based_from_launch():
    assert puzzle.puzzle_number(date(2026, 9, 2)) == 1
    assert puzzle.puzzle_number(date(2026, 9, 11)) == 10


def test_cosine_scores_normalizes_before_dot():
    emb = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    s = puzzle.cosine_scores(emb, 0)
    np.testing.assert_allclose(s, [1.0, 0.0, np.sqrt(0.5)], atol=1e-6)


def test_percentile_scores_rank_space():
    # target 0; others score 0.9, 0.1, 0.5 -> nearest gets 1.0, farthest 0.0, target 1.0
    s = np.array([1.0, 0.9, 0.1, 0.5])
    p = puzzle.percentile_scores(s, 0)
    np.testing.assert_allclose(p, [1.0, 1.0, 0.0, 0.5])


def test_blended_scores_weights_percentiles():
    a = np.array([1.0, 0.9, 0.1, 0.5])   # percentiles: 1.0, 0.0, 0.5
    b = np.array([1.0, 0.1, 0.9, 0.5])   # percentiles: 0.0, 1.0, 0.5
    blended = puzzle.blended_scores(a, b, 0, weight_a=0.7)
    np.testing.assert_allclose(blended, [1.0, 0.7, 0.3, 0.5])


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
    names = rng.normal(size=(20, 6)).astype(np.float32)
    p = puzzle.build_puzzle(date(2026, 9, 2), embeddings, names, BAG, WORDS)
    assert p["date"] == "2026-09-02"
    assert p["puzzle_number"] == 1
    t = p["target"]
    assert WORDS[t] in BAG
    assert set(p["scores"]) == set(p["bands"]) == {"cooc", "chem", "core"}
    for key in embeddings:
        assert len(p["scores"][key]) == 20
        assert len(p["bands"][key]) == 20
        assert p["scores"][key][t] == 1.0
        assert p["bands"][key][t] == 3
        assert all(b in (0, 1, 2, 3) for b in p["bands"][key])
    assert all(-1.0001 <= s <= 1.0001 for s in p["scores"]["cooc"])
    assert all(0.0 <= s <= 1.0 for s in p["scores"]["core"])  # blended percentile


def test_build_puzzle_core_is_blend_not_raw_cosine():
    rng = np.random.default_rng(1)
    embeddings = {k: rng.normal(size=(20, 4)).astype(np.float32) for k in ["cooc", "chem", "core"]}
    names = rng.normal(size=(20, 6)).astype(np.float32)
    p = puzzle.build_puzzle(date(2026, 9, 2), embeddings, names, BAG, WORDS)
    raw = puzzle.cosine_scores(embeddings["core"], p["target"])
    assert not np.allclose(p["scores"]["core"], np.round(raw, 4))
