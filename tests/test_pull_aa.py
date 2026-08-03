import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_aa

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "aa_leaderboard.html").read_text()


def test_parse_reads_every_scored_row():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert [r["model"] for r in rows] == [
        "Claude Opus 5 (max)", "Kimi K3 (max)", "Kimi K3 (low)",
        "GLM-5.2", "Llama 3.3 70B",
    ]


def test_parse_reads_creator_and_index():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    kimi = next(r for r in rows if r["model"] == "Kimi K3 (max)")
    assert kimi["creator"] == "Moonshot AI"
    assert kimi["intelligence_index"] == 57


def test_parse_drops_rows_with_no_numeric_index():
    """Header rows and unrated models both fail the integer check."""
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert all(r["model"] != "Unrated Model" for r in rows)
    assert all(r["model"] != "Model" for r in rows)


def test_parse_returns_empty_for_markup_with_no_table():
    assert pull_aa.parse_leaderboard("<html><body><p>nope</p></body></html>") == []


def test_best_by_slug_keeps_the_highest_variant():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    best = pull_aa.best_by_slug(rows)
    kimi = best["kimik3"]
    assert kimi["intelligence_index"] == 57
    assert kimi["variant"] == "max"
    assert kimi["aa_model"] == "Kimi K3 (max)"


def test_best_by_slug_labels_a_row_with_no_parenthetical():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert best["glm52"]["variant"] == "default"
    assert best["glm52"]["intelligence_index"] == 34


def test_best_by_slug_collapses_variants_to_one_entry_per_model():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert sorted(best) == ["claudeopus5", "glm52", "kimik3", "llama3370b"]


def test_best_by_slug_is_order_independent():
    """The winner must not depend on which variant the page lists first."""
    rows = [
        {"model": "M (low)", "creator": "C", "intelligence_index": 10},
        {"model": "M (max)", "creator": "C", "intelligence_index": 20},
    ]
    assert pull_aa.best_by_slug(rows)["m"]["intelligence_index"] == 20
    assert pull_aa.best_by_slug(rows[::-1])["m"]["intelligence_index"] == 20


def test_best_by_slug_keeps_different_sizes_apart():
    """Size is identity: 72B and 7B are different models, not variants.

    pull_arena's _strip_decorations drops size tokens, which is right for its
    pairwise ratio matching and fatally wrong here — both would key to 'qwen25'
    and one would silently overwrite the other.
    """
    rows = [
        {"model": "Qwen2.5 72B", "creator": "Alibaba", "intelligence_index": 30},
        {"model": "Qwen2.5 7B", "creator": "Alibaba", "intelligence_index": 12},
    ]
    best = pull_aa.best_by_slug(rows)
    assert sorted(best) == ["qwen2572b", "qwen257b"]
    assert best["qwen2572b"]["intelligence_index"] == 30
    assert best["qwen257b"]["intelligence_index"] == 12
