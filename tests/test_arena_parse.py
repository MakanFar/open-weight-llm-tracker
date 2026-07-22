import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena

FIXTURE = Path(__file__).parent / "fixtures" / "arena_sample.html"


@pytest.fixture
def rows():
    return pull_arena.parse_leaderboard(FIXTURE.read_text())


def test_parses_all_rows(rows):
    assert len(rows) == 4


def test_ranks_in_order(rows):
    assert [r["rank"] for r in rows] == [1, 10, 18, 37]


def test_extracts_score_and_ci(rows):
    glm = next(r for r in rows if r["rank"] == 10)
    assert glm["net_improvement_pct"] == 6.50
    assert glm["net_improvement_ci"] == 1.20


def test_maps_org_from_keyword(rows):
    glm = next(r for r in rows if r["rank"] == 10)
    assert glm["org"] == "Zhipu AI"
    assert glm["matched_keyword"] == "glm"


def test_does_not_set_open_weight(rows):
    """open_weight is decided by repo resolution (Task 4), never by parsing."""
    assert all("open_weight" not in r for r in rows)


def test_keeps_raw_cells(rows):
    first = rows[0]
    assert first["raw"][0] == "1"
    assert "Claude Fable 5" in first["raw"][1]


def test_by_header_populated(rows):
    first = rows[0]
    assert first["by_header"]["Sessions"] == "23,549"


def test_empty_html_returns_no_rows():
    assert pull_arena.parse_leaderboard("<html><body></body></html>") == []
