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
