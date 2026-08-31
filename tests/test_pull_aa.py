"""pull_aa.py is a scraper and nothing else.

The name->repo join it used to perform on the way to disk now lives in
aa_join.py and runs at the point of use; those tests moved to
tests/test_aa_join.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import yaml
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

    names.strip_decorations drops size tokens, which is right for pull_arena's
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


def test_refresh_writes_every_scraped_row(tmp_path):
    """Including the proprietary ones. The scrape does not decide relevance.

    Claude Opus 5 will never be a tracked model, but discarding it here is
    what made the file unrepairable: a score dropped at write time can only
    come back by fetching AA again.
    """
    out = tmp_path / "aa_scores.yaml"
    n = pull_aa.refresh(out, FIXTURE)
    assert n == 4

    scores = yaml.safe_load(out.read_text())["scores"]
    assert sorted(scores) == ["claudeopus5", "glm52", "kimik3", "llama3370b"]
    assert scores["kimik3"]["intelligence_index"] == 57
    assert scores["claudeopus5"]["intelligence_index"] == 61


def test_refresh_records_the_source_once_for_the_whole_file(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    pull_aa.refresh(out, FIXTURE)
    doc = yaml.safe_load(out.read_text())
    assert doc["source"] == pull_aa.LEADERBOARD_URL
    assert "source" not in doc["scores"]["kimik3"]


def test_refresh_does_not_repeat_the_slug_inside_the_entry(tmp_path):
    """model_slug is the key; storing it twice invites the two to disagree."""
    out = tmp_path / "aa_scores.yaml"
    pull_aa.refresh(out, FIXTURE)
    entry = yaml.safe_load(out.read_text())["scores"]["kimik3"]
    assert sorted(entry) == ["aa_model", "intelligence_index", "variant"]


def test_refresh_leaves_the_sidecar_untouched_when_the_fetch_failed(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, None) is None
    assert out.read_text() == before


def test_refresh_treats_a_zero_row_parse_as_failure(tmp_path):
    """Empty parse means AA's markup changed — do not erase good data."""
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, "<html><body>redesigned</body></html>") is None
    assert out.read_text() == before


def test_refresh_reads_no_model_files(tmp_path, monkeypatch):
    """The scraper must not depend on the index it is scored against.

    That dependency is what made the write a snapshot of a join and put
    pull_aa and discover in a cycle neither could go first in.
    """
    def fail(*a, **kw):
        raise AssertionError("pull_aa opened a file it has no business reading")

    monkeypatch.setattr(pull_aa.yaml, "safe_load", fail)
    assert pull_aa.refresh(tmp_path / "aa_scores.yaml", FIXTURE) == 4


def test_fetch_html_returns_none_on_error():
    def boom(url, **kw):
        raise RuntimeError("HTTP Error 429: Too Many Requests")
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=boom) is None


def test_fetch_html_returns_none_on_bad_status():
    class Resp:
        status_code = 503
        text = "nope"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) is None


def test_fetch_html_returns_the_body_on_success():
    class Resp:
        status_code = 200
        text = "<html>ok</html>"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) == "<html>ok</html>"
