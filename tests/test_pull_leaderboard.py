"""The sidecar records WHICH metric it captured, because v2 dropped MMLU.

The live Open LLM Leaderboard publishes 'MMLU-PRO' and 'MMLU-PRO Raw' — there
is no plain 'MMLU' column on any of its 4576 rows. Matching only the exact key
"mmlu" therefore scored nothing and wrote an empty sidecar. MMLU-PRO is a
different, much harsher scale (~40 where MMLU reads ~80), so the metric name
travels with the score rather than being assumed by the reader.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pull_leaderboard as pl


def test_extract_score_prefers_plain_mmlu():
    """Archived v1 rows still carry it; it stays the better metric when present."""
    assert pl.extract_score({"MMLU": 78.6}) == ("MMLU", 78.6)
    assert pl.extract_score({"mmlu": 78.6}) == ("MMLU", 78.6)
    assert pl.extract_score({"MMLU-PRO": 50.0, "MMLU": 78.6}) == ("MMLU", 78.6)


def test_extract_score_falls_back_to_mmlu_pro():
    assert pl.extract_score({"MMLU-PRO": 41.2}) == ("MMLU-PRO", 41.2)
    assert pl.extract_score({"mmlu-pro": 41.2}) == ("MMLU-PRO", 41.2)


def test_extract_score_ignores_the_raw_variant():
    """'MMLU-PRO Raw' is the 0-1 fraction, not the published percentage."""
    assert pl.extract_score({"MMLU-PRO Raw": 0.412, "MMLU-PRO": 41.2}) == \
        ("MMLU-PRO", 41.2)
    assert pl.extract_score({"MMLU-PRO Raw": 0.412}) is None


def test_extract_score_returns_none_without_a_known_metric():
    assert pl.extract_score({"other": 1}) is None
    assert pl.extract_score({"MMLU": "n/a"}) is None


def test_build_scores_records_the_metric_alongside_the_score():
    rows = [{"fullname": "meta-llama/Llama-3.3-70B-Instruct", "MMLU-PRO": 41.2}]
    scores = pl.build_scores(["meta-llama/Llama-3.3-70B-Instruct"], rows)
    assert scores == {"meta-llama/Llama-3.3-70B-Instruct": {
        "metric": "MMLU-PRO", "score": 41.2, "source": pl.LEADERBOARD_URL}}


def test_build_scores_rounds_to_one_decimal():
    """The dataset carries full float precision; a table cell does not need it."""
    rows = [{"fullname": "org/m", "MMLU-PRO": 38.0079048463357}]
    assert pl.build_scores(["org/m"], rows)["org/m"]["score"] == 38.0


def test_build_scores_maps_repos_case_insensitively():
    rows = [
        {"fullname": "meta-llama/Llama-3.1-405B-Instruct", "MMLU": 88.6},
        {"fullname": "some/UnlistedModel", "MMLU": 10.0},
    ]
    repos = ["meta-llama/Llama-3.1-405B-Instruct", "google/gemma-2-27b-it"]
    scores = pl.build_scores(repos, rows)
    assert scores == {"meta-llama/Llama-3.1-405B-Instruct": {
        "metric": "MMLU", "score": 88.6, "source": pl.LEADERBOARD_URL}}


def test_build_scores_skips_rows_with_no_usable_metric():
    rows = [{"fullname": "org/m", "MMLU-PRO Raw": 0.4}]
    assert pl.build_scores(["org/m"], rows) == {}


def test_fetch_rows_skips_non_dict_items():
    page = {"rows": [{"row": {"fullname": "org/m", "MMLU": 70.0}}, "garbage", None]}
    got = pl.fetch_rows(get_json=lambda url: page)
    assert {"fullname": "org/m", "MMLU": 70.0} in got
    # no exception, bad items skipped
    assert all(isinstance(r, dict) for r in got)


def test_fetch_rows_reports_failure_rather_than_an_empty_result():
    """A 429 must be distinguishable from 'the leaderboard has no rows'."""
    def boom(url):
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    assert pl.fetch_rows(get_json=boom) is None


def test_fetch_rows_reports_failure_when_a_later_page_dies():
    """A partial fetch would silently drop every score past the failing page."""
    pages = []

    def flaky(url):
        pages.append(url)
        if len(pages) == 1:
            return {"rows": [{"row": {"fullname": f"org/m{i}", "MMLU": 70.0}}
                             for i in range(100)]}
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    assert pl.fetch_rows(get_json=flaky) is None


def test_fetch_rows_returns_empty_list_for_a_genuinely_empty_dataset():
    assert pl.fetch_rows(get_json=lambda url: {"rows": []}) == []


def test_refresh_scores_leaves_the_sidecar_untouched_on_fetch_failure(tmp_path):
    """The committed sidecar is data — a failed run must not erase it."""
    out = tmp_path / "leaderboard_scores.yaml"
    out.write_text("scores:\n  org/m:\n    metric: MMLU-PRO\n    score: 41.2\n")
    before = out.read_text()

    assert pl.refresh_scores(out, ["org/m"], None) is None
    assert out.read_text() == before


def test_refresh_scores_writes_when_the_fetch_succeeded(tmp_path):
    out = tmp_path / "leaderboard_scores.yaml"
    rows = [{"fullname": "org/m", "MMLU-PRO": 41.2}]

    assert pl.refresh_scores(out, ["org/m"], rows) == 1
    assert yaml.safe_load(out.read_text())["scores"]["org/m"]["score"] == 41.2


def test_refresh_scores_can_write_a_legitimately_empty_result(tmp_path):
    """--no-fetch passes [] on purpose; that is not a failure."""
    out = tmp_path / "leaderboard_scores.yaml"
    assert pl.refresh_scores(out, ["org/m"], []) == 0
    assert yaml.safe_load(out.read_text())["scores"] == {}
