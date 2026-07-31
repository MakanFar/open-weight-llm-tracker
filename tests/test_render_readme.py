import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import render_readme as rr


def _model(**kw):
    base = dict(name="M", developer="Org", release_date=date(2025, 1, 1),
                params_total_b=7, params_active_b=7, architecture="dense",
                context_window=4096, modality="text", license="mit",
                commercial_use=True, hf_repo="org/m",
                benchmark={"name": "MMLU", "score": 70.0, "source": "vendor"})
    base.update(kw)
    return base


def _lb(metric, score):
    return {"org/m": {"metric": metric, "score": score}}


def test_mmlu_cell_prefers_leaderboard_plain():
    m = _model()
    assert rr.mmlu_cell(m, _lb("MMLU", 78.6)) == "78.6"


def test_mmlu_cell_falls_back_to_manual_marked():
    m = _model()
    assert rr.mmlu_cell(m, {}) == "70.0*"


def test_mmlu_cell_ignores_an_mmlu_pro_score():
    """MMLU-PRO is a different scale — it must not be shown as an MMLU figure."""
    m = _model()
    assert rr.mmlu_cell(m, _lb("MMLU-PRO", 41.2)) == "70.0*"


def test_mmlu_pro_cell_shows_only_mmlu_pro_scores():
    m = _model()
    assert rr.mmlu_pro_cell(m, _lb("MMLU-PRO", 41.2)) == "41.2"
    assert rr.mmlu_pro_cell(m, _lb("MMLU", 78.6)) == "—"
    assert rr.mmlu_pro_cell(m, {}) == "—"


def test_mmlu_pro_cell_never_falls_back_to_the_manual_figure():
    """models.yaml benchmark.score is MMLU; it belongs in the MMLU column."""
    m = _model(benchmark={"name": "MMLU", "score": 88.6, "source": "vendor"})
    assert rr.mmlu_pro_cell(m, {}) == "—"


def test_table_has_distinct_mmlu_and_mmlu_pro_columns():
    table = rr.build_table([_model()], _lb("MMLU-PRO", 41.2), {})
    head = table.splitlines()[0]
    assert "| MMLU |" in head
    assert "| MMLU-Pro |" in head
    body = table.splitlines()[2]
    assert "41.2" in body and "70.0*" in body


def test_arena_cell_shows_rank_or_dash():
    m = _model(hf_repo="org/m")
    assert rr.arena_cell(m, {"org/m": 5}) == "5"
    assert rr.arena_cell(m, {}) == "—"


def test_load_leaderboard_tolerates_missing_file(tmp_path):
    assert rr.load_leaderboard(tmp_path / "nope.yaml") == {}


def test_load_leaderboard_parses_metric_and_score(tmp_path):
    f = tmp_path / "lb.yaml"
    f.write_text("scores:\n  Org/M:\n    metric: MMLU-PRO\n    score: 41.2\n")
    assert rr.load_leaderboard(f) == {"org/m": {"metric": "MMLU-PRO",
                                                "score": 41.2}}


def test_load_leaderboard_skips_entries_missing_a_metric(tmp_path):
    """A score with no metric name cannot be placed in either column."""
    f = tmp_path / "lb.yaml"
    f.write_text("scores:\n  Org/M:\n    score: 41.2\n  Org/N:\n    metric: MMLU\n")
    assert rr.load_leaderboard(f) == {}


def test_load_arena_ranks_parses_resolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text("arena_agent:\n- resolved_repo: Org/M\n  rank: 3\n- resolved_repo: null\n  rank: 4\n")
    assert rr.load_arena_ranks(f) == {"org/m": 3}
