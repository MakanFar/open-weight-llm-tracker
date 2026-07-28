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


def test_mmlu_cell_prefers_leaderboard_plain():
    m = _model()
    assert rr.mmlu_cell(m, {"org/m": 78.6}) == "78.6"


def test_mmlu_cell_falls_back_to_manual_marked():
    m = _model()
    assert rr.mmlu_cell(m, {}) == "70.0*"


def test_arena_cell_shows_rank_or_dash():
    m = _model(hf_repo="org/m")
    assert rr.arena_cell(m, {"org/m": 5}) == "5"
    assert rr.arena_cell(m, {}) == "—"


def test_load_leaderboard_tolerates_missing_file(tmp_path):
    assert rr.load_leaderboard(tmp_path / "nope.yaml") == {}


def test_load_leaderboard_parses_scores(tmp_path):
    f = tmp_path / "lb.yaml"
    f.write_text("scores:\n  Org/M:\n    mmlu: 78.6\n")
    assert rr.load_leaderboard(f) == {"org/m": 78.6}


def test_load_arena_ranks_parses_resolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text("arena_agent:\n- resolved_repo: Org/M\n  rank: 3\n- resolved_repo: null\n  rank: 4\n")
    assert rr.load_arena_ranks(f) == {"org/m": 3}
