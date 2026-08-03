import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import render_readme as rr


def _model(**kw):
    base = dict(name="M", developer="Org", release_date=date(2025, 1, 1),
                params_total_b=7, params_active_b=7, architecture="dense",
                context_window=4096, modality="text", license="mit",
                commercial_use=True, hf_repo="org/m")
    base.update(kw)
    return base


def test_arena_cell_matches_by_resolved_repo():
    m = _model(hf_repo="zai-org/GLM-5.2", name="GLM 5.2")
    ranks = {"repos": {"zai-org/glm-5.2": 12}, "names": {}}
    assert rr.arena_cell(m, ranks) == "12"


def test_arena_cell_falls_back_to_name_when_the_repo_never_resolved():
    """A rank we already hold must not be hidden by a failed HF lookup."""
    m = _model(hf_repo="moonshotai/Kimi-K3", name="Kimi K3")
    ranks = {"repos": {}, "names": {"kimik3": 5}}
    assert rr.arena_cell(m, ranks) == "5"


def test_arena_cell_dash_when_the_model_is_not_ranked():
    m = _model(hf_repo="org/m", name="M")
    assert rr.arena_cell(m, {"repos": {}, "names": {}}) == "—"


def test_load_arena_ranks_indexes_resolved_and_unresolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(
        "arena_agent:\n"
        "- resolved_repo: Org/M\n  rank: 3\n  model: M Thing\n"
        "- resolved_repo: null\n  rank: 5\n  model: Kimi K3 Moonshot\n")
    ranks = rr.load_arena_ranks(f)
    assert ranks["repos"] == {"org/m": 3}
    assert ranks["names"]["kimik3"] == 5


def test_aa_cell_shows_the_index():
    m = _model(hf_repo="moonshotai/Kimi-K3")
    aa = {"moonshotai/kimi-k3": {"index": 57, "variant": "max"}}
    assert rr.aa_cell(m, aa) == "57"


def test_aa_cell_dashes_when_aa_does_not_rate_the_model():
    """There is no fallback: models.yaml no longer carries a score."""
    assert rr.aa_cell(_model(hf_repo="org/m"), {}) == "—"


def test_load_aa_scores_parses_the_sidecar(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  Moonshot/Kimi-K3:\n    intelligence_index: 57\n"
                 "    variant: max\n")
    assert rr.load_aa_scores(f) == {
        "moonshot/kimi-k3": {"index": 57, "variant": "max"}}


def test_load_aa_scores_tolerates_a_missing_file(tmp_path):
    assert rr.load_aa_scores(tmp_path / "nope.yaml") == {}


def test_load_aa_scores_skips_entries_with_no_numeric_index(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  org/m:\n    variant: max\n")
    assert rr.load_aa_scores(f) == {}


def test_table_has_an_aa_index_column_and_no_mmlu():
    table = rr.build_table([_model(hf_repo="org/m")],
                           {"org/m": {"index": 42, "variant": "max"}},
                           {"repos": {}, "names": {}})
    head = table.splitlines()[0]
    assert "| AA Index |" in head
    assert "MMLU" not in head
    assert "42" in table.splitlines()[2]
