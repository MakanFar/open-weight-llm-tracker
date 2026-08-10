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
    ranks = {"repos": {"zai-org/glm-5.2": 12}, "names": {}, "identities": {}}
    assert rr.arena_cell(m, ranks) == "12"


def test_arena_cell_falls_back_to_name_when_the_repo_never_resolved():
    """A rank we already hold must not be hidden by a failed HF lookup."""
    m = _model(hf_repo="moonshotai/Kimi-K3", name="Kimi K3")
    ranks = {"repos": {}, "names": {"kimik3": 5}, "identities": {}}
    assert rr.arena_cell(m, ranks) == "5"


def test_arena_cell_dash_when_the_model_is_not_ranked():
    m = _model(hf_repo="org/m", name="M")
    assert rr.arena_cell(m, {"repos": {}, "names": {}, "identities": {}}) == "—"


def test_load_arena_ranks_indexes_resolved_and_unresolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(
        "arena_agent:\n"
        "- resolved_repo: Org/M\n  rank: 3\n  model: M Thing\n"
        "- resolved_repo: null\n  rank: 5\n  model: Kimi K3 Moonshot\n")
    ranks = rr.load_arena_ranks(f)
    assert ranks["repos"] == {"org/m": 3}
    assert ranks["names"]["kimik3"] == 5
    assert ranks["identities"] == {"m": 3}


def test_aa_cell_shows_the_index():
    m = _model(hf_repo="moonshotai/Kimi-K3")
    aa = {"repos": {"moonshotai/kimi-k3": {"index": 57, "variant": "max"}},
          "identities": {}}
    assert rr.aa_cell(m, aa) == "57"


def test_aa_cell_dashes_when_aa_does_not_rate_the_model():
    """There is no fallback to a manual figure: models.yaml carries no score."""
    empty = {"repos": {}, "identities": {}}
    assert rr.aa_cell(_model(hf_repo="org/m"), empty) == "—"


def test_load_aa_scores_parses_the_sidecar(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  Moonshot/Kimi-K3:\n    intelligence_index: 57\n"
                 "    variant: max\n")
    loaded = rr.load_aa_scores(f)
    assert loaded["repos"] == {
        "moonshot/kimi-k3": {"index": 57, "variant": "max"}}
    assert loaded["identities"] == {
        "kimik3": {"index": 57, "variant": "max"}}


def test_load_aa_scores_tolerates_a_missing_file(tmp_path):
    assert rr.load_aa_scores(tmp_path / "nope.yaml") == {
        "repos": {}, "identities": {}}


def test_load_aa_scores_skips_entries_with_no_numeric_index(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  org/m:\n    variant: max\n")
    assert rr.load_aa_scores(f) == {"repos": {}, "identities": {}}


def test_commercial_badge_unmarked_when_verified():
    m = _model(commercial_use=True, commercial_use_verified=True)
    assert rr.commercial_badge(m) == "Yes"


def test_commercial_badge_marked_with_a_trailing_question_mark_when_unverified():
    """A row promoted automatically infers commercial_use from the licence
    tag; nobody has read the licence. Rendering it identically to a checked
    value would publish an unverified legal claim as a settled one."""
    m = _model(commercial_use=True, commercial_use_verified=False)
    assert rr.commercial_badge(m) == "Yes?"


def test_commercial_badge_marked_when_verified_field_is_absent():
    """Absent must behave like False — most rows predate this field."""
    m = _model(commercial_use="conditional")
    assert "commercial_use_verified" not in m
    assert rr.commercial_badge(m) == "Conditional?"


def test_table_has_an_aa_index_column_and_no_mmlu():
    table = rr.build_table(
        [_model(hf_repo="org/m")],
        {"repos": {"org/m": {"index": 42, "variant": "max"}}, "identities": {}},
        {"repos": {}, "names": {}, "identities": {}})
    head = table.splitlines()[0]
    assert "| AA Index |" in head
    assert "MMLU" not in head
    assert "42" in table.splitlines()[2]


# --- identity fallback -------------------------------------------------------

def test_aa_cell_falls_back_to_repo_identity():
    """The DeepSeek V4 Flash split: arena resolved the bare repo, AA scored the
    dated snapshot. Both name the same weights."""
    m = _model(hf_repo="deepseek-ai/DeepSeek-V4-Flash")
    aa = {"repos": {"deepseek-ai/deepseek-v4-flash-0731":
                    {"index": 50, "variant": "max"}},
          "identities": {"deepseekv4flash": {"index": 50, "variant": "max"}}}
    assert rr.aa_cell(m, aa) == "50"


def test_aa_cell_prefers_an_exact_repo_match_over_an_identity_match():
    m = _model(hf_repo="deepseek-ai/DeepSeek-V4-Flash")
    aa = {"repos": {"deepseek-ai/deepseek-v4-flash": {"index": 44, "variant": "default"}},
          "identities": {"deepseekv4flash": {"index": 50, "variant": "max"}}}
    assert rr.aa_cell(m, aa) == "44"


def test_arena_cell_falls_back_to_repo_identity():
    """Arena resolved the NVFP4 mirror; the tracked row is the BF16 release."""
    m = _model(hf_repo="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
               name="Nemotron 3 Ultra")
    ranks = {"repos": {}, "names": {}, "identities": {"nemotron3ultra550ba55b": 42}}
    assert rr.arena_cell(m, ranks) == "42"


def test_arena_cell_falls_back_to_a_vendor_prefixed_display_name():
    """Arena writes 'Tencent Hy3'; models.yaml calls the model 'Hy3'.

    The renderer's old private normalizer stripped a TRAILING org token but not
    a LEADING one, so this never matched.
    """
    m = _model(hf_repo="tencent/Hy3", name="Hy3")
    ranks = rr.load_arena_ranks_from_rows(
        [{"rank": 27, "model": "Tencent Hy3 Tencent · Apache 2.0",
          "resolved_repo": None}])
    assert rr.arena_cell(m, ranks) == "27"


def test_arena_cell_matches_thinking_machines_inkling():
    m = _model(hf_repo="thinkingmachines/Inkling", name="Inkling")
    ranks = rr.load_arena_ranks_from_rows(
        [{"rank": 36, "model": "Thinking Machines Inkling Thinky · Apache 2.0",
          "resolved_repo": None}])
    assert rr.arena_cell(m, ranks) == "36"


# --- collision guard ---------------------------------------------------------

def test_ambiguous_identity_is_dropped_rather_than_guessed(capsys):
    """Two sidecar entries sharing an identity means we cannot know which one
    the tracked row refers to. A wrong number is worse than no number."""
    m = _model(hf_repo="org/Model-C")
    aa = rr.load_aa_scores_from_dict({
        "org/Model-A": {"intelligence_index": 10},
        "org/Model-B": {"intelligence_index": 20},
    }, identity_of=lambda repo: "collide")
    assert rr.aa_cell(m, aa) == "—"
    assert "collide" in capsys.readouterr().out


def test_distinct_sizes_do_not_collide():
    """405B and 8B are different models. If repo_identity ever stripped size,
    this pair would share an identity and BOTH would be dropped by the guard."""
    aa = rr.load_aa_scores_from_dict({
        "meta-llama/Llama-3.1-405B-Instruct": {"intelligence_index": 30},
        "meta-llama/Llama-3.1-8B-Instruct": {"intelligence_index": 12},
    })
    assert aa["identities"]["llama31405b"]["index"] == 30
    assert aa["identities"]["llama318b"]["index"] == 12


def test_arena_same_repo_at_several_reasoning_efforts_keeps_the_best_rank():
    """The arena leaderboard legitimately lists one model at several reasoning
    efforts as separate rows, resolving to the SAME repo with different ranks.
    That must not look like a collision: the best (lowest) rank wins, silently,
    with no warning."""
    rows = [
        {"rank": 4, "model": "GLM 5.2 (Max) Z.ai · MIT",
         "resolved_repo": "zai-org/GLM-5.2"},
        {"rank": 8, "model": "GLM 5.2 (High) Z.ai · MIT",
         "resolved_repo": "zai-org/GLM-5.2"},
    ]
    ranks = rr.load_arena_ranks_from_rows(rows)
    assert ranks["identities"] == {"glm52": 4}


def test_arena_ambiguous_identity_is_dropped_rather_than_guessed(capsys):
    """Two DIFFERENT resolved repos that share a repo_identity (a bare repo and
    its dated-snapshot sibling) but carry different ranks cannot be resolved —
    dropped and reported, same as the AA-side guard."""
    rows = [
        {"rank": 4, "model": "DeepSeek V4 Flash 0731",
         "resolved_repo": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        {"rank": 9, "model": "DeepSeek V4 Flash",
         "resolved_repo": "deepseek-ai/DeepSeek-V4-Flash"},
    ]
    # Confirm the chosen pair actually collides under names.repo_identity.
    assert rr.names.repo_identity("deepseek-ai/DeepSeek-V4-Flash-0731") == \
        rr.names.repo_identity("deepseek-ai/DeepSeek-V4-Flash")

    ranks = rr.load_arena_ranks_from_rows(rows)
    assert ranks["identities"] == {}
    assert "deepseekv4flash" in capsys.readouterr().out
