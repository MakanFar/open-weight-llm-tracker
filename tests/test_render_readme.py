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
    aa = {"moonshotai/kimi-k3": {"aa_model": "Kimi K3 (max)",
                                 "intelligence_index": 57, "variant": "max"}}
    assert rr.aa_cell(m, aa) == "57"


def test_aa_cell_dashes_when_aa_does_not_rate_the_model():
    """There is no fallback to a manual figure: models.yaml carries no score."""
    assert rr.aa_cell(_model(hf_repo="org/m"), {}) == "—"


def test_aa_cell_does_an_exact_lookup_and_needs_no_identity_fallback():
    """aa_join keys the map by the ROW's own hf_repo, so there is nothing left
    to reconcile here.

    aa_cell used to fall back to repo_identity because the sidecar stored
    whichever repo string the scrape-time join landed on — AA scored
    DeepSeek-V4-Flash-0731 against a tracked DeepSeek-V4-Flash. The join now
    happens against the row being rendered, so the key IS the row's repo.
    Reconciling two spellings of it is no longer possible, or needed.
    """
    m = _model(hf_repo="deepseek-ai/DeepSeek-V4-Flash")
    aa = {"deepseek-ai/deepseek-v4-flash": {"aa_model": "DeepSeek V4 Flash",
                                            "intelligence_index": 44,
                                            "variant": "default"}}
    assert rr.aa_cell(m, aa) == "44"


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
        {"org/m": {"aa_model": "M", "intelligence_index": 42, "variant": "max"}},
        {"repos": {}, "names": {}, "identities": {}})
    head = table.splitlines()[0]
    assert "| AA Index |" in head
    assert "MMLU" not in head
    assert "42" in table.splitlines()[2]


# --- identity fallback -------------------------------------------------------

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
    """Two entries sharing an identity means we cannot know which one a
    tracked row refers to. A wrong number is worse than no number."""
    index = rr._index_by_identity(
        [("org/Model-A", 10), ("org/Model-B", 20)],
        identity_of=lambda repo: "collide")
    assert index == {}
    assert "collide" in capsys.readouterr().out


def test_identical_values_under_one_identity_are_kept():
    """Two spellings of the same repo naturally agree; that is not ambiguity."""
    assert rr._index_by_identity(
        [("org/Model-A", 7), ("org/Model-B", 7)],
        identity_of=lambda repo: "same") == {"same": 7}


def test_distinct_sizes_do_not_collide():
    """405B and 8B are different models. If repo_identity ever stripped size,
    this pair would share an identity and BOTH would be dropped by the guard."""
    index = rr._index_by_identity([
        ("meta-llama/Llama-3.1-405B-Instruct", 30),
        ("meta-llama/Llama-3.1-8B-Instruct", 12),
    ])
    assert index["llama31405b"] == 30
    assert index["llama318b"] == 12


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


# --- freshness stamp and badges --------------------------------------------

def test_load_generated_reads_the_discovery_stamp(tmp_path):
    f = tmp_path / "candidates.yaml"
    f.write_text("generated: '2026-08-28'\nmodels: []\n")
    assert rr.load_generated(f) == "2026-08-28"


def test_load_generated_accepts_an_unquoted_yaml_date(tmp_path):
    """YAML parses a bare 2026-08-28 into a date object, not a string."""
    f = tmp_path / "candidates.yaml"
    f.write_text("generated: 2026-08-28\nmodels: []\n")
    assert rr.load_generated(f) == "2026-08-28"


def test_load_generated_returns_none_when_missing_or_malformed(tmp_path):
    """Same contract as the arena and AA loaders: never raise, so the render
    always completes and the badge is simply omitted."""
    assert rr.load_generated(tmp_path / "nope.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("generated: '   '\nmodels: []\n")
    assert rr.load_generated(bad) is None
    worse = tmp_path / "worse.yaml"
    worse.write_text("[not, a, mapping]\n")
    assert rr.load_generated(worse) is None


def test_badges_double_the_hyphens_in_a_date():
    """A single hyphen is shields.io's field separator; an ISO date rendered
    raw would silently produce a badge reading '2026' with colour '08'."""
    out = rr.badges(33, "2026-08-28")
    assert "2026--08--28" in out


def test_badges_omit_the_freshness_one_when_nothing_is_stamped():
    """Better no claim than a badge implying a run that never happened."""
    out = rr.badges(33, None)
    assert "last%20discovery%20run" not in out
    assert "models-33-" in out


# --- which total the table prints ------------------------------------------

def test_table_prints_the_vendors_figure_when_there_is_one():
    """753.9 is the tensor count; 744 is what z.ai calls GLM-5 and what every
    reader will compare against. The measured anchor stays in models.yaml."""
    m = _model(params_total_b=753.9, params_active_b=40.0, architecture="moe",
               params_total_stated_b=744.0)
    assert rr.display_total(m) == 744.0


def test_table_falls_back_to_the_measured_count():
    """Most models publish no distinct headline figure. There is no competing
    claim for those rows, so the measured count IS the published number."""
    assert rr.display_total(_model(params_total_b=310.8)) == 310.8


def test_display_total_ignores_a_junk_stated_value():
    """row_errors rejects these, but candidates.yaml is hand-edited and this
    renderer must not raise on a row that slipped through."""
    assert rr.display_total(_model(params_total_b=70.0,
                                   params_total_stated_b="70B")) == 70.0
    assert rr.display_total(_model(params_total_b=70.0,
                                   params_total_stated_b=0)) == 70.0


# --- three states of commercial_use, not two -------------------------------

def test_commercial_badge_verified_is_bare():
    assert rr.commercial_badge(_model(commercial_use=True,
                                      commercial_use_verified=True)) == "Yes"


def test_commercial_badge_unverified_keeps_the_question_mark():
    assert rr.commercial_badge(_model(commercial_use=True)) == "Yes?"


def test_commercial_badge_marks_a_licence_the_vendor_never_published():
    """18 of 26 unverified rows ship no licence file at all. `?` reads as a
    backlog someone will work through; these can never be worked through, and
    conflating the two makes the `?` mean nothing."""
    m = _model(commercial_use=True, license_text_published=False)
    assert rr.commercial_badge(m) == "Yes†"


def test_verified_beats_the_unpublished_marker():
    """If a human settled it some other way, that wins over 'no file on HF'."""
    m = _model(commercial_use=True, commercial_use_verified=True,
               license_text_published=False)
    assert rr.commercial_badge(m) == "Yes"


def test_absent_flag_is_not_the_same_as_false():
    """Absent means nobody checked; False means someone checked and there is
    no licence text. Only the second earns the marker."""
    assert rr.commercial_badge(_model(commercial_use=True,
                                      license_text_published=None)) == "Yes?"
