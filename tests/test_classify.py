import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import classify


def _row(**kw):
    # Schema-complete by default (adds the fields missing_vitals does not
    # check but validate.py — and now classify.schema_errors — does), so
    # route()'s "promote" tests exercise a row that would actually pass
    # validate.py, not one that would fail it for reasons unrelated to what
    # the test is checking.
    base = dict(name="M", hf_repo="org/m", developer="org",
                release_date=date(2025, 6, 1),
                params_total_b=70.0, params_active_b=70.0, architecture="dense",
                context_window=131072, modality="text", license="mit",
                commercial_use=True, downloads=0)
    base.update(kw)
    return base


# --- notability ------------------------------------------------------------

def test_notable_via_aa_index():
    assert classify.is_notable(_row(aa_index=57)) is True


def test_notable_via_arena_rank():
    assert classify.is_notable(_row(arena_rank=12)) is True


def test_notable_via_downloads_at_the_boundary():
    # today pinned to _row()'s default release_date (2025-06-01) so this
    # exercises the DOWNLOADS threshold boundary, not the new recency
    # window — release_date == today is trivially "within window".
    today = date(2025, 6, 1)
    assert classify.is_notable(_row(downloads=500_000), today=today) is True
    assert classify.is_notable(_row(downloads=499_999), today=today) is False


def test_not_notable_with_no_signal_and_few_downloads():
    """This is what keeps research artifacts out of models.yaml."""
    assert classify.is_notable(_row(downloads=1200)) is False


def test_aa_index_zero_is_a_real_value_not_absent_but_still_below_the_floor():
    """Artificial Analysis scores start at 0; a truthiness check would wrongly
    treat a genuine 0 as absent (None). But 0 is also far below
    AA_NOTABILITY_FLOOR, so — unlike before the floor existed — it must NOT
    confer notability on its own. It still must not raise or be confused
    with None: notability via another signal (downloads here) still works."""
    assert classify.is_notable(_row(aa_index=0)) is False
    assert classify.is_notable(_row(aa_index=0, downloads=600_000),
                               today=date(2025, 6, 1)) is True


def test_low_aa_index_alone_is_not_notable():
    """granite-4.1-3b-base at aa=5 is exactly the case the floor exists to
    reject: AA rated it and placed it nowhere near the frontier."""
    assert classify.is_notable(_row(aa_index=5)) is False


def test_aa_index_at_the_floor_boundary():
    assert classify.is_notable(_row(aa_index=classify.AA_NOTABILITY_FLOOR)) is True
    assert classify.is_notable(_row(aa_index=classify.AA_NOTABILITY_FLOOR - 1)) is False


def test_low_aa_index_still_notable_via_downloads():
    """gpt-oss-20b: aa 15, 8.5M downloads. The floor must gate the AA-alone
    path only, never weaken the downloads path."""
    assert classify.is_notable(_row(aa_index=15, downloads=8_500_000),
                               today=date(2025, 6, 1)) is True


def test_low_aa_index_still_notable_via_arena_rank():
    """The floor must not weaken the arena_rank path either."""
    assert classify.is_notable(_row(aa_index=5, arena_rank=12)) is True


def test_notable_with_arena_rank_zero():
    """Arena leaderboard ranks start at 0; a truthiness check would wrongly
    treat a genuine 0 as absent."""
    assert classify.is_notable(_row(arena_rank=0)) is True


def test_notable_with_downloads_none():
    """downloads=None must not raise; it should be treated as 0 (not notable)."""
    assert classify.is_notable(_row(downloads=None)) is False


# --- downloads recency gate --------------------------------------------

def test_downloads_notable_within_recency_window():
    """A recent model with real adoption is genuinely notable."""
    row = _row(downloads=600_000, release_date=date(2025, 1, 1))
    today = date(2025, 1, 1) + timedelta(
        days=classify.NOTABILITY_DOWNLOADS_MAX_AGE_DAYS - 1)
    assert classify.is_notable(row, today=today) is True


def test_downloads_not_notable_just_outside_recency_window():
    """The same row, one day past the window, must lose downloads-only
    notability — this is the Mistral-7B-v0.1 case (2023, 601k downloads)
    the window exists to reject."""
    row = _row(downloads=600_000, release_date=date(2025, 1, 1))
    today = date(2025, 1, 1) + timedelta(
        days=classify.NOTABILITY_DOWNLOADS_MAX_AGE_DAYS + 1)
    assert classify.is_notable(row, today=today) is False


def test_downloads_recency_window_boundary_is_inclusive():
    row = _row(downloads=600_000, release_date=date(2025, 1, 1))
    today = date(2025, 1, 1) + timedelta(
        days=classify.NOTABILITY_DOWNLOADS_MAX_AGE_DAYS)
    assert classify.is_notable(row, today=today) is True


def test_aa_path_unaffected_by_downloads_recency_window():
    """A five-year-old model AA still rates stays notable — the recency
    requirement governs the downloads path only."""
    row = _row(aa_index=57, release_date=date(2020, 1, 1), downloads=0)
    assert classify.is_notable(row, today=date(2026, 1, 1)) is True


def test_arena_path_unaffected_by_downloads_recency_window():
    """A five-year-old model still holding an arena rank stays notable —
    same rationale as the AA case above."""
    row = _row(arena_rank=3, release_date=date(2020, 1, 1), downloads=0)
    assert classify.is_notable(row, today=date(2026, 1, 1)) is True


def test_unparseable_release_date_does_not_block_downloads_path():
    """A bad release_date is a schema problem (see schema_errors), not
    evidence of staleness. is_notable must treat it as passing the recency
    check, so route() still sends the row to review — carrying the
    schema-invalid reason — rather than silently dropping it."""
    row = _row(downloads=600_000, release_date="sometime in 2025")
    assert classify.is_notable(row, today=date(2026, 1, 1)) is True
    assert classify.route(row, set(), today=date(2026, 1, 1)) == "review"


def test_missing_release_date_does_not_block_downloads_path():
    row = _row(downloads=600_000)
    del row["release_date"]
    assert classify.is_notable(row, today=date(2026, 1, 1)) is True


def test_context_window_true_rejected():
    """Python's bool is a subclass of int, so naive isinstance(ctx, int) would
    let True through as a context window. Must explicitly reject booleans."""
    assert "no-context-window" in classify.missing_vitals(_row(context_window=True), set())


# --- vitals ----------------------------------------------------------------

def test_complete_dense_row_has_no_missing_vitals():
    assert classify.missing_vitals(_row(), set()) == []


def test_moe_with_active_equal_to_total_is_incomplete():
    """The blocker on 15 of 17 signalled rows."""
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=753.3)
    assert "moe-active-params-unknown" in classify.missing_vitals(row, set())


def test_moe_with_a_real_active_figure_is_complete():
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=32.0)
    assert classify.missing_vitals(row, set()) == []


def test_zero_context_window_is_incomplete():
    assert "no-context-window" in classify.missing_vitals(_row(context_window=0), set())


def test_unallowlisted_licence_is_incomplete():
    assert "license-not-allowlisted" in classify.missing_vitals(_row(license="other"), set())


def test_inexact_repo_match_is_incomplete():
    assert "inexact-repo-match" in classify.missing_vitals(_row(needs_hf_repo=True), set())


def test_family_already_tracked_is_incomplete():
    """GLM-5.2 must not auto-promote while GLM-5.1 is tracked."""
    row = _row(hf_repo="zai-org/GLM-5.2")
    assert "family-already-tracked" in classify.missing_vitals(row, {"glm"})


def test_distill_repo_is_flagged_derivative_or_base():
    """Four of the 12 wrong auto-promotions were DeepSeek-R1 distills — a
    derivative of an already-tracked model, not a primary release."""
    row = _row(hf_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    assert "derivative-or-base" in classify.missing_vitals(row, set())


def test_distill_is_matched_case_insensitively_anywhere_in_the_repo_id():
    row = _row(hf_repo="deepseek-ai/deepseek-r1-DISTILL-llama-70b")
    assert "derivative-or-base" in classify.missing_vitals(row, set())


def test_trailing_base_repo_is_flagged_derivative_or_base():
    """granite-4.1-30b-base and Mistral-7B-v0.1-adjacent base releases were
    two more of the wrong promotions: pretraining checkpoints, not the
    models this index tracks as primary rows."""
    row = _row(hf_repo="ibm-granite/granite-4.1-30b-base")
    assert "derivative-or-base" in classify.missing_vitals(row, set())


def test_trailing_base_matches_underscore_separator_too():
    row = _row(hf_repo="org/some_model_Base")
    assert "derivative-or-base" in classify.missing_vitals(row, set())


def test_base_as_part_of_a_longer_word_is_not_flagged():
    """'base' must be its OWN trailing token (preceded by - or _), not a
    substring — a repo genuinely named ...-Firebase or ...-Codebase is not a
    base-model release and must not be misflagged."""
    row = _row(hf_repo="org/some-model-Firebase")
    assert "derivative-or-base" not in classify.missing_vitals(row, set())


def test_base_mid_repo_id_is_not_flagged():
    """Only a TRAILING base token counts — 'base' earlier in the id (e.g. a
    hypothetical 'base-camp' release) is not a base-model marker."""
    row = _row(hf_repo="org/base-camp-7B")
    assert "derivative-or-base" not in classify.missing_vitals(row, set())


def test_ordinary_repo_is_not_flagged_derivative_or_base():
    assert classify.missing_vitals(_row(), set()) == []


def test_reports_every_reason_not_just_the_first():
    row = _row(architecture="moe", params_total_b=100.0, params_active_b=100.0,
               context_window=0, license="other")
    reasons = classify.missing_vitals(row, set())
    assert set(reasons) >= {"moe-active-params-unknown", "no-context-window",
                            "license-not-allowlisted"}


# --- routing ---------------------------------------------------------------

def test_notable_and_complete_promotes():
    assert classify.route(_row(aa_index=29), set()) == "promote"


def test_notable_and_incomplete_goes_to_review():
    row = _row(aa_index=51, architecture="moe",
               params_total_b=753.3, params_active_b=753.3)
    assert classify.route(row, set()) == "review"


def test_not_notable_is_dropped_even_when_complete():
    """110 complete-but-unremarkable rows must never reach models.yaml."""
    assert classify.route(_row(downloads=10), set()) == "drop"


# --- schema gate -------------------------------------------------------

def test_schema_errors_flags_a_row_missing_vitals_would_wave_through():
    """release_date is not one of missing_vitals's checks (moe/context/
    licence/needs_hf_repo/family stem) but it IS one of validate.py's
    REQUIRED fields — this is the gap Finding A closes."""
    row = _row(release_date="sometime in 2025")
    assert classify.missing_vitals(row, set()) == []
    assert classify.schema_errors(row) != []


def test_notable_but_schema_invalid_goes_to_review_not_promote():
    row = _row(aa_index=29, release_date="sometime in 2025")
    assert classify.route(row, set()) == "review"


def test_notable_and_schema_valid_still_promotes():
    assert classify.route(_row(aa_index=29), set()) == "promote"
