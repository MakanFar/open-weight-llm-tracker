import sys
from datetime import date
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
    assert classify.is_notable(_row(downloads=500_000)) is True
    assert classify.is_notable(_row(downloads=499_999)) is False


def test_not_notable_with_no_signal_and_few_downloads():
    """This is what keeps research artifacts out of models.yaml."""
    assert classify.is_notable(_row(downloads=1200)) is False


def test_notable_with_aa_index_zero():
    """Artificial Analysis scores start at 0; a truthiness check would wrongly
    treat a genuine 0 as absent."""
    assert classify.is_notable(_row(aa_index=0)) is True


def test_notable_with_arena_rank_zero():
    """Arena leaderboard ranks start at 0; a truthiness check would wrongly
    treat a genuine 0 as absent."""
    assert classify.is_notable(_row(arena_rank=0)) is True


def test_notable_with_downloads_none():
    """downloads=None must not raise; it should be treated as 0 (not notable)."""
    assert classify.is_notable(_row(downloads=None)) is False


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
