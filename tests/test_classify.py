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


def test_aa_index_zero_is_a_real_value_not_absent_and_still_notable():
    """Artificial Analysis scores start at 0; a truthiness check would wrongly
    treat a genuine 0 as absent (None). is_notable no longer applies a floor
    to the aa_index path at all — ANY aa_index, however low, is enough to
    STAGE the row for a human to look at (see AA_PROMOTION_FLOOR's docstring
    in classify.py for why that floor moved to missing_vitals instead)."""
    assert classify.is_notable(_row(aa_index=0)) is True


def test_low_aa_index_alone_is_still_notable():
    """granite-4.1-3b-base at aa=5 must still be STAGED — AA having rated it
    at all is a reason a human should look, not a reason to hide the row.
    Whether it PROMOTES unattended is missing_vitals's job (see the
    'promotion floor' tests below), not is_notable's."""
    assert classify.is_notable(_row(aa_index=5)) is True


def test_low_aa_index_still_notable_via_downloads():
    """gpt-oss-20b: aa 15, 8.5M downloads. Any aa_index already confers
    notability on its own now, but this also confirms the downloads path
    keeps working alongside it."""
    assert classify.is_notable(_row(aa_index=15, downloads=8_500_000),
                               today=date(2025, 6, 1)) is True


def test_low_aa_index_still_notable_via_arena_rank():
    """A weak aa_index alongside an arena_rank is still notable via either
    path independently."""
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
    # aa_index at the promotion floor: this test is about the OTHER vitals
    # checks (moe/context/licence/etc), so it needs a signal that itself
    # clears AA_PROMOTION_FLOOR, or the new weak-aa-signal reason would fire
    # and make an otherwise-complete row look incomplete.
    assert classify.missing_vitals(_row(aa_index=classify.AA_PROMOTION_FLOOR), set()) == []


def test_moe_with_active_equal_to_total_is_incomplete():
    """The blocker on 15 of 17 signalled rows."""
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=753.3)
    assert "moe-active-params-unknown" in classify.missing_vitals(row, set())


def test_moe_with_a_real_active_figure_is_complete():
    row = _row(architecture="moe", params_total_b=753.3, params_active_b=32.0,
               aa_index=classify.AA_PROMOTION_FLOOR)
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


def test_family_collision_reviewed_clears_the_collision():
    """A human who reviewed the collision and chose coexist unblocks the row.

    Without this, a notable, complete, schema-clean row whose ONLY blocker is
    a family collision regenerates identically on every run forever —
    candidates.yaml is rebuilt each run and models.yaml is append-only, so
    the reviewer's decision has nowhere to live. allenai/Olmo-3.1-32B-Think
    is the real row this exists for.
    """
    row = _row(hf_repo="zai-org/GLM-5.2", family_collision_reviewed=True)
    assert "family-already-tracked" not in classify.missing_vitals(row, {"glm"})


def test_family_collision_reviewed_only_honours_a_real_bool():
    """A non-bool truthy value must NOT clear the collision.

    candidates.yaml is hand-edited and validate.py never sees it, so this is
    the only place a typo can be caught. A string like "yes" (or "no" — also
    truthy) would otherwise silently publish a row no human actually cleared,
    which is exactly the failure validate.row_errors guards against for
    commercial_use_verified.
    """
    for value in ("yes", "no", 1, "true"):
        row = _row(hf_repo="zai-org/GLM-5.2", family_collision_reviewed=value)
        assert "family-already-tracked" in classify.missing_vitals(row, {"glm"}), \
            f"{value!r} must not clear the collision"


def test_family_collision_reviewed_does_not_clear_other_gaps():
    """The marker is scoped to the collision, not a blanket promote override."""
    row = _row(hf_repo="zai-org/GLM-5.2", family_collision_reviewed=True,
               context_window=0)
    assert "no-context-window" in classify.missing_vitals(row, {"glm"})


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
    assert classify.missing_vitals(_row(aa_index=classify.AA_PROMOTION_FLOOR), set()) == []


def test_reports_every_reason_not_just_the_first():
    row = _row(architecture="moe", params_total_b=100.0, params_active_b=100.0,
               context_window=0, license="other")
    reasons = classify.missing_vitals(row, set())
    assert set(reasons) >= {"moe-active-params-unknown", "no-context-window",
                            "license-not-allowlisted"}


# --- promotion floor (missing_vitals) ---------------------------------
# AA_PROMOTION_FLOOR no longer lives in is_notable (staging) — it lives here
# (unattended promotion). A weak aa_index stages the row but must never be
# enough, on its own, to promote it.

def test_weak_aa_alone_stages_but_does_not_auto_promote():
    """aa=15 (gpt-oss-20b's actual score) with no other signal must be
    STAGED (is_notable True) but never published unattended: route() sends
    it to review, and missing_vitals names the reason."""
    row = _row(aa_index=15)
    assert classify.is_notable(row) is True
    assert classify.route(row, set()) == "review"
    assert "aa-below-promotion-floor" in classify.missing_vitals(row, set())


def test_aa_at_the_promotion_floor_boundary_promotes():
    row = _row(aa_index=classify.AA_PROMOTION_FLOOR)
    assert classify.route(row, set()) == "promote"


def test_aa_one_below_the_promotion_floor_does_not_promote():
    row = _row(aa_index=classify.AA_PROMOTION_FLOOR - 1)
    assert classify.route(row, set()) == "review"


def test_weak_aa_with_arena_rank_still_promotes():
    """The arena rank is an independent strong signal — a weak AA score
    alongside it must not block promotion. today is pinned to the row's
    release_date so the (separate, newer) arena recency gate is cleared and
    this test isolates the AA interaction it is actually about."""
    today = date(2025, 6, 1)
    row = _row(aa_index=15, arena_rank=3, release_date=today)
    assert classify.route(row, set(), today=today) == "promote"


def test_weak_aa_with_qualifying_recent_downloads_still_promotes():
    today = date(2025, 6, 1)
    row = _row(aa_index=15, downloads=600_000, release_date=today)
    assert classify.route(row, set(), today=today) == "promote"


def test_weak_aa_with_old_high_downloads_does_not_promote():
    """Downloads that are high but stale don't count (see the recency
    window); a weak AA score alongside them must send the row to review,
    not promote it — is_notable is still True (aa_index is present), so the
    row is not dropped, only held back from unattended promotion."""
    today = date(2026, 1, 1)
    row = _row(aa_index=15, downloads=8_500_000, release_date=date(2020, 1, 1))
    assert classify.is_notable(row, today=today) is True
    assert classify.route(row, set(), today=today) == "review"


def test_no_aa_with_arena_rank_still_promotes():
    """An arena rank alone, with no AA score at all, still promotes a
    complete row -- provided the release is recent (see the arena-recency
    tests below). today is pinned to the row's release_date so this test
    exercises "no AA needed", not the new recency gate."""
    today = date(2025, 6, 1)
    row = _row(arena_rank=3, release_date=today)
    assert classify.route(row, set(), today=today) == "promote"


def test_arena_rank_within_recency_window_promotes():
    """A ranked row released inside NOTABILITY_DOWNLOADS_MAX_AGE_DAYS of
    today clears the promotion floor via arena_rank alone."""
    today = date(2025, 6, 1)
    row = _row(arena_rank=186, release_date=today)
    assert classify.route(row, set(), today=today) == "promote"


def test_arena_rank_outside_recency_window_reviews_not_drops():
    """The regression this task exists to fix: arena.ai's scraper now reads
    the historical text leaderboard (213 models back to 2023), not the
    current-only Agent board the old exemption assumed. An arena rank is
    still evidence the model is GOOD; it is no longer evidence the model is
    CURRENT, so an old ranked row must not auto-publish. It must also not
    be silently dropped -- an arena rank is still a real signal a human
    should see, so it stays in the review queue.

    microsoft/Phi-3-mini-4k-instruct is the actual regression: ranked #186,
    released 2024-04-22, and one of 15 stale models a live discover.py run
    auto-promoted before this fix.
    """
    old = date(2024, 4, 22)
    today = date(2026, 8, 10)
    row = _row(arena_rank=186, release_date=old)
    assert classify.route(row, set(), today=today) == "review"


def test_arena_rank_outside_recency_window_still_notable():
    """Staging is unaffected by the recency gate -- only unattended
    promotion is. An old ranked row must still surface in candidates.yaml
    for a human to see (see test_arena_path_unaffected_by_downloads_recency_window
    above, which covers is_notable directly; this confirms route agrees)."""
    old = date(2024, 4, 22)
    today = date(2026, 8, 10)
    row = _row(arena_rank=186, release_date=old)
    assert classify.is_notable(row, today=today) is True


def test_aa_at_floor_outside_recency_window_still_promotes():
    """AA keeps its recency exemption on the promotion floor: Artificial
    Analysis genuinely delists models it no longer rates (verified), so an
    aa_index at or above AA_PROMOTION_FLOOR is still evidence of CURRENT
    relevance even for an old release -- unlike an arena rank, which is
    now recency-gated (see test_arena_rank_outside_recency_window_reviews_not_drops)."""
    old = date(2024, 4, 22)
    today = date(2026, 8, 10)
    row = _row(aa_index=classify.AA_PROMOTION_FLOOR, release_date=old)
    assert classify.route(row, set(), today=today) == "promote"


def test_arena_rank_with_unparseable_release_date_not_blocked():
    """A hand-edited candidates.yaml row with a bad release_date is a schema
    problem (see schema_errors), not evidence of staleness -- the arena-
    recency check specifically must not treat it as stale. The row still
    ends up in review (schema_errors catches the bad date), but not because
    this check misread a string as "old": missing_vitals must not carry
    "aa-below-promotion-floor" for it."""
    today = date(2026, 8, 10)
    row = _row(arena_rank=186, release_date="sometime in 2025")
    assert "aa-below-promotion-floor" not in classify.missing_vitals(row, set(), today=today)
    assert classify.schema_errors(row) != []
    assert classify.route(row, set(), today=today) == "review"


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
    licence/needs_hf_repo/family stem/promotion-floor) but it IS one of
    validate.py's REQUIRED fields — this is the gap Finding A closes."""
    row = _row(release_date="sometime in 2025", aa_index=classify.AA_PROMOTION_FLOOR)
    assert classify.missing_vitals(row, set()) == []
    assert classify.schema_errors(row) != []


def test_string_params_total_b_is_schema_invalid_and_routes_to_review():
    """The exact pre-merge-review scenario: a hand-edited "700B" (units left
    on) clears every missing_vitals check -- MoE active params, context,
    licence, needs_hf_repo, family stem are all fine -- so before validate's
    type check existed this promoted straight through, and
    render_readme.human_params then raised ValueError on f"{total:g}" for a
    str, killing the weekly PR after the schema gate had already said OK.

    architecture="moe" here is deliberate, not incidental: with the default
    "dense" the pre-existing active==total equality check happens to catch
    mismatched strings too ("700B" != "22B"), which would mask whether the
    new type check is actually what is doing the work. MoE has no such
    check, so this isolates the fix precisely -- before it, this row's
    schema_errors was [] and route was "promote".
    """
    row = _row(architecture="moe", params_total_b="700B", params_active_b="22B",
               aa_index=classify.AA_PROMOTION_FLOOR)
    assert classify.missing_vitals(row, set()) == []
    assert classify.schema_errors(row) != []
    assert classify.route(row, set()) == "review"


def test_notable_but_schema_invalid_goes_to_review_not_promote():
    row = _row(aa_index=29, release_date="sometime in 2025")
    assert classify.route(row, set()) == "review"


def test_notable_and_schema_valid_still_promotes():
    assert classify.route(_row(aa_index=29), set()) == "promote"
