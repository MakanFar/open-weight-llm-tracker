#!/usr/bin/env python3
"""
Decide what happens to a discovered model: promote, review, or drop.

WHY A NOTABILITY BAR EXISTS:
    Completeness and worth are anti-correlated here. Of 358 discovered rows,
    those carrying a third-party signal were 2 complete / 15 incomplete, while
    those with no signal were 110 complete / 231 incomplete. Promoting on
    completeness alone would publish MagenticBrain, BAR-7B and granite
    previews while blocking Kimi K3, GLM-5.2 and gpt-oss-120b.

WHY DOWNLOADS AND NOT JUST THE LEADERBOARDS:
    A signal-only bar admits 17 rows and would reject 13 of the 16 models this
    tracker already curates by hand — a bar its own editorial practice
    rejects. Downloads catches the flagships no leaderboard rates (Llama-3-8B,
    the Qwen3-2507 line, GLM-4.7-Flash). It skews small, which is a known and
    accepted weakness: a large model with modest adoption and no leaderboard
    coverage will not auto-promote.
"""
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names
import validate

NOTABILITY_DOWNLOADS = 500_000

# Downloads is a LAGGING popularity metric: counts accumulate for years and
# favour whatever is small enough to run locally or useful as a fine-tune
# starting point. A dry run over the 358 staged candidates showed all 7
# current downloads-only promotions skewing old — Mistral-7B-v0.1 (2023, a
# BASE checkpoint) and Mistral-7B-Instruct-v0.2 (2023) outranked GLM-5.2 on
# raw download count alone, which is backwards: a recent model with real
# adoption is genuinely notable, an old one with accumulated adoption is
# just a popular base. AA and arena both already carry their own currency —
# AA only counts a score it still rates, arena only ranks the current
# leaderboard — so this window applies to the downloads path ONLY (see
# is_notable): a row must have been released within this many days to earn
# notability via adoption alone.
NOTABILITY_DOWNLOADS_MAX_AGE_DAYS = 365

# Below this, Artificial Analysis has already rated the model and placed it
# well short of the frontier — evidence the row needs a human, not evidence
# it should stay invisible. granite-4.1-3b-base at aa=5 is the case this
# exists to reject: "AA rated it at all" must not, on its own, publish it
# without anyone looking.
#
# THIS FLOOR DOES NOT GATE is_notable / STAGING. Any aa_index, however low,
# is enough to queue a row in candidates.yaml for review — a low score is
# evidence a model needs a second look, not evidence it is uninteresting.
# The floor gates only missing_vitals's unattended-PROMOTION check: a row
# whose sole claim to notability is an aa_index below this line is routed to
# review, never auto-promoted into models.yaml, no matter how complete it
# otherwise is. It must never weaken arena_rank or downloads as INDEPENDENT
# promotion signals: gpt-oss-20b (aa 15, 8.5M downloads) still promotes via
# downloads even though its AA score alone would not clear this bar.
#
# Several models this tracker ALREADY carries sit below 20 (Llama 3.3 70B,
# Llama 4 Scout/Maverick: aa 9-14) — a human hand-curated those for reasons
# other than their AA score, and this floor has no opinion on rows a human
# already decided to keep. It governs unattended promotion only.
AA_PROMOTION_FLOOR = 20

# A distill token anywhere in the repo id, case-insensitive. Deliberately a
# substring match, not a whole-token match: every real-world example
# ("-Distill-Qwen-32B", "-Distill-Llama-70B") has clean separators around it,
# and a substring match is simpler than tokenizing for no loss of accuracy
# here.
#
# KNOWN GAP: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B is a distill whose name
# never says so — it will still slip through this check. Catching it would
# require cross-referencing vendor announcements against repo ids, which is
# exactly the kind of heuristic this module elsewhere avoids building. Left
# as a known limitation rather than over-engineered around.
_DISTILL_TOKEN = re.compile(r"distill", re.IGNORECASE)

# A trailing "-base" or "_base" token, case-insensitive. Anchored to the END
# of the repo id and requires the - or _ separator immediately before "base"
# so a repo that merely ENDS in the letters "base" with no separator (e.g.
# ".../some-model-Firebase") is not misflagged as a base-model release.
_TRAILING_BASE_TOKEN = re.compile(r"[-_]base$", re.IGNORECASE)


def is_derivative_or_base(hf_repo):
    """True if the repo id names a distill or a base-model checkpoint.

    Both are derivative or non-final releases this index does not carry as
    primary rows — four of the wrong 12 auto-promotions this gate exists to
    fix were DeepSeek-R1 distills of an already-tracked model, and three more
    were base checkpoints (granite-4.1-30b-base, granite-4.1-3b-base,
    Mistral-7B-v0.1). See _DISTILL_TOKEN's docstring for the one distill this
    cannot catch.
    """
    repo = hf_repo or ""
    return bool(_DISTILL_TOKEN.search(repo) or _TRAILING_BASE_TOKEN.search(repo))


def _within_downloads_recency_window(row, today):
    """True unless release_date parses to a real date more than
    NOTABILITY_DOWNLOADS_MAX_AGE_DAYS days before `today`.

    A missing or unparseable release_date passes here on purpose: that is a
    SCHEMA defect (a hand-edited candidates.yaml row, e.g. release_date:
    "sometime in 2025"), not evidence the model is stale. Blocking the
    downloads path on a bad date too would make is_notable return False,
    route() would then return "drop", and the row would vanish silently
    instead of surfacing in the review queue. schema_errors() already
    catches the bad date and demotes the row to review with a
    schema-invalid reason — that is the correct outcome, and it only
    happens if is_notable lets the row past this check first.

    release_date may be a real datetime.date (rows built by this pipeline)
    or a string (hand-edited YAML) — isinstance guards against both without
    raising.
    """
    value = row.get("release_date")
    if not isinstance(value, date):
        return True
    return (today - value).days <= NOTABILITY_DOWNLOADS_MAX_AGE_DAYS


def is_notable(row, today=None):
    """True if the row is worth STAGING in candidates.yaml for a human to
    look at. This is a visibility question, not a publication question —
    keeping those separate is the whole point of this function's shape.
    Whether a notable row may PUBLISH unattended is a stricter, later
    question that missing_vitals answers (see AA_PROMOTION_FLOOR there).

    arena_rank and ANY aa_index (however low — a weak score is evidence the
    model needs review, not evidence it is uninteresting) confer notability
    with no recency requirement: AA only counts a score it still rates
    today, and arena only ranks the CURRENT leaderboard, so both are already
    self-refreshing signals of present relevance. downloads is not — it is
    a cumulative lifetime count — so it additionally requires the release to
    fall within NOTABILITY_DOWNLOADS_MAX_AGE_DAYS of `today` (see that
    constant's docstring). `today` defaults to date.today() but is
    injectable so callers (and tests) can pin it for determinism, matching
    the pattern discover.sweep_orgs already uses.
    """
    if row.get("arena_rank") is not None:
        return True
    if row.get("aa_index") is not None:
        return True
    if (row.get("downloads") or 0) < NOTABILITY_DOWNLOADS:
        return False
    return _within_downloads_recency_window(row, today or date.today())


def _clears_promotion_floor(row, today):
    """True if the row carries a signal strong enough to PUBLISH unattended:
    an arena rank, qualifying recent downloads, or an aa_index at or above
    AA_PROMOTION_FLOOR. Mirrors is_notable's three paths, except the AA leg
    is stricter here on purpose — is_notable treats any aa_index as reason
    enough to stage the row; this treats only a score clearing the floor as
    reason enough to publish it without a human confirming it first.
    """
    if row.get("arena_rank") is not None:
        return True
    if (row.get("downloads") or 0) >= NOTABILITY_DOWNLOADS and \
            _within_downloads_recency_window(row, today):
        return True
    aa = row.get("aa_index")
    return aa is not None and aa >= AA_PROMOTION_FLOOR


def missing_vitals(row, tracked_stems, today=None):
    """Every reason this row cannot be promoted unreviewed. [] means it can.

    Returns ALL reasons rather than the first, so one review pass shows a
    human everything the row needs.

    This only checks the fields worth an unassisted promotion (MoE active
    params, context window, licence allowlist, needs_hf_repo, family stem,
    distill/base derivative status, and — see below — whether the row's
    only claim to notability is an AA score too weak to publish without a
    human confirming it). It does NOT check release_date's type, the
    modality/architecture/
    commercial_use enums, or the dense params_active_b == params_total_b
    rule — those are validate.py's job, and route() below also calls
    schema_errors() to cover them before deciding to promote.

    `today` is injectable, defaulting to date.today(), purely because the
    weak-aa-signal check below needs "today" for its downloads-recency leg —
    same pattern as is_notable and route.
    """
    reasons = []

    if row.get("architecture") == "moe" and \
            row.get("params_active_b") == row.get("params_total_b"):
        reasons.append("moe-active-params-unknown")

    ctx = row.get("context_window")
    if not isinstance(ctx, int) or isinstance(ctx, bool) or ctx <= 0:
        reasons.append("no-context-window")

    if row.get("license") not in validate.LICENSES:
        reasons.append("license-not-allowlisted")

    if row.get("needs_hf_repo"):
        reasons.append("inexact-repo-match")

    if is_derivative_or_base(row.get("hf_repo")):
        reasons.append("derivative-or-base")

    # A collision here means "a human should look", not "this is a duplicate":
    # family_stem cannot tell a version bump from a distinct product line (see
    # its docstring — DeepSeek-V3 and DeepSeek-R1 both collapse to "deepseek").
    # Routing to review is the entire intended response to a collision.
    stem = names.family_stem(row.get("hf_repo") or "")
    if stem and stem in tracked_stems:
        reasons.append("family-already-tracked")

    # is_notable stages a row on ANY aa_index, however low; this is the
    # stricter question of whether that (or another) signal is strong enough
    # to publish unattended. Reached via route() this can only fire in the
    # weak-aa-alone case — arena_rank and the downloads leg here are
    # identical to is_notable's, so if either had made the row notable it
    # already clears the floor too. Called directly (e.g. discover.py's
    # re-enrichment gate) on a hand-edited candidates.yaml row with no
    # signal at all, it fires as well — which is correct: such a row is not
    # promotable unattended either.
    if not _clears_promotion_floor(row, today or date.today()):
        reasons.append("aa-below-promotion-floor")

    return reasons


def schema_errors(row):
    """Validator complaints validate.py would raise about this row if it were
    appended to models.yaml as-is. [] means the row is schema-clean.

    missing_vitals only judges whether a row is worth promoting unassisted;
    it says nothing about whether the row is even well-formed. Auto-built
    rows always are, but a carried-forward candidates.yaml row is hand-edited
    by a human and can carry a release_date like "sometime in 2025" —
    missing_vitals has no opinion on that, so without this check such a row
    would promote, and render_readme.py's release_date sort would then raise
    TypeError comparing a datetime.date to a str, killing the weekly PR with
    no PR ever opening. Reuses validate.row_errors rather than
    reimplementing CI's rules, so the two can never drift apart.
    """
    return validate.row_errors(row)


def route(row, tracked_stems, today=None):
    """'promote' | 'review' | 'drop'.

    Notability gates first: completeness alone is not evidence of worth (see
    module docstring), so an unremarkable-but-complete row is dropped rather
    than promoted. A notable row is then checked two ways before an
    unreviewed promotion is allowed: missing_vitals (is it worth promoting
    unassisted?) and schema_errors (would validate.py accept it?). Either one
    failing sends the row to review instead — a schema failure is never
    promoted, no matter how complete the row otherwise looks.

    today is threaded straight to is_notable and missing_vitals (see there)
    so the downloads path's recency window can be pinned for deterministic
    tests.
    """
    if not is_notable(row, today):
        return "drop"
    return "review" if missing_vitals(row, tracked_stems, today) or schema_errors(row) \
        else "promote"
