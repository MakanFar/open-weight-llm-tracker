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
# just a popular base. For is_notable (STAGING), AA and arena both still
# carry their own currency — AA only counts a score it still rates — so
# this window applies to the downloads leg of is_notable ONLY. Arena is the
# exception documented at is_notable: the scraper now reads a historical
# leaderboard, not a current-only one, so an arena rank no longer implies
# recency the way a fresh AA score does. This same window is therefore also
# applied to the arena leg of _clears_promotion_floor (unattended PROMOTION
# only — see that function): a row must have been released within this many
# days of `today` to earn notability via downloads, or to auto-publish via
# an arena rank.
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


def _within_recency_window(row, today):
    """True unless release_date parses to a real date more than
    NOTABILITY_DOWNLOADS_MAX_AGE_DAYS days before `today`.

    Shared by every caller that needs "is this release still current": the
    downloads leg of is_notable, and the downloads AND arena legs of
    _clears_promotion_floor. One predicate, reused, so those checks cannot
    drift apart on what "recent" means.

    A missing or unparseable release_date passes here on purpose: that is a
    SCHEMA defect (a hand-edited candidates.yaml row, e.g. release_date:
    "sometime in 2025"), not evidence the model is stale. Blocking a caller
    on a bad date too would make it fail for the wrong reason — e.g.
    is_notable would return False, route() would then return "drop", and
    the row would vanish silently instead of surfacing in the review queue.
    schema_errors() already catches the bad date and demotes the row to
    review with a schema-invalid reason — that is the correct outcome, and
    it only happens if this check lets the row past first.

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
    question that _clears_promotion_floor answers (see AA_PROMOTION_FLOOR
    there, and the arena note there too — this function's arena leg and
    that one's no longer agree, on purpose; see below).

    ANY aa_index (however low — a weak score is evidence the model needs
    review, not evidence it is uninteresting) confers notability with no
    recency requirement: AA only counts a score it still rates today, so an
    aa_index is a self-refreshing signal of present relevance.

    arena_rank ALSO confers notability with no recency requirement here —
    but not because arena is self-refreshing the way AA is. That used to be
    true (this docstring used to claim "arena only ranks the CURRENT
    leaderboard"), back when the scraper read arena.ai's Agent board (46
    live models). It now reads arena's TEXT leaderboard, a historical
    ranking of 213 models spanning years — #211 is meta-llama/Llama-2-13b.
    An arena rank is evidence a model is GOOD; it is no longer evidence a
    model is CURRENT — those are different claims, and this function only
    ever needed the first one: staging is where a human looks at a row, so
    surfacing an old ranked model in candidates.yaml is the correct,
    harmless outcome. Publishing it unattended is not, which is why
    _clears_promotion_floor now requires an arena-ranked row to ALSO fall
    within NOTABILITY_DOWNLOADS_MAX_AGE_DAYS, exactly like its downloads
    leg — see that function.

    downloads is neither self-refreshing: it is a cumulative lifetime
    count, so it additionally requires the release to fall within
    NOTABILITY_DOWNLOADS_MAX_AGE_DAYS of `today` (see that constant's
    docstring) even just to stage. `today` defaults to date.today() but is
    injectable so callers (and tests) can pin it for determinism, matching
    the pattern discover.sweep_orgs already uses.
    """
    if row.get("arena_rank") is not None:
        return True
    if row.get("aa_index") is not None:
        return True
    if (row.get("downloads") or 0) < NOTABILITY_DOWNLOADS:
        return False
    return _within_recency_window(row, today or date.today())


def _clears_promotion_floor(row, today):
    """True if the row carries a signal strong enough to PUBLISH unattended:
    a RECENT arena rank, qualifying recent downloads, or an aa_index at or
    above AA_PROMOTION_FLOOR. Mirrors is_notable's three paths, with two
    deliberate differences:

    - the AA leg is stricter — is_notable treats any aa_index as reason
      enough to STAGE the row; this treats only a score clearing the floor
      as reason enough to PUBLISH it without a human confirming it first.
      AA stays exempt from the recency window on both functions: Artificial
      Analysis genuinely delists models it no longer rates (verified), so
      an aa_index being present at all is still evidence the model is
      CURRENT, not just that it once scored well.

    - the arena leg is ALSO stricter here, unlike in is_notable: an arena
      rank alone no longer clears this floor. See is_notable's docstring —
      the scraper now reads arena's historical text leaderboard (213
      models back to 2023), so a rank is evidence of quality, not currency.
      Unattended publication needs both, so the arena leg is held to the
      exact same NOTABILITY_DOWNLOADS_MAX_AGE_DAYS window as the downloads
      leg below. microsoft/Phi-3-mini-4k-instruct (arena #186, released
      2024-04-22) is the actual regression this exists to stop: ranked,
      complete, and stale — a live discover.py run auto-promoted it (and
      14 other 2023-2024 models the same way) before this check existed.
      It must route to review instead, never drop — the rank is still a
      real signal, just not one strong enough to publish unattended.
    """
    if row.get("arena_rank") is not None and _within_recency_window(row, today):
        return True
    if (row.get("downloads") or 0) >= NOTABILITY_DOWNLOADS and \
            _within_recency_window(row, today):
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
    #
    # family_collision_reviewed is that human's answer coming back. Without
    # it the collision is unresolvable by construction: candidates.yaml is
    # rebuilt every run and models.yaml is append-only, so a reviewer who
    # decides "coexist" has nowhere to record it and the row regenerates
    # identically forever (22 of 85 staged rows sat here, one of them —
    # allenai/Olmo-3.1-32B-Think — notable, complete and schema-clean with
    # this as its ONLY blocker). Setting the field promotes the row on the
    # next run, and discover.PROMOTION_STRIP_FIELDS drops the marker on the
    # way into models.yaml, so it exists only for as long as the question does.
    #
    # `is True`, not a truthiness test: candidates.yaml is hand-edited and
    # validate.py never reads it, so this is the only place a typo can be
    # caught. `family_collision_reviewed: no` parses to the string "no" under
    # a quoted spelling and is truthy — a bare `if` would read a reviewer's
    # explicit "no" as "yes" and publish the row. Same reasoning as
    # validate.row_errors's isinstance guard on commercial_use_verified.
    #
    # It clears the collision ONLY. Every other reason still applies, so the
    # marker can never become a blanket promote override.
    stem = names.family_stem(row.get("hf_repo") or "")
    if stem and stem in tracked_stems and \
            row.get("family_collision_reviewed") is not True:
        reasons.append("family-already-tracked")

    # is_notable stages a row on ANY aa_index or arena_rank, however low or
    # stale; this is the stricter question of whether that (or another)
    # signal is strong enough to publish unattended. The reason string kept
    # its original name ("aa-below-promotion-floor") even though it now
    # also covers a stale arena rank — both are the same shape of failure
    # (a real signal that doesn't clear the unattended-publish bar), and a
    # human reading candidates.yaml only needs to know the row needs a
    # second look, not which leg tripped. Reached via route() this fires
    # when: the row's only signal is a weak aa_index, OR its arena_rank is
    # present but the release is outside NOTABILITY_DOWNLOADS_MAX_AGE_DAYS
    # (see _clears_promotion_floor — is_notable's arena leg is deliberately
    # more permissive than this one now), OR its downloads are high but
    # stale. Called directly (e.g. discover.py's re-enrichment gate) on a
    # hand-edited candidates.yaml row with no signal at all, it fires as
    # well — which is correct: such a row is not promotable unattended
    # either.
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
