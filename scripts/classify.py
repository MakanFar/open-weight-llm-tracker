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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names
import validate

NOTABILITY_DOWNLOADS = 500_000


def is_notable(row):
    """True if the model is worth publishing without a human asking for it."""
    if row.get("aa_index") is not None or row.get("arena_rank") is not None:
        return True
    return (row.get("downloads") or 0) >= NOTABILITY_DOWNLOADS


def missing_vitals(row, tracked_stems):
    """Every reason this row cannot be promoted unreviewed. [] means it can.

    Returns ALL reasons rather than the first, so one review pass shows a
    human everything the row needs.
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

    # A collision here means "a human should look", not "this is a duplicate":
    # family_stem cannot tell a version bump from a distinct product line (see
    # its docstring — DeepSeek-V3 and DeepSeek-R1 both collapse to "deepseek").
    # Routing to review is the entire intended response to a collision.
    stem = names.family_stem(row.get("hf_repo") or "")
    if stem and stem in tracked_stems:
        reasons.append("family-already-tracked")

    return reasons


def route(row, tracked_stems):
    """'promote' | 'review' | 'drop'.

    Notability gates first: completeness alone is not evidence of worth (see
    module docstring), so an unremarkable-but-complete row is dropped rather
    than promoted. Only a notable row's completeness is then checked to decide
    between an unreviewed promotion and a human review pass.
    """
    if not is_notable(row):
        return "drop"
    return "review" if missing_vitals(row, tracked_stems) else "promote"
