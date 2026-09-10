#!/usr/bin/env python3
"""Decide whether a discovery run produced anything worth a pull request.

WHY THIS EXISTS:
    discover.yml runs daily, and two things in its output move on EVERY run
    whether or not the tracker learned anything. write_candidates() re-stamps
    candidates.yaml with the run date, and the README renders that stamp as a
    freshness badge. peter-evans/create-pull-request opens a PR on any diff,
    so without a gate the daily schedule produces a date-bump PR every day —
    a rolling diff that shifts under a reviewer mid-review, or up to seven
    empty PRs a week. A human gate nobody reads is not a gate.

WHAT COUNTS AS SUBSTANTIVE:
    Only the four PUBLISHED files. arena_agent_rankings.yaml and
    aa_scores.yaml are deliberately NOT watched: they are scrape INPUTS, and
    arena's rows carry vote counts and confidence intervals that differ on
    literally every scrape, so watching them would gate on nothing. Their
    churn still reaches a PR when it matters, because it matters exactly when
    it changes a rendered file — an AA score that moves rewrites README.md's
    AA Index column, and that IS watched. When a PR does open, add-paths
    still carries the fresh sidecars along with it.

THE FAILURE MODE TO FEAR IS THE QUIET ONE:
    Over-suppression means a frontier release is discovered and then never
    surfaced, with no PR and no error to notice. That is strictly worse than
    the noise this removes, which is why the masking is deliberately narrow —
    two anchored, single-line patterns, each pinned by a test — rather than a
    tolerant "ignore anything date-shaped". Anything this file does not
    recognise counts as a change.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The published index. Everything a reader consumes, and nothing else.
WATCHED = ("models.yaml", "candidates.yaml", "README.md", "models.json")

# Anchored at line start and matching the whole line, so a mask can only ever
# eat the one line it is aimed at. Loosening either of these to a substring
# search is how this file would start hiding real diffs.
_STAMP_RE = re.compile(r"(?m)^generated:.*$")
_BADGE_RE = re.compile(r"(?m)^\[!\[index updated\].*$")
_JSON_STAMP_RE = re.compile(r'(?m)^\s*"generated":.*$')

# THREE files carry the run date, not two. models.json republishes the same
# stamp as a top-level field, and it was missed on the first pass here — the
# gate then fired on a simulated no-op run and reported models.json as a
# substantive change. That is the SAFE direction to be wrong in (a spurious
# PR, not a swallowed release) but it defeats the point, and it is the reason
# each mask below is pinned by a test to the code that emits it: a fourth
# consumer of load_generated() would otherwise reintroduce this silently.
_MASKS = {
    "candidates.yaml": (_STAMP_RE, "generated: <run date>"),
    "README.md": (_BADGE_RE, "<freshness badge>"),
    "models.json": (_JSON_STAMP_RE, '  "generated": <run date>'),
}


def mask(path, text):
    """Blank out the run-date fields, which move on every run by design.

    Returns text unchanged for a path with nothing to mask, and None for a
    file that does not exist (so appearing or vanishing reads as a change).
    """
    if text is None:
        return None
    rule = _MASKS.get(path)
    if rule is None:
        return text
    pattern, placeholder = rule
    return pattern.sub(placeholder, text)


def changed_paths(before, after, watched=WATCHED):
    """Watched paths that differ once the run stamp is masked out.

    before/after are {path: text or None}. A path missing from either mapping
    is treated as absent, not as unchanged: a file that appeared or was
    deleted is a change worth a PR.
    """
    return [path for path in watched
            if mask(path, before.get(path)) != mask(path, after.get(path))]


def _committed(path, rev="HEAD", root=ROOT):
    """The file as of `rev`, or None if it is not in that commit."""
    proc = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=str(root),
                          capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def _working(path, root=ROOT):
    try:
        return (Path(root) / path).read_text()
    except OSError:
        return None


def main(argv=None):
    rev = (argv or sys.argv[1:] or ["HEAD"])[0]
    before = {p: _committed(p, rev) for p in WATCHED}
    after = {p: _working(p) for p in WATCHED}
    changed = changed_paths(before, after)

    if changed:
        print(f"substantive changes in: {', '.join(changed)}", file=sys.stderr)
    else:
        print("nothing but the run stamp moved — no PR", file=sys.stderr)
    print("true" if changed else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
