#!/usr/bin/env python3
"""Re-derive every candidate row's `needs_review` list in place.

WHY THIS EXISTS:
    discover.py computes needs_review while it builds the queue, and then
    writes candidates.yaml and exits. Anything that edits the file AFTER that
    — a reviewer filling a gap by hand, or the agent step in discover.yml —
    changes the facts without changing the verdict, so the row ends up
    carrying `params_active_b: 16.0` and `needs_review:
    [moe-active-params-unknown]` at the same time. Contradictory, and it
    hides what is genuinely still outstanding from the person reading the PR.

    The stale list is harmless to the pipeline itself — the next run rebuilds
    it, and classify.route() re-derives its own verdict rather than trusting
    the field — but "harmless to the machine, misleading to the human" is not
    good enough for the one file a reviewer is asked to work through.

WHAT IT IS NOT:
    Not a promotion path. It only rewrites needs_review; it never moves a row
    into models.yaml, never touches models.yaml at all, and never edits any
    other field. Promotion stays with discover.py on the next run, which
    re-runs the full notability + vitals + schema gate. That ordering is
    deliberate: an edit lands in a PR a human sees BEFORE the row it unblocks
    can publish.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify
import discover
import names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"


def _load(path):
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        raise SystemExit(f"{path}: expected a mapping with a `models:` list")
    return doc


def restate(rows, tracked, today=None):
    """Set each row's needs_review from the facts it now carries.

    Returns [(row, before, after)] for rows whose verdict changed. A row that
    ends with nothing outstanding loses the key entirely rather than carrying
    an empty list: `needs_review: []` reads as "reviewed and cleared", which
    is a claim this function is in no position to make.
    """
    changed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        before = list(row.get("needs_review") or [])
        after = classify.review_reasons(row, tracked, today=today)
        if after == before:
            continue
        if after:
            row["needs_review"] = after
        else:
            row.pop("needs_review", None)
        changed.append((row, before, after))
    return changed


def count_with(rows, reasons):
    """Rows carrying any of `reasons`. Used to skip an agent step with no work."""
    wanted = set(reasons)
    return [r for r in rows if isinstance(r, dict)
            and wanted & set(r.get("needs_review") or [])]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="apply the changes (default: report only)")
    ap.add_argument("--count-with", nargs="+", metavar="REASON",
                    help="print how many rows carry any of these reasons, "
                         "then exit without touching the file")
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--data", default=str(DATA))
    args = ap.parse_args(argv)

    doc = _load(args.candidates)
    rows = doc["models"]

    if args.count_with:
        hits = count_with(rows, args.count_with)
        for row in hits:
            print(f"  {row.get('hf_repo')}: "
                  f"{', '.join(row.get('needs_review') or [])}", file=sys.stderr)
        print(len(hits))
        return 0

    tracked = discover.tracked_identities(args.data)
    changed = restate(rows, tracked)

    for row, before, after in changed:
        cleared = [r for r in before if r not in after]
        added = [r for r in after if r not in before]
        bits = []
        if cleared:
            bits.append("cleared " + ", ".join(cleared))
        if added:
            bits.append("now also " + ", ".join(added))
        print(f"  {row.get('hf_repo')}: {'; '.join(bits)}")

    if not changed:
        print("needs_review is already current on every row.")
        return 0
    if not args.write:
        print(f"\n{len(changed)} row(s) would change. Re-run with --write.")
        return 0

    discover.write_candidates(args.candidates, rows,
                              generated=doc.get("generated"))
    print(f"\nRewrote {len(changed)} row(s) in {args.candidates}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
