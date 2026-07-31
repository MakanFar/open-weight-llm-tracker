#!/usr/bin/env python3
"""
Fetch MMLU scores for tracked models from the HF Open LLM Leaderboard and write
them to a committed sidecar, leaderboard_scores.yaml, keyed by hf_repo.

WHY A SIDECAR:
    render_readme.py must stay offline and deterministic (CI re-renders and
    diffs the README). So all network access lives here; render only reads the
    committed file. Nothing is written to models.yaml — it stays human-curated.

METRIC CAVEAT:
    Leaderboard v2 does not publish plain MMLU. Its 4576 rows carry MMLU-PRO
    and "MMLU-PRO Raw" only, so the original exact-match on "mmlu" scored
    nothing and wrote an empty sidecar — silently, because render falls back
    to models.yaml. Each entry now records the metric it actually captured;
    MMLU-PRO is a harsher scale (~40 where MMLU reads ~80) and must never be
    rendered in the same column as an MMLU figure.

COVERAGE CAVEAT:
    The Open LLM Leaderboard v2 was archived; many 2026 frontier models are not
    listed — of the 16 models tracked on 2026-07-30, 5 appear. render_readme.py
    falls back to the manual benchmark.score for any repo missing here, so an
    empty or partial file is safe.

Usage:
  .venv/bin/python scripts/pull_leaderboard.py            # fetch + write
  .venv/bin/python scripts/pull_leaderboard.py --no-fetch # rewrite from empty (offline)
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
OUT = ROOT / "leaderboard_scores.yaml"

LEADERBOARD_URL = "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
# HF datasets-server rows endpoint for the aggregated leaderboard contents.
# Verify field names against the live response on first real run; the parse
# logic below is tolerant and fully unit-tested with fixtures.
ROWS_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=open-llm-leaderboard/contents&config=default&split=train"
            "&offset={offset}&length=100")
_REPO_KEYS = ("fullname", "eval_name", "model", "Model")

# Metric columns we accept, best first. Plain MMLU is kept ahead of MMLU-PRO
# for archived v1 rows, but the live v2 dataset publishes only MMLU-PRO — see
# the METRIC CAVEAT above. "MMLU-PRO Raw" is deliberately absent: it is the
# 0-1 fraction, not the published percentage.
_METRIC_KEYS = ("mmlu", "mmlu-pro")


def extract_score(row):
    """Return (metric_name, value) from a leaderboard row, or None.

    Tolerant of key casing. The metric name is returned rather than assumed
    because MMLU and MMLU-PRO are different scales and the caller has to
    label which one it got.
    """
    lowered = {}
    for k, v in row.items():
        if isinstance(k, str):
            lowered.setdefault(k.strip().lower(), v)
    for key in _METRIC_KEYS:
        v = lowered.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return key.upper(), float(v)
    return None


def _repo_of(row):
    for k in _REPO_KEYS:
        v = row.get(k)
        if isinstance(v, str) and "/" in v:
            return v
    return None


def build_scores(repos, rows):
    """{repo: {metric, score, source}} for tracked repos found in rows."""
    by_repo = {}
    for row in rows:
        repo = _repo_of(row)
        if repo:
            found = extract_score(row)
            if found is not None:
                by_repo[repo.lower()] = found
    out = {}
    for repo in repos:
        found = by_repo.get(repo.lower())
        if found is not None:
            metric, score = found
            out[repo] = {"metric": metric, "score": round(score, 1),
                         "source": LEADERBOARD_URL}
    return out


def fetch_rows(get_json=hf_meta._http_get_json):
    """Paginated fetch of leaderboard rows. None if ANY page failed.

    None and [] mean different things: [] is "the dataset has no rows", None
    is "we could not read it". They must not be conflated, because the caller
    overwrites a committed file with whatever it gets back — returning [] on a
    429 erased real scores that had been fetched minutes earlier.

    A partial result is treated as failure too: stopping at a dead page would
    silently drop every score beyond it, which looks identical to those models
    being unlisted.
    """
    rows = []
    offset = 0
    while True:
        try:
            page = get_json(ROWS_URL.format(offset=offset))
        except Exception as exc:
            print(f"  ! leaderboard fetch failed at offset {offset}: {exc}")
            return None
        items = (page or {}).get("rows") or []
        if not items:
            break
        rows.extend(it.get("row", it) for it in items if isinstance(it, dict))
        if len(items) < 100:
            break
        offset += 100
    return rows


def refresh_scores(out_path, repos, rows):
    """Write the sidecar unless the fetch failed. Returns count, or None.

    rows=None (fetch failure) leaves the existing file exactly as it is.
    rows=[] is a legitimate empty result and is written.
    """
    if rows is None:
        print(f"  ! leaving {Path(out_path).name} unchanged")
        return None
    scores = build_scores(repos, rows)
    write_scores(out_path, scores)
    return len(scores)


def _tracked_repos():
    doc = yaml.safe_load(DATA.read_text()) or {}
    return [m["hf_repo"] for m in doc.get("models", []) if m.get("hf_repo")]


HEADER = ("# AUTO-FETCHED by scripts/pull_leaderboard.py — do not hand-edit.\n"
          "# HF Open LLM Leaderboard scores keyed by hf_repo. Each entry names\n"
          "# the metric it carries: v2 publishes MMLU-PRO, not MMLU, and the two\n"
          "# are different scales rendered in different README columns.\n"
          "# render_readme.py falls back to models.yaml benchmark.score for the\n"
          "# MMLU column when a repo is absent here.\n")


def write_scores(path, scores):
    Path(path).write_text(HEADER + yaml.safe_dump({"scores": scores},
                                                  sort_keys=True,
                                                  allow_unicode=True, width=100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="write scores: {} without any network call")
    args = ap.parse_args()

    repos = _tracked_repos()
    rows = [] if args.no_fetch else fetch_rows()
    n = refresh_scores(OUT, repos, rows)
    if n is None:
        return
    print(f"Wrote {n} leaderboard score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
