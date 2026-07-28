#!/usr/bin/env python3
"""
Fetch MMLU scores for tracked models from the HF Open LLM Leaderboard and write
them to a committed sidecar, leaderboard_scores.yaml, keyed by hf_repo.

WHY A SIDECAR:
    render_readme.py must stay offline and deterministic (CI re-renders and
    diffs the README). So all network access lives here; render only reads the
    committed file. Nothing is written to models.yaml — it stays human-curated.

COVERAGE CAVEAT:
    The Open LLM Leaderboard v2 was archived; many 2026 frontier models are not
    listed. render_readme.py falls back to the manual benchmark.score for any
    repo missing here, so an empty or partial file is safe.

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


def extract_mmlu(row):
    """Read a plain-MMLU score from a leaderboard row, tolerant of key casing."""
    for k in row:
        if k.lower() == "mmlu":
            v = row[k]
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _repo_of(row):
    for k in _REPO_KEYS:
        v = row.get(k)
        if isinstance(v, str) and "/" in v:
            return v
    return None


def build_scores(repos, rows):
    """{repo: {mmlu, source}} for tracked repos found in rows with an MMLU."""
    by_repo = {}
    for row in rows:
        repo = _repo_of(row)
        if repo:
            mmlu = extract_mmlu(row)
            if mmlu is not None:
                by_repo[repo.lower()] = mmlu
    out = {}
    for repo in repos:
        mmlu = by_repo.get(repo.lower())
        if mmlu is not None:
            out[repo] = {"mmlu": mmlu, "source": LEADERBOARD_URL}
    return out


def fetch_rows(get_json=hf_meta._http_get_json):
    """Best-effort paginated fetch of leaderboard rows. [] on failure."""
    rows = []
    offset = 0
    while True:
        try:
            page = get_json(ROWS_URL.format(offset=offset))
        except Exception as exc:
            print(f"  ! leaderboard fetch failed at offset {offset}: {exc}")
            break
        items = (page or {}).get("rows") or []
        if not items:
            break
        rows.extend(it.get("row", it) for it in items if isinstance(it, dict))
        if len(items) < 100:
            break
        offset += 100
    return rows


def _tracked_repos():
    doc = yaml.safe_load(DATA.read_text()) or {}
    return [m["hf_repo"] for m in doc.get("models", []) if m.get("hf_repo")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true",
                    help="write scores: {} without any network call")
    args = ap.parse_args()

    repos = _tracked_repos()
    rows = [] if args.no_fetch else fetch_rows()
    scores = build_scores(repos, rows)

    header = ("# AUTO-FETCHED by scripts/pull_leaderboard.py — do not hand-edit.\n"
              "# MMLU from the HF Open LLM Leaderboard, keyed by hf_repo.\n"
              "# render_readme.py falls back to models.yaml benchmark.score when a\n"
              "# repo is absent here.\n")
    OUT.write_text(header + yaml.safe_dump({"scores": scores}, sort_keys=True,
                                           allow_unicode=True, width=100))
    print(f"Wrote {len(scores)} leaderboard score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
