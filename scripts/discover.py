#!/usr/bin/env python3
"""
Discover NEW open-weight LLMs and stage them as candidates for review. This
does NOT touch models.yaml directly — it writes candidates.yaml so a human
approves them (via PR) before they land.

WHY AN ORG SWEEP, NOT A GLOBAL SCAN:
    This script used to sort ALL of Hugging Face by created_at and take the
    newest N. That population is dominated by finetunes and quantizations, so
    a frontier release essentially never appears in it. A run on 2026-07-21
    scanned 300 models and produced zero candidates (110 derivative, 189
    non-org, 1 blocked — every single one filtered).

    Now we issue one list_models(author=org) call per allowlisted org. That
    surfaced GLM-5.2, Kimi-K2.7, MiniMax-M3 and Gemma 4 immediately.

    Consequence: the org-follower probe that policed the unbounded query is
    gone. The allowlist IS the query.

WHERE NEW ORGS COME FROM:
    An allowlist only finds what it already knows. scripts/pull_arena.py
    surfaces orgs seen on the arena leaderboard that we have no mapping for
    (Tencent, Xiaomi, Thinking Machines as of 2026-07-21) and they are
    reported here for a human to add.

Usage:
  pip install -r requirements.txt
  python scripts/discover.py                  # org sweep (+ arena if present)
  python scripts/discover.py --min-params 3
  python scripts/discover.py --no-arena       # skip the arena merge
  HUGGINGFACE_TOKEN=hf_xxx python scripts/discover.py   # higher rate limits

NOTES / deliberate choices (the "don'ts"):
  - We do NOT exclude gated models — Llama & Gemma are gated; excluding them
    would drop the flagships. We keep them and flag for review.
  - safetensors.total is TOTAL params. For MoE we cannot infer active params
    from the API, so params_active_b is left equal to total and flagged TODO.
  - We do NOT trust card eval results as the benchmark column — left blank.
  - license tag is uploader-supplied; commercial_use is a *guess* to be checked.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("Install deps first:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"
ARENA = ROOT / "arena_agent_rankings.yaml"

# The orgs we sweep. This list IS the query — adding an org here is how the
# tracker gains coverage. pull_arena.py reports leaderboard orgs missing here.
ORG_ALLOWLIST = [
    "meta-llama", "Qwen", "deepseek-ai", "mistralai", "google", "microsoft",
    "CohereForAI", "CohereLabs", "ai21labs", "allenai", "nvidia", "01-ai",
    "tiiuae", "databricks", "HuggingFaceTB", "ibm-granite", "internlm",
    "THUDM", "zai-org", "moonshotai", "openai", "xai-org", "stabilityai",
    "MiniMaxAI", "XiaomiMiMo", "poolside", "thinkingmachines", "baidu",
]


def existing_repos():
    """Every hf_repo already tracked or already staged, lowercased."""
    repos = set()
    for path in (DATA, CANDIDATES):
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text()) or {}
        for m in doc.get("models", []) or []:
            if m.get("hf_repo"):
                repos.add(m["hf_repo"].lower())
    return repos


def sweep_orgs(api, orgs, min_params, known):
    """One list_models call per org. Returns (candidates, skip_counts).

    A failure on one org is logged and skipped — it never aborts the sweep.
    """
    candidates = []
    skips = {"known": 0, "derivative": 0, "small": 0, "license": 0,
             "no_params": 0, "org_error": 0}
    seen = set(known)

    for org in orgs:
        try:
            models = api.list_models(
                author=org,
                pipeline_tag="text-generation",
                sort="created_at",
                limit=50,
                expand=hf_meta.EXPAND,
            )
        except Exception as exc:
            print(f"  ! {org}: {exc}")
            skips["org_error"] += 1
            continue

        for info in models:
            if info.id.lower() in seen:
                skips["known"] += 1
                continue
            keep, reason = hf_meta.should_track(info, min_params)
            if not keep:
                skips[reason] += 1
                continue
            candidates.append(
                hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"]))
            seen.add(info.id.lower())

    return candidates, skips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-params", type=float, default=3.0,
                    help="minimum total params in billions")
    ap.add_argument("--no-arena", action="store_true",
                    help="skip merging arena_agent_rankings.yaml")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    known = existing_repos()

    print(f"Sweeping {len(ORG_ALLOWLIST)} orgs (min {args.min_params}B params)")
    candidates, skips = sweep_orgs(api, ORG_ALLOWLIST, args.min_params, known)
    print(f"  org sweep found {len(candidates)} candidate(s); skipped: {skips}")

    candidates.sort(key=lambda c: c["release_date"], reverse=True)

    header = (
        "# AUTO-GENERATED candidate models from scripts/discover.py\n"
        "# Review each entry, fix the TODO fields (active params, architecture,\n"
        "# context, benchmark, commercial_use), then move approved rows into\n"
        "# models.yaml and delete them here.\n"
    )
    CANDIDATES.write_text(header + yaml.safe_dump({"models": candidates},
                                                  sort_keys=False,
                                                  allow_unicode=True, width=100))
    print(f"\nWrote {len(candidates)} candidate(s) to {CANDIDATES.name}")
    # exit 0 always; the Action decides whether the diff is non-empty


if __name__ == "__main__":
    main()
