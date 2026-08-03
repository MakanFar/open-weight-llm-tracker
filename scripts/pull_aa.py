#!/usr/bin/env python3
"""
Scrape the Artificial Analysis Intelligence Index into a committed sidecar,
aa_scores.yaml, keyed by hf_repo.

WHY THIS REPLACED THE HF LEADERBOARD:
    scripts/pull_leaderboard.py could not fill the benchmark column. HF Open
    LLM Leaderboard v2 publishes no plain MMLU, it is archived so no 2026 model
    appears in it, and the HF model card API returns no structured eval data at
    all — 0 of 42 tracked/candidate repos carry a model-index.

WHAT THE INDEX IS:
    A 0-100 composite: Agents 34%, Coding 24%, Scientific Reasoning 24%,
    General 18%. It is versioned and re-weighted periodically, so scores are
    NOT comparable across time. Only the current snapshot is stored.

OPENNESS IS NOT READ FROM AA:
    AA's page carries no openness column, and we would not trust it if it did.
    Proprietary rows simply fail to match models.yaml and fall out. Open-weight
    status stays decided by HF repo resolution alone.
"""
import re
import sys
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
OUT = ROOT / "aa_scores.yaml"

LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models"

# Cell positions in the data rows, matching the column header:
#   Model | Context Window | Creator | AA Intelligence Index | Cost | ...
_MODEL, _CREATOR, _INDEX = 0, 2, 3


def parse_leaderboard(html):
    """Rows of {model, creator, intelligence_index} from the leaderboard table.

    Header rows are not skipped positionally — they are discarded by the same
    integer check that discards unrated models, so a change in how many header
    rows AA emits cannot silently shift the data.
    """
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) <= _INDEX:
            continue
        raw = cells[_INDEX].strip()
        if not re.fullmatch(r"\d+", raw):
            continue
        rows.append({
            "model": cells[_MODEL],
            "creator": cells[_CREATOR],
            "intelligence_index": int(raw),
        })
    return rows


_PAREN_RE = re.compile(r"\(([^)]*)\)")


def split_variant(display):
    """('Kimi K3 (max)') -> ('Kimi K3', 'max'). No parenthetical -> 'default'."""
    found = _PAREN_RE.search(display)
    variant = found.group(1).strip().lower() if found else "default"
    base = _PAREN_RE.sub(" ", display)
    base = re.sub(r"\s+", " ", base).strip()
    return base, variant


def best_by_slug(rows):
    """Highest-scoring variant per model, keyed by slug.

    AA lists the same weights at several reasoning efforts and the spread is
    wide (Kimi K3 scores 57 at max, 47 at low), so the winner is chosen
    explicitly rather than by whichever row happens to come last. The variant
    that won is recorded so the number can be traced back to a row.
    """
    best = {}
    for row in rows:
        base, variant = split_variant(row["model"])
        key = names.slug(names.strip_variant_suffix(base))
        if not key:
            continue
        current = best.get(key)
        if current is None or row["intelligence_index"] > current["intelligence_index"]:
            best[key] = {
                "model_slug": key,
                "aa_model": row["model"],
                "variant": variant,
                "intelligence_index": row["intelligence_index"],
            }
    return best
