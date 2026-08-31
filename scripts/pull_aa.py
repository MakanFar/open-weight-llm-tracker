#!/usr/bin/env python3
"""
Scrape the Artificial Analysis Intelligence Index into a committed sidecar,
aa_scores.yaml, keyed by AA name slug.

THIS SCRIPT ONLY SCRAPES. It reads no models.yaml, no candidates.yaml, and
resolves no display name to a repo — aa_join.py does that at the point of
use. Keeping the join out of the write is what makes the sidecar a plain
record of the leaderboard rather than a snapshot of a join whose inputs move
underneath it; see aa_join's module docstring for why that mattered.

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
    Proprietary rows are stored like any other and simply go unclaimed when
    aa_join joins them against models.yaml. Open-weight status stays decided
    by HF repo resolution alone.
"""
import argparse
import re
import sys
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "aa_scores.yaml"

# The open-weights view. NOTE: ?weights=open is applied client-side by AA's
# JS — the server returns the full table either way (verified byte-identical,
# 130 scored rows, proprietary models still first). It is kept because it is
# the URL recorded as the file's `source`, so a human clicking through lands
# on the open-weights list. Do NOT rely on it to pre-filter: proprietary rows
# are stored like any other and simply go unclaimed when aa_join joins them
# against models.yaml.
LEADERBOARD_URL = "https://artificialanalysis.ai/leaderboards/models?weights=open"

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


def best_by_slug(rows):
    """Highest-scoring variant per model, keyed by slug.

    AA lists the same weights at several reasoning efforts and the spread is
    wide (Kimi K3 scores 57 at max, 47 at low), so the winner is chosen
    explicitly rather than by whichever row happens to come last. The variant
    that won is recorded so the number can be traced back to a row.
    """
    best = {}
    for row in rows:
        base, variant = names.split_variant(row["model"])
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


HEADER = (
    "# AUTO-SCRAPED by scripts/pull_aa.py — do not hand-edit.\n"
    "# Artificial Analysis Intelligence Index (0-100 composite), by AA name slug.\n"
    "# variant records which reasoning-effort row won (the highest scoring).\n"
    "# The index is re-weighted between versions, so values are NOT comparable\n"
    "# across time. EVERY scored row is stored, including the proprietary ones\n"
    "# this tracker will never carry: the name->repo join is NOT done here, it\n"
    "# is done by aa_join at the point of use, against rows as they stand then.\n"
)


def fetch_html(url, get=requests.get):
    """Leaderboard HTML, or None on any failure."""
    try:
        resp = get(url, timeout=30,
                   headers={"User-Agent": "Mozilla/5.0 (owlt-aa/1.0)"})
    except Exception as exc:
        print(f"  ! AA fetch failed: {exc}")
        return None
    if getattr(resp, "status_code", None) != 200:
        print(f"  ! AA returned HTTP {getattr(resp, 'status_code', '?')}")
        return None
    return resp.text


def write_scores(path, entries):
    """Write every scraped entry, keyed by slug.

    model_slug is dropped: it is the key. `source` is hoisted out of the rows
    — it is one URL for the whole scrape, and repeating it on ~130 entries
    buried the data under its own provenance.
    """
    scores = {key: {"aa_model": e["aa_model"],
                    "intelligence_index": e["intelligence_index"],
                    "variant": e["variant"]}
              for key, e in entries.items()}
    Path(path).write_text(HEADER + yaml.safe_dump(
        {"source": LEADERBOARD_URL, "scores": scores},
        sort_keys=True, allow_unicode=True, width=100))


def refresh(path, html):
    """Rewrite the sidecar. Returns the entry count, or None if nothing was written.

    None means the run failed and the committed file was left alone. A zero-row
    parse is a failure: AA is never legitimately empty, so an empty parse means
    the markup changed and writing would erase good data.
    """
    if html is None:
        print(f"  ! leaving {Path(path).name} unchanged")
        return None
    rows = parse_leaderboard(html)
    if not rows:
        print(f"  ! parsed 0 rows — AA markup may have changed; "
              f"leaving {Path(path).name} unchanged")
        return None
    entries = best_by_slug(rows)
    write_scores(path, entries)
    return len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="parse a saved page instead of fetching")
    args = ap.parse_args()

    html = Path(args.html).read_text() if args.html else fetch_html(LEADERBOARD_URL)
    n = refresh(OUT, html)
    if n is None:
        return
    print(f"Wrote {n} AA score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
