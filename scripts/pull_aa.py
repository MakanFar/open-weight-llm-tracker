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
DATA = ROOT / "models.yaml"
OUT = ROOT / "aa_scores.yaml"

# The open-weights view. NOTE: ?weights=open is applied client-side by AA's
# JS — the server returns the full table either way (verified byte-identical,
# 130 scored rows, proprietary models still first). It is kept because it is
# the URL recorded as each score's `source`, so a human clicking through lands
# on the open-weights list. Do NOT rely on it to pre-filter: the filtering that
# matters happens when match_to_tracked joins against models.yaml, and
# proprietary rows simply fail to match.
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


def tracked_models(path=DATA):
    """models.yaml rows that carry an hf_repo."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    rows = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("hf_repo")]


def _keys_for(model):
    """Slugs a tracked model may be known by on a leaderboard, in try-order.

    The name is tried before the hf_repo tail: models.yaml's `name` is the
    curated, human-facing label, while the repo tail is a fallback for when
    AA's display name doesn't match it (e.g. "Some Model" vs repo
    "zai-org/GLM-5.2"). Order must be explicit and a set cannot express it —
    iterating a set is hash-randomized, so if the two sources ever resolved to
    different AA entries, which one won would vary between runs on identical
    inputs. Returning an ordered, deduplicated list fixes both the order and
    the case where the two sources agree (no repeated key).
    """
    candidates = [model["name"], model["hf_repo"].split("/")[-1]]
    keys = []
    for c in candidates:
        if not c:
            continue
        key = names.slug(names.strip_variant_suffix(c))
        if key and key not in keys:
            keys.append(key)
    return keys


def match_to_tracked(best, tracked):
    """Join AA entries onto tracked models. Returns (scores, unmatched).

    Matching is local: AA publishes display names, not repo ids, and doing the
    lookup here avoids depending on HF search, whose rate limits have already
    caused silent data loss elsewhere in this repo.

    An AA entry may be claimed by at most one tracked model, first claim wins.
    Two tracked rows can produce overlapping keys (a near-duplicate
    models.yaml entry, or one row's name-key colliding with another row's
    repo-tail-key); without this guard both would silently get a score
    pointing at the same AA measurement. The collision is also printed so it
    surfaces during a discovery run instead of quietly corrupting the data.

    unmatched is expected to be long — most AA rows are proprietary models this
    tracker will never carry — so it is informational, never an error. An
    entry claimed by any tracked model (even one whose claim was rejected as a
    duplicate) is not unmatched.
    """
    scores, claimed_by = {}, {}
    for model in tracked:
        for key in _keys_for(model):
            entry = best.get(key)
            if entry is None:
                continue
            prior = claimed_by.get(key)
            if prior is not None:
                print(f"  ! {model['hf_repo']} and {prior} both match AA "
                      f"model {entry['aa_model']!r}; keeping {prior}")
                break
            claimed_by[key] = model["hf_repo"]
            scores[model["hf_repo"]] = {
                "intelligence_index": entry["intelligence_index"],
                "variant": entry["variant"],
                "aa_model": entry["aa_model"],
                "source": LEADERBOARD_URL,
            }
            break
    unmatched = sorted(e["aa_model"] for k, e in best.items() if k not in claimed_by)
    return scores, unmatched


HEADER = (
    "# AUTO-SCRAPED by scripts/pull_aa.py — do not hand-edit.\n"
    "# Artificial Analysis Intelligence Index (0-100 composite), by hf_repo.\n"
    "# variant records which reasoning-effort row won (the highest scoring).\n"
    "# The index is re-weighted between versions, so values are NOT comparable\n"
    "# across time. unmatched lists AA rows no tracked model claimed.\n"
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


def write_scores(path, scores, unmatched):
    Path(path).write_text(HEADER + yaml.safe_dump(
        {"scores": scores, "unmatched": unmatched},
        sort_keys=True, allow_unicode=True, width=100))


def refresh(path, html, tracked):
    """Rewrite the sidecar. Returns the score count, or None if nothing was written.

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
    scores, unmatched = match_to_tracked(best_by_slug(rows), tracked)
    write_scores(path, scores, unmatched)
    return len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="parse a saved page instead of fetching")
    args = ap.parse_args()

    html = Path(args.html).read_text() if args.html else fetch_html(LEADERBOARD_URL)
    n = refresh(OUT, html, tracked_models())
    if n is None:
        return
    print(f"Wrote {n} AA score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
