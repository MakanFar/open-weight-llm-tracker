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
CANDIDATES = ROOT / "candidates.yaml"
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


def _rows_with_repo(path):
    """A YAML file's `models:` rows that carry an hf_repo. [] if unreadable."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return []
    rows = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("hf_repo")]


def tracked_models(path=DATA):
    """models.yaml rows that carry an hf_repo."""
    return _rows_with_repo(path)


def staged_models(path=CANDIDATES):
    """candidates.yaml rows that carry an hf_repo."""
    return _rows_with_repo(path)


def joinable_models(data_path=DATA, candidates_path=CANDIDATES):
    """Every row worth scoring: published models plus the review queue.

    Staged candidates are included because the score is most useful BEFORE
    promotion, not after — a reviewer deciding whether to take a model wants
    its index in hand. Scoring only models.yaml also made the `unmatched` list
    lie: 15 of 95 entries were models already sitting in the queue, which
    drowned the entries that are genuine coverage gaps.

    models.yaml wins on a duplicate hf_repo. A row mid-promotion can briefly
    appear in both files, and scoring it twice would trip the double-claim
    guard in match_to_tracked and drop the score entirely.
    """
    rows, seen = [], set()
    for row in list(_rows_with_repo(data_path)) + list(_rows_with_repo(candidates_path)):
        key = row["hf_repo"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


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

    It carries each row's score and variant, not just its display name. The
    join is against a SNAPSHOT of models.yaml/candidates.yaml, and a model
    published after the last scrape has no row to claim it — so its score
    lands here. Keeping only the name discarded the one number needed to
    repair that later, which is why GLM-5.3-Flash's index of 57 could only be
    recovered by hitting AA again. With the score kept, rejoin() fixes it
    offline.
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
    unmatched = sorted(
        ({"aa_model": e["aa_model"],
          "intelligence_index": e["intelligence_index"],
          "variant": e["variant"]}
         for k, e in best.items() if k not in claimed_by),
        key=lambda e: e["aa_model"])
    return scores, unmatched


HEADER = (
    "# AUTO-SCRAPED by scripts/pull_aa.py — do not hand-edit.\n"
    "# Artificial Analysis Intelligence Index (0-100 composite), by hf_repo.\n"
    "# variant records which reasoning-effort row won (the highest scoring).\n"
    "# The index is re-weighted between versions, so values are NOT comparable\n"
    "# across time. unmatched lists AA rows no tracked model claimed, with\n"
    "# their scores, so --rejoin can re-run the join without re-fetching.\n"
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


def entries_from_sidecar(doc):
    """Rebuild best_by_slug's index from an already-written aa_scores.yaml.

    Both halves of the file are scraped rows that differ only in whether some
    tracked model claimed them, so both feed the index and the file becomes
    re-joinable with no network. An `unmatched` written before those entries
    carried a score is a list of bare strings; those are skipped, so a
    --rejoin against a pre-migration file re-joins only what was already
    scored. One real scrape re-populates it.

    The slug is re-derived rather than read back, so names.slug stays the
    single authority on the key: a stored one would go stale the moment that
    normalisation changed, and silently stop matching.
    """
    if not isinstance(doc, dict):
        return {}
    scored = doc.get("scores")
    rows = list(scored.values()) if isinstance(scored, dict) else []
    unclaimed = doc.get("unmatched")
    if isinstance(unclaimed, list):
        rows += unclaimed

    entries = {}
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        display = entry.get("aa_model")
        index = entry.get("intelligence_index")
        if not isinstance(display, str) or not display:
            continue
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        base, variant = split_variant(display)
        key = names.slug(names.strip_variant_suffix(base))
        if not key or key in entries:
            continue
        entries[key] = {
            "model_slug": key,
            "aa_model": display,
            "variant": entry.get("variant") or variant,
            "intelligence_index": index,
        }
    return entries


def rejoin(path, tracked):
    """Re-run the name->repo join against the current files. No network.

    The join in match_to_tracked is against whatever models.yaml and
    candidates.yaml held when the scrape ran, and discover.yml scrapes BEFORE
    it discovers — so a model that arrives this week is joined against an
    index that predates it and renders no score for a full week. That is
    exactly the release the tracker exists to surface. Re-joining costs one
    file read.

    Returns the score count, or None when nothing was written.
    """
    name = Path(path).name
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"  ! cannot read {name}: {exc}; leaving it unchanged")
        return None
    entries = entries_from_sidecar(doc)
    if not entries:
        print(f"  ! {name} holds no scored entries; leaving it unchanged")
        return None
    scores, unmatched = match_to_tracked(entries, tracked)
    write_scores(path, scores, unmatched)
    return len(scores)


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
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--html", help="parse a saved page instead of fetching")
    source.add_argument("--rejoin", action="store_true",
                        help="re-run the name->repo join against the current "
                             "models.yaml/candidates.yaml, without fetching")
    args = ap.parse_args()

    if args.rejoin:
        n = rejoin(OUT, joinable_models())
        if n is not None:
            print(f"Re-joined {n} AA score(s) in {OUT.name}")
        return

    html = Path(args.html).read_text() if args.html else fetch_html(LEADERBOARD_URL)
    n = refresh(OUT, html, joinable_models())
    if n is None:
        return
    print(f"Wrote {n} AA score(s) to {OUT.name}")


if __name__ == "__main__":
    main()
