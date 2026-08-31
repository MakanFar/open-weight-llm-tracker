#!/usr/bin/env python3
"""Join Artificial Analysis leaderboard entries onto tracked rows.

WHY THE JOIN IS NOT DONE AT SCRAPE TIME:
    AA publishes display names, not repo ids, so something has to resolve
    "GLM-5.3-Flash" to zai-org/GLM-5.3-Flash. pull_aa.py used to do that
    resolution as it wrote, which made aa_scores.yaml a file with three
    inputs — the scrape, models.yaml and candidates.yaml — while presenting
    itself as one. Derived data with a stale input goes stale, and this input
    was stale by construction: discover.yml scrapes BEFORE it discovers, so
    the join ran against an index that predated the week's own releases. A
    model promoted this run rendered no score for a week, which is exactly
    the release the tracker exists to surface.

    It also made a cycle. pull_aa read models.yaml/candidates.yaml; discover
    read aa_scores.yaml for the notability floor. Neither could go first.
    Resolving at the point of use breaks it: the scraper reads no repo data
    at all, so the order is simply scrape -> discover -> render, and each
    step joins against the rows it actually has.

WHAT MOVED HERE:
    The row-loading helpers and the claim guard, so pull_aa.py, discover.py,
    render_readme.py and render_json.py share exactly one implementation.
    This module deliberately imports no HTTP client: the renderers are
    diff-checked by CI and must stay hermetic.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"
AA = ROOT / "aa_scores.yaml"


def load_entries(path=AA):
    """{slug: entry} from aa_scores.yaml's `scores:` mapping.

    Returns {} on a missing, unreadable or malformed file — never raises — so
    every caller degrades to "AA rates nothing" rather than failing a render.

    The key is RE-DERIVED from each entry's aa_model rather than read back
    from the file, through the same names.split_variant / slug path that
    pull_aa.best_by_slug used to write it. The written key is a slug too, but
    a stored key is frozen against the normalisation that produced it: change
    names.slug and every key in the committed file silently stops matching,
    with no error and no diff to point at. Re-deriving keeps names.slug the
    single authority.
    """
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    if not isinstance(scores, dict):
        return {}

    entries = {}
    for entry in scores.values():
        if not isinstance(entry, dict):
            continue
        display = entry.get("aa_model")
        index = entry.get("intelligence_index")
        if not isinstance(display, str) or not display:
            continue
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        base, variant = names.split_variant(display)
        key = names.slug(names.strip_variant_suffix(base))
        if not key or key in entries:
            continue
        entries[key] = {"aa_model": display,
                        "intelligence_index": index,
                        "variant": entry.get("variant") or variant}
    return entries


def keys_for(row):
    """Slugs a tracked row may be known by on a leaderboard, in try-order.

    `name` before the repo tail: models.yaml's name is the curated,
    human-facing label, and the repo tail is the fallback for when AA's
    display name does not match it (AA's "Some Model" against repo
    zai-org/GLM-5.2). The order must be explicit and a set cannot express it
    — iterating a set is hash-randomized, so if the two ever resolved to
    different AA entries, which one won would vary between runs on identical
    inputs.
    """
    if not isinstance(row, dict):
        return []
    keys = []
    for candidate in (row.get("name"), (row.get("hf_repo") or "").split("/")[-1]):
        if not candidate:
            continue
        key = names.slug(names.strip_variant_suffix(str(candidate)))
        if key and key not in keys:
            keys.append(key)
    return keys


def _entry_identity(display):
    """The dated-snapshot-insensitive key for an AA display name.

    names.repo_identity drops trailing date, precision and variant tokens; it
    wants a repo id, and a display name is the same thing with spaces for
    separators, so it is spelled as one.
    """
    base, _ = names.split_variant(display)
    return names.repo_identity(base.replace(" ", "-"))


def _identity_index(entries):
    """{identity: entry_key}, ambiguous identities dropped and reported.

    AA names the dated snapshot it measured while models.yaml names the
    release: "DeepSeek V4 Flash 0731" against deepseek-ai/DeepSeek-V4-Flash.
    Nothing in the exact keys can bridge that, and it is a real published
    score on a real tracked row, so the identity is the last resort.

    An identity claimed by two entries cannot be resolved — we would be
    guessing which snapshot the row means — so it is dropped rather than
    guessed at. A wrong number is worse than no number.
    """
    grouped = {}
    for key, entry in entries.items():
        grouped.setdefault(_entry_identity(entry["aa_model"]), []).append(key)

    index = {}
    for identity, keys in grouped.items():
        if not identity:
            continue
        if len(keys) > 1:
            print(f"  ! AA identity {identity!r} claimed by {sorted(keys)}; "
                  f"not joining on it")
            continue
        index[identity] = keys[0]
    return index


def claims(entries, rows):
    """Join AA entries onto rows. Returns ({lower_repo: entry}, claimed_keys).

    An AA entry may be claimed by at most one row, first claim wins. Two rows
    can produce overlapping keys — a near-duplicate models.yaml entry, or one
    row's name-key colliding with another row's repo-tail-key — and without
    this guard both would silently get a score pointing at the same
    measurement. The collision is printed, not swallowed: it is how
    google/gemma-4-31B and google/gemma-4-31B-it announced themselves as the
    same model before validate.py's identity check caught it.

    Keys are tried in a fixed order — the row's name, its repo tail, then its
    repo identity — and the result is keyed by the row's OWN hf_repo. That is
    why callers need no fallback of their own: the old sidecar stored
    whichever repo string the scrape-time join landed on, and render_readme
    reconciled the two spellings afterwards. The reconciliation still has to
    happen, but it belongs here, against the AA display name that actually
    differs (see _identity_index), not against a repo string invented one
    layer up.
    """
    identities = _identity_index(entries)
    joined, claimed_by = {}, {}
    for row in rows:
        repo = (row.get("hf_repo") or "") if isinstance(row, dict) else ""
        if not repo:
            continue
        keys = keys_for(row)
        # Last resort, after both exact keys: see _identity_index.
        fallback = identities.get(names.repo_identity(repo))
        if fallback and fallback not in keys:
            keys.append(fallback)
        for key in keys:
            entry = entries.get(key)
            if entry is None:
                continue
            prior = claimed_by.get(key)
            if prior is not None:
                print(f"  ! {repo} and {prior} both match AA "
                      f"model {entry['aa_model']!r}; keeping {prior}")
                break
            claimed_by[key] = repo
            joined[repo.lower()] = entry
            break
    return joined, claimed_by


def join(entries, rows):
    """{lower_repo: entry} for callers with no use for the claimed keys."""
    return claims(entries, rows)[0]


def unclaimed(entries, claimed_keys):
    """AA display names no row claimed, sorted. Takes the keys claims() built.

    Expected to be long — most AA rows are proprietary models this tracker
    will never carry — so it is informational, never an error. Unlike the
    `unmatched` list the old sidecar froze at scrape time, this is computed
    against the rows as they stand right now, so it is a true coverage gap
    rather than a week-old one.
    """
    return sorted(entry["aa_model"] for key, entry in entries.items()
                  if key not in claimed_keys)


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


def union(primary, secondary):
    """primary + secondary, deduplicated on lowercased hf_repo, primary first.

    A row mid-promotion briefly appears in both files, and joining it twice
    would trip the claim guard in claims() and drop the score entirely.
    """
    rows, seen = [], set()
    for row in list(primary) + list(secondary):
        key = row["hf_repo"].lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def joinable_models(data_path=DATA, candidates_path=CANDIDATES):
    """Every row worth joining: published models plus the review queue.

    Used where the question is coverage — which AA rows nothing accounts for
    — and by discover.py, which stamps the index onto candidates so a
    reviewer sees the score while deciding whether to take a model. The
    RENDERERS deliberately do not use this: a render must be a function of
    models.yaml and the scrape alone, or a staged candidate could claim an
    entry out from under the published row that should carry it.
    """
    return union(_rows_with_repo(data_path), _rows_with_repo(candidates_path))
