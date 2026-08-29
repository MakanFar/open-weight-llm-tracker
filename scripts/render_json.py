#!/usr/bin/env python3
"""
Render models.json from models.yaml.

YAML is the source of truth; JSON is what people actually fetch. The two
carry the same index, joined with the same arena and Artificial Analysis
sidecars the README table uses, in the same newest-first order — a consumer
reading both should never have to reconcile them.

  python scripts/render_json.py

Like render_readme.py this rewrites the whole file, and CI fails on any diff
between the committed models.json and a fresh render. Nothing here reads the
clock: the `generated` stamp comes from candidates.yaml, written by the
discovery run (see discover.write_candidates), so re-rendering on a later day
produces an identical file.
"""
import json
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names  # noqa: E402
import render_readme as rr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
OUT = ROOT / "models.json"

SOURCE = "https://github.com/MakanFar/open-weight-llm-tracker"
DATA_LICENSE = "CC-BY-4.0"

# The published index, in the order SCHEMA.md documents them. Anything not on
# this list is not part of the index: the discovery-only fields
# (discovered_via, needs_review, downloads, ...) exist to move a row through
# review and would be republished as fact by a blind dict copy.
FIELDS = ("name", "hf_repo", "developer", "release_date", "params_total_b",
          "params_total_stated_b", "params_active_b", "params_active_source",
          "architecture",
          "context_window", "modality", "license", "commercial_use",
          "commercial_use_verified", "commercial_use_source",
          "license_text_published", "license_notes",
          "weights_url", "notes")


def _jsonable(value):
    return value.isoformat() if isinstance(value, date) else value


def model_entry(model, aa, ranks):
    """One index row, plus the two joined third-party signals.

    arena_rank and aa_index are numbers or null, never the em dash the table
    prints — a consumer filtering on aa_index should not have to special-case
    a glyph.
    """
    out = {field: _jsonable(model[field])
           for field in FIELDS if model.get(field) is not None}
    # Always explicit, never inferred from absence. This flag is what
    # separates a licence someone read from one guessed off an HF tag — the
    # table shows it as a trailing `?` — and a consumer that treats a missing
    # key as "verified" republishes an unchecked legal claim as settled.
    out["commercial_use_verified"] = bool(model.get("commercial_use_verified"))

    repo = model.get("hf_repo") or ""
    rank = ranks["repos"].get(repo.lower())
    if rank is None and repo:
        rank = ranks["identities"].get(names.repo_identity(repo))
    if rank is None:
        for key in names.display_identity(str(model.get("name") or "")):
            rank = ranks["names"].get(key)
            if rank is not None:
                break
    out["arena_rank"] = rank

    entry = aa["repos"].get(repo.lower())
    if entry is None and repo:
        entry = aa["identities"].get(names.repo_identity(repo))
    out["aa_index"] = entry["index"] if entry else None
    return out


def build(models, aa, ranks, generated):
    models = sorted(models,
                    key=lambda m: (m.get("release_date") or date.min,
                                   m.get("params_total_b", 0)),
                    reverse=True)
    return {
        "source": SOURCE,
        "license": DATA_LICENSE,
        "generated": generated,
        "count": len(models),
        "models": [model_entry(m, aa, ranks) for m in models],
    }


def main():
    doc = yaml.safe_load(DATA.read_text())
    payload = build(doc["models"], rr.load_aa_scores(), rr.load_arena_ranks(),
                    rr.load_generated())
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Rendered models.json with {payload['count']} models.")


if __name__ == "__main__":
    main()
