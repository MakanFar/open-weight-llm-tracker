#!/usr/bin/env python3
"""
Auto-fill model fields from the Hugging Face Hub.

For each model in models.yaml that has an `hf_repo`, this fetches:
  - license           (from repo card metadata)
  - context_window    (from config.json -> max_position_embeddings)
  - downloads         (popularity signal, written to notes if you want)
  - a suggested params figure when the repo exposes `safetensors.total`

It does NOT overwrite fields you've curated by hand unless you pass --overwrite.
By default it prints a diff of what it *would* change so you stay in control of
the benchmark column and license nuances.

Usage:
  pip install -r requirements.txt
  python scripts/pull_hf.py                 # dry-run: show suggested fills
  python scripts/pull_hf.py --write         # apply non-destructive fills
  python scripts/pull_hf.py --write --overwrite   # trust HF over local values

Note: huggingface.co must be reachable from wherever you run this. It works on a
normal machine; some sandboxes block the domain.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
API = "https://huggingface.co/api/models/{repo}"
CONFIG = "https://huggingface.co/{repo}/resolve/main/config.json"

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions")


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "owlt-puller/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_repo(repo):
    """Return a dict of fields we could derive from HF, or {} on failure."""
    out = {}
    try:
        meta = _get_json(API.format(repo=repo))
    except Exception as e:
        print(f"  ! {repo}: API error: {e}", file=sys.stderr)
        return out

    card = meta.get("cardData") or {}
    if card.get("license"):
        out["license_hf"] = card["license"]
    if isinstance(meta.get("downloads"), int):
        out["downloads"] = meta["downloads"]
    # params, when the repo publishes safetensors totals
    st = meta.get("safetensors") or {}
    if isinstance(st.get("total"), int) and st["total"] > 0:
        out["params_total_b"] = round(st["total"] / 1e9, 1)

    try:
        cfg = _get_json(CONFIG.format(repo=repo))
        for k in CTX_KEYS:
            if isinstance(cfg.get(k), int):
                out["context_window"] = cfg[k]
                break
    except Exception:
        pass  # some repos gate config.json; skip quietly
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply fills to models.yaml")
    ap.add_argument("--overwrite", action="store_true",
                    help="let HF values replace existing curated values")
    args = ap.parse_args()

    doc = yaml.safe_load(DATA.read_text())
    models = doc["models"]
    changed = 0

    for m in models:
        repo = m.get("hf_repo")
        if not repo:
            continue
        print(f"* {m['name']}  ({repo})")
        derived = fetch_repo(repo)
        if not derived:
            continue

        # license is advisory only — we surface HF's tag but never silently
        # overwrite your hand-checked commercial_use flag.
        if "license_hf" in derived:
            print(f"    HF license tag: {derived['license_hf']}  (local: {m.get('license')})")

        for field in ("context_window", "params_total_b"):
            if field not in derived:
                continue
            cur = m.get(field)
            new = derived[field]
            if cur == new:
                continue
            if cur in (None, "", 0) or args.overwrite:
                print(f"    set {field}: {cur} -> {new}")
                m[field] = new
                changed += 1
            else:
                print(f"    (keep) {field}: local={cur}  hf={new}  "
                      f"[use --overwrite to change]")

        if "downloads" in derived:
            print(f"    downloads: {derived['downloads']:,}")

    if args.write and changed:
        DATA.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                       width=100))
        print(f"\nWrote {changed} field update(s) to {DATA.name}")
    elif changed:
        print(f"\n{changed} field(s) would change. Re-run with --write to apply.")
    else:
        print("\nNothing to change.")


if __name__ == "__main__":
    main()
