#!/usr/bin/env python3
"""
Discover NEW open-weight LLMs from the Hugging Face Hub and stage them as
candidates for review. This does NOT touch models.yaml directly — it writes
new rows to candidates.yaml so a human approves them (via PR) before they land.

Design follows what the Hub `list_models` API actually supports (see NOTES):
  - one paginated query with expand[] returns params/license/context/date/etc.
    so we don't fetch each repo's config.json separately
  - num_parameters filters by size at the API (skip tiny toy models)
  - filter by pipeline_tag + library + license tags
  - sort by created_at to see what's new first

Usage:
  pip install -r requirements.txt
  python scripts/discover.py                 # write candidates.yaml
  python scripts/discover.py --limit 300 --min-params 3
  python scripts/discover.py --no-orgs-only              # include user accounts too
  HUGGINGFACE_TOKEN=hf_xxx python scripts/discover.py    # higher rate limits

NOTES / deliberate choices (the "don'ts"):
  - We do NOT exclude gated models — Llama & Gemma are gated; excluding them
    would drop the flagships. We keep them and flag for review.
  - safetensors.total is TOTAL params. For MoE we cannot infer active params
    from the API, so params_active_b is left equal to total and flagged TODO.
  - We do NOT trust card eval results as the benchmark column — left blank.
  - We drop quantizations/adapters/merges (GGUF, AWQ, GPTQ, LoRA, ...) by name.
  - license tag is uploader-supplied; commercial_use is a *guess* to be checked.
  - --orgs-only (default) keeps models from ORGANIZATION accounts only, dropping
    individual-user uploads. There is no list_models flag for this, so we probe
    /api/organizations/{author}/overview (200 = org, 404 = user) and cache it.
    Note: "org" != "reputable lab" — anyone can make a free org. ORG_ALLOWLIST
    below is always trusted; AUTHOR_BLOCKLIST is always dropped.
"""
import argparse
import inspect
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("Install deps first:  pip install -r requirements.txt")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"

# ---- configuration -------------------------------------------------------
# License tags we accept (HF's vocabulary, NOT clean SPDX). Named open-weight
# licenses are included on purpose so we don't miss Llama/Gemma/Qwen.
ACCEPTED_LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4",
    "gemma", "qwen", "deepseek",
    "cc-by-4.0", "cc-by-nc-4.0", "other",
}

# Organization namespaces we always trust (skip the org-probe for these).
ORG_ALLOWLIST = {
    "meta-llama", "Qwen", "deepseek-ai", "mistralai", "google", "microsoft",
    "CohereForAI", "CohereLabs", "ai21labs", "allenai", "nvidia", "01-ai",
    "tiiuae", "databricks", "HuggingFaceTB", "ibm-granite", "internlm",
    "THUDM", "zai-org", "moonshotai", "openai", "xai-org", "stabilityai", "MiniMaxAI",
    "XiaomiMiMo","poolside", "thinkingmachines","baidu"
}

# Author namespaces we always drop, even if they're organizations
# (e.g. quant/merge shops). Add offenders here as you spot them in PRs.
AUTHOR_BLOCKLIST = {
    "TheBloke", "bartowski", "mradermacher", "unsloth", "MaziyarPanahi",
}

# Substrings in a repo id that mark a derivative we don't want to track.
EXCLUDE_PATTERNS = re.compile(
    r"(gguf|awq|gptq|-int4|-int8|-fp8|-bnb|-mlx|-onnx|lora|adapter|"
    r"draft|distill-gguf|-4bit|-8bit|quantized|merge)",
    re.IGNORECASE,
)

# license tag -> our commercial_use guess (ALWAYS re-checked by a human)
COMMERCIAL_GUESS = {
    "apache-2.0": True, "mit": True, "bsd-3-clause": True,
    "cc-by-4.0": True,
    "cc-by-nc-4.0": False,
    "llama2": "conditional", "llama3": "conditional", "llama3.1": "conditional",
    "llama3.2": "conditional", "llama3.3": "conditional", "llama4": "conditional",
    "gemma": "conditional", "qwen": "conditional", "deepseek": True,
    "other": "conditional",
}

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions")
EXPAND = ["safetensors", "cardData", "config", "downloads",
          "createdAt", "lastModified", "gated", "tags", "library_name"]


_ORG_CACHE = {}


def get_org_overview(name, token=None):
    """Return the HF org overview payload, or None if `name` isn't an org.

    Caches definitive answers only. Transient failures (rate limits, network)
    return None without caching, so a later call can retry.
    """
    if name in _ORG_CACHE:
        return _ORG_CACHE[name]

    url = f"https://huggingface.co/api/organizations/{name}/overview"
    headers = {"User-Agent": "owlt-discover/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            _ORG_CACHE[name] = None   # definitive: it's a user
        return None                   # 403/429/5xx -> don't poison the cache
    except Exception:
        return None

    _ORG_CACHE[name] = data
    return data


def is_organization(name, token=None, min_followers=1000):
    if name in ORG_ALLOWLIST:
        return True
    data = get_org_overview(name, token)
    if data is None:
        return False
    return data.get("numFollowers", 0) >= min_followers


def existing_repos():
    repos = set()
    for path in (DATA, CANDIDATES):
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text()) or {}
        for m in doc.get("models", []) or []:
            if m.get("hf_repo"):
                repos.add(m["hf_repo"].lower())
    return repos


def license_of(info):
    cd = getattr(info, "card_data", None) or {}
    lic = cd.get("license") if isinstance(cd, dict) else getattr(cd, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return lic


def context_of(info):
    cfg = getattr(info, "config", None) or {}
    if isinstance(cfg, dict):
        for k in CTX_KEYS:
            if isinstance(cfg.get(k), int):
                return cfg[k]
        # some configs nest under "tokenizer_config" / "text_config"
        for sub in ("text_config", "llm_config"):
            inner = cfg.get(sub) or {}
            for k in CTX_KEYS:
                if isinstance(inner.get(k), int):
                    return inner[k]
    return None


def params_b_of(info):
    st = getattr(info, "safetensors", None)
    if st is None:
        return None
    total = st.get("total") if isinstance(st, dict) else getattr(st, "total", None)
    if isinstance(total, (int, float)) and total > 0:
        return round(total / 1e9, 1)
    return None


def to_date(v):
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def build_query(api, args):
    """Assemble list_models kwargs, including only ones this client supports."""
    sig = inspect.signature(api.list_models).parameters
    kwargs = dict(
        pipeline_tag="text-generation",
        sort="created_at",
        limit=args.limit,
        expand=EXPAND,
    )
    if "library" in sig:
        kwargs["library"] = "transformers"
    if "num_parameters" in sig:
        kwargs["num_parameters"] = f"min:{args.min_params}B"
    if "direction" in sig:
        kwargs["direction"] = -1
    # keep only supported kwargs (older clients differ)
    return {k: v for k, v in kwargs.items() if k in sig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200,
                    help="max models to scan from the newest end")
    ap.add_argument("--min-params", type=float, default=3.0,
                    help="minimum total params in billions")
    ap.add_argument("--orgs-only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="keep only models from organization accounts "
                         "(default on; use --no-orgs-only to include users)")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    known = existing_repos()

    kwargs = build_query(api, args)
    print(f"Querying HF list_models with: {kwargs}")
    try:
        stream = api.list_models(**kwargs)
    except TypeError as e:
        # very old client: drop expand/num_parameters and retry minimally
        print(f"  (client rejected some kwargs: {e}; retrying minimal)")
        stream = api.list_models(pipeline_tag="text-generation",
                                 sort="created_at", limit=args.limit)

    candidates, scanned = [], 0
    skipped = {"known": 0, "derivative": 0, "small": 0, "license": 0,
               "no_params": 0, "blocked": 0, "not_org": 0}
    for info in stream:
        scanned += 1
        repo = info.id
        author = repo.split("/")[0]
        if repo.lower() in known:
            skipped["known"] += 1
            continue
        if EXCLUDE_PATTERNS.search(repo):
            skipped["derivative"] += 1
            continue
        if author in AUTHOR_BLOCKLIST:
            skipped["blocked"] += 1
            continue
        if args.orgs_only and not is_organization(author, token):
            skipped["not_org"] += 1
            continue

        params = params_b_of(info)
        if params is None:
            skipped["no_params"] += 1
            continue
        if params < args.min_params:
            skipped["small"] += 1
            continue

        lic = license_of(info)
        if lic not in ACCEPTED_LICENSES:
            skipped["license"] += 1
            continue

        candidates.append({
            "name": repo.split("/")[-1],
            "hf_repo": repo,
            "developer": author,
            "release_date": to_date(getattr(info, "created_at", None)) or date.today(),
            "params_total_b": params,
            "params_active_b": params,   # TODO: set active params for MoE by hand
            "architecture": "dense",     # TODO: mark 'moe' if applicable
            "context_window": context_of(info) or 0,   # 0 => fill during review
            "modality": "text",
            "license": lic,
            "commercial_use": COMMERCIAL_GUESS.get(lic, "conditional"),
            "license_notes": "AUTO-DISCOVERED — verify license terms.",
            "benchmark": {"name": "MMLU", "score": None,
                          "source": "TODO: fill from a leaderboard"},
            "weights_url": f"https://huggingface.co/{repo}",
            "downloads": getattr(info, "downloads", None),
            "notes": "Auto-discovered candidate; review before merging into models.yaml.",
        })
        known.add(repo.lower())

    # newest first
    candidates.sort(key=lambda c: c["release_date"], reverse=True)

    header = (
        "# AUTO-GENERATED candidate models from scripts/discover.py\n"
        "# Review each entry, fix the TODO fields (active params, architecture,\n"
        "# context, benchmark, commercial_use), then move approved rows into\n"
        "# models.yaml and delete them here.\n"
    )
    CANDIDATES.write_text(header + yaml.safe_dump({"models": candidates},
                                                 sort_keys=False, allow_unicode=True,
                                                 width=100))

    print(f"\nScanned {scanned} models. Skipped: {skipped}")
    print(f"Wrote {len(candidates)} new candidate(s) to {CANDIDATES.name}")
    # exit 0 always; the Action decides whether the diff is non-empty


if __name__ == "__main__":
    main()