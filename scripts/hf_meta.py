#!/usr/bin/env python3
"""
Shared Hugging Face metadata helpers.

Both discovery sources (the org sweep in discover.py and arena-resolved repos
in pull_arena.py) build candidate rows through this module, so a candidate has
exactly one construction path and one set of filter rules.
"""
import json
import re
import urllib.request
from datetime import date, datetime

# License tags we accept (HF's vocabulary, NOT clean SPDX). Named open-weight
# licenses are included on purpose so we don't miss Llama/Gemma/Qwen.
ACCEPTED_LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3", "llama4",
    "gemma", "qwen", "deepseek",
    "cc-by-4.0", "cc-by-nc-4.0", "other",
}

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

# Substrings in a repo id that mark something we don't track. Covers
# quantizations, speculative-decoding drafts, and non-generative heads.
#   *-FP8 / *-MXFP8 / *-NVFP4 / *-w4a16   -> quantizations
#   eagle3_* / dflash_* / draft           -> speculative decoding artifacts
#   *-Embed / reranker                    -> not text-generation models
EXCLUDE_PATTERNS = re.compile(
    r"(gguf|awq|gptq|[-_.]int4|[-_.]int8|[-_.]fp8|mxfp8|nvfp4|w4a16|w8a8|-qat-|-bnb|-mlx|"
    r"-onnx|lora|adapter|draft|eagle3|dflash|-embed|reranker|"
    r"[-_.]4bit|[-_.]8bit|quantized|merge)",
    re.IGNORECASE,
)

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions")

# Expert-count keys that mark a mixture-of-experts model. Vendors disagree on
# the name, so we accept all the ones in circulation. A count of 1 is not a
# mixture. Active params still cannot be derived from these — routing is not
# a simple ratio — so params_active_b stays a TODO even for a detected MoE.
MOE_KEYS = ("num_local_experts", "n_routed_experts", "num_experts",
            "moe_num_experts")
NESTED_CONFIG_KEYS = ("text_config", "llm_config")
EXPAND = ["safetensors", "cardData", "config", "downloads",
          "createdAt", "lastModified", "gated", "tags", "library_name"]


def is_derivative(repo_id):
    """True if the repo id marks a quant/adapter/merge/non-generative variant."""
    return bool(EXCLUDE_PATTERNS.search(repo_id))


def license_of(info):
    cd = getattr(info, "card_data", None) or {}
    lic = cd.get("license") if isinstance(cd, dict) else getattr(cd, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return lic


def context_of(info):
    """Context length from the API expand's config field, if it carries one."""
    return _ctx_from_config(getattr(info, "config", None) or {})


_CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "owlt-puller/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _ctx_from_config(cfg):
    if not isinstance(cfg, dict):
        return None
    for k in CTX_KEYS:
        if isinstance(cfg.get(k), int):
            return cfg[k]
    for sub in ("text_config", "llm_config"):
        inner = cfg.get(sub) or {}
        if isinstance(inner, dict):
            for k in CTX_KEYS:
                if isinstance(inner.get(k), int):
                    return inner[k]
    return None


def fetch_config(repo, get_json=_http_get_json):
    """The repo's config.json as a dict. None on any failure (gated, 404, timeout)."""
    try:
        cfg = get_json(_CONFIG_URL.format(repo=repo))
    except Exception:
        return None
    return cfg if isinstance(cfg, dict) else None


def architecture_from_config(cfg):
    """'moe' when the config declares more than one expert, else 'dense'.

    'dense' is also what an unusable or silent config yields — the schema
    enum has no "unknown", and dense was the blanket default before this
    existed, so a miss is no worse than the old behaviour.
    """
    if not isinstance(cfg, dict):
        return "dense"
    for scope in (cfg, *(cfg.get(k) for k in NESTED_CONFIG_KEYS)):
        if not isinstance(scope, dict):
            continue
        for key in MOE_KEYS:
            n = scope.get(key)
            if isinstance(n, int) and not isinstance(n, bool) and n > 1:
                return "moe"
    return "dense"


def resolve_facts(info, get_json=_http_get_json):
    """(context_window, architecture), fetching config.json at most once.

    The API expand answers both fields for some repos; when it does not, one
    config.json body serves both rather than one request per field.
    """
    api_cfg = getattr(info, "config", None) or {}
    ctx = _ctx_from_config(api_cfg)
    arch = architecture_from_config(api_cfg)
    if ctx and arch == "moe":
        return ctx, arch

    cfg = fetch_config(info.id, get_json)
    if cfg is not None:
        ctx = ctx or _ctx_from_config(cfg)
        if arch != "moe":
            arch = architecture_from_config(cfg)
    return ctx or 0, arch


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


def should_track(info, min_params):
    """Return (keep, skip_reason). skip_reason is None when keep is True."""
    if is_derivative(info.id):
        return False, "derivative"
    params = params_b_of(info)
    if params is None:
        return False, "no_params"
    if params < min_params:
        return False, "small"
    if license_of(info) not in ACCEPTED_LICENSES:
        return False, "license"
    return True, None


def candidate_from_repo(info, discovered_via, arena_rank=None,
                        needs_hf_repo=None, resolution_confidence=None,
                        context_window=None, architecture="dense"):
    """Build a candidates.yaml row from an HF ModelInfo.

    Caller is responsible for having run should_track() first.

    arena_rank / needs_hf_repo / resolution_confidence are arena-only. Each is
    omitted from the row entirely when None, so org-sweep candidates carry no
    empty arena fields. needs_hf_repo=True marks a repo matched by an inexact
    name match — the reviewer must confirm it is really this model before
    promotion — and resolution_confidence records why it was flagged.
    """
    repo = info.id
    author = repo.split("/")[0]
    params = params_b_of(info)
    lic = license_of(info)

    candidate = {
        "name": repo.split("/")[-1],
        "hf_repo": repo,
        "developer": author,
        "release_date": to_date(getattr(info, "created_at", None)) or date.today(),
        "params_total_b": params,
        "params_active_b": params,   # TODO: set active params for MoE by hand
        "architecture": architecture,
        "context_window": context_window if context_window is not None
        else (context_of(info) or 0),   # 0 => fill during review
        "modality": "text",
        "license": lic,
        "commercial_use": COMMERCIAL_GUESS.get(lic, "conditional"),
        "license_notes": "AUTO-DISCOVERED — verify license terms.",
        "weights_url": f"https://huggingface.co/{repo}",
        "downloads": getattr(info, "downloads", None),
        "discovered_via": list(discovered_via),
        "notes": "Auto-discovered candidate; review before merging into models.yaml.",
    }
    if arena_rank is not None:
        candidate["arena_rank"] = arena_rank
    if needs_hf_repo is not None:
        candidate["needs_hf_repo"] = needs_hf_repo
    if resolution_confidence is not None:
        candidate["resolution_confidence"] = resolution_confidence
    return candidate
