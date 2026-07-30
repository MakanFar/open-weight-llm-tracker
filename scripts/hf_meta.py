#!/usr/bin/env python3
"""
Shared Hugging Face metadata helpers.

Both discovery sources (the org sweep in discover.py and arena-resolved repos
in pull_arena.py) build candidate rows through this module, so a candidate has
exactly one construction path and one set of filter rules.
"""
import re
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
    r"(gguf|awq|gptq|-int4|-int8|-fp8|mxfp8|nvfp4|w4a16|w8a8|-qat-|-bnb|-mlx|"
    r"-onnx|lora|adapter|draft|eagle3|dflash|-embed|reranker|"
    r"-4bit|-8bit|quantized|merge)",
    re.IGNORECASE,
)

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions")
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
    cfg = getattr(info, "config", None) or {}
    if isinstance(cfg, dict):
        for k in CTX_KEYS:
            if isinstance(cfg.get(k), int):
                return cfg[k]
        # some configs nest under "text_config" / "llm_config"
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
                        needs_hf_repo=None, resolution_confidence=None):
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
