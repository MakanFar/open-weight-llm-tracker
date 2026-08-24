#!/usr/bin/env python3
"""
Shared Hugging Face metadata helpers.

Both discovery sources (the org sweep in discover.py and arena-resolved repos
in pull_arena.py) build candidate rows through this module, so a candidate has
exactly one construction path and one set of filter rules.
"""
import json
import os
import re
import urllib.request
from datetime import date, datetime

# Env vars that may carry a Hugging Face access token, in priority order.
# HF_TOKEN and HUGGING_FACE_HUB_TOKEN are what huggingface_hub itself reads,
# so HfApi is already authenticated when either is set. HUGGINGFACE_TOKEN is
# here because discover.yml has been passing the repo secret under that name
# since before any code read a token — it reached neither HfApi nor the
# urllib fetchers below, so the secret was doing nothing at all.
_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

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

# The boilerplate stamped onto every auto-discovered row's `notes` field.
# Named so discover.promotion_row can strip it by identity (candidate_from_repo
# is the only writer of this exact sentence) rather than duplicating the
# literal, which would silently drift the moment either string changed.
AUTO_DISCOVERY_NOTE = "Auto-discovered candidate; review before merging into models.yaml."

CTX_KEYS = ("max_position_embeddings", "max_sequence_length", "n_positions",
            "model_max_length")

# Upper bound on a believable context window, guarding the model_max_length
# key above. transformers writes model_max_length = 1e30 when the length is
# unset, and that sentinel reaches config.json as well as the
# tokenizer_config.json enrich.context_from_tokenizer already screens (see
# _SENTINEL_MAX_LENGTH there). Nothing downstream would catch it:
# validate.py only checks context_window is a positive int, so a sentinel
# would render as a 1e30-token window in the README. The bound is deliberately
# far above any real model (DeepSeek-V4 ships 1M) — it rejects sentinels and
# unit mix-ups, not ambitious context windows.
MAX_PLAUSIBLE_CTX = 64 * 1024 * 1024

# Expert-count keys that mark a mixture-of-experts model. Vendors disagree on
# the name, so we accept all the ones in circulation. A count of 1 is not a
# mixture. Active params still cannot be derived from these — routing is not
# a simple ratio — so params_active_b stays a TODO even for a detected MoE.
MOE_KEYS = ("num_local_experts", "n_routed_experts", "num_experts",
            "moe_num_experts")
NESTED_CONFIG_KEYS = ("text_config", "llm_config")
# pipeline_tag is requested explicitly: without it the API returns None for
# that field, modality_of falls through to the `tags` fallback, and every
# genuinely-text model resolves to None instead of "text" — which routes the
# entire promotable set to review as schema-invalid. The sweep FILTERS on
# pipeline_tag but that does not imply the value is returned.
EXPAND = ["safetensors", "cardData", "config", "downloads",
          "createdAt", "lastModified", "gated", "tags", "library_name",
          "pipeline_tag"]


def is_derivative(repo_id):
    """True if the repo id marks a quant/adapter/merge/non-generative variant."""
    return bool(EXCLUDE_PATTERNS.search(repo_id))


def license_of(info):
    cd = getattr(info, "card_data", None) or {}
    lic = cd.get("license") if isinstance(cd, dict) else getattr(cd, "license", None)
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return lic


# pipeline_tag values HF itself uses to mean "this model takes non-text
# input" -- checked against allenai/Molmo2-8B, Qwen/Qwen3.5-27B, and
# meta-llama/Llama-4-Scout-17B-16E-Instruct (already tracked as multimodal),
# all of which carry image-text-to-text. any-to-any, video-text-to-text and
# visual-question-answering are the same family of tag for other input
# shapes and are included on the same evidence: they describe a model that
# is not pure text-in/text-out, which is the only thing "multimodal" means
# here.
MULTIMODAL_PIPELINE_TAGS = {
    "image-text-to-text", "any-to-any", "video-text-to-text",
    "visual-question-answering",
}
TEXT_PIPELINE_TAGS = {"text-generation"}

# Fallback vocabulary for repos where pipeline_tag is silent but a tag
# already says the same thing -- allenai/Molmo2-8B duplicates its
# image-text-to-text pipeline_tag as a plain "multimodal" tag, so that
# literal is included even though it is not itself a pipeline_tag value.
MULTIMODAL_TAGS = MULTIMODAL_PIPELINE_TAGS | {"multimodal", "vision-language"}


def modality_of(info):
    """'multimodal' / 'text' / None from HF's own signals on a ModelInfo.

    pipeline_tag is checked first: it is the single most authoritative field
    HF publishes for this (allenai/Olmo-3.1-32B-Instruct and zai-org/GLM-5.2
    both carry pipeline_tag=text-generation and are correctly text;
    allenai/Molmo2-8B, Qwen/Qwen3.5-27B, and the already-tracked
    meta-llama/Llama-4-Scout-17B-16E-Instruct all carry
    pipeline_tag=image-text-to-text and are multimodal). tags is a fallback
    for the rarer repo that omits pipeline_tag but still tags itself
    "multimodal" or similar.

    None -- not a default of "text" -- is the answer when neither signal is
    present. Defaulting to "text" was the actual bug this replaces:
    candidate_from_repo used to hardcode "text" unconditionally, so a
    multimodal model could auto-promote into models.yaml (and the rendered
    README) publishing a fact nobody checked. None is honest about not
    knowing, and validate.py already rejects a modality outside its allowed
    set -- so classify.schema_errors demotes a None-modality row to review
    instead of silently asserting "text" the way the old hardcode did.
    """
    pipeline_tag = getattr(info, "pipeline_tag", None)
    if pipeline_tag in MULTIMODAL_PIPELINE_TAGS:
        return "multimodal"
    if pipeline_tag in TEXT_PIPELINE_TAGS:
        return "text"
    tags = getattr(info, "tags", None) or []
    if any(t in MULTIMODAL_TAGS for t in tags):
        return "multimodal"
    return None


def context_of(info):
    """Context length from the API expand's config field, if it carries one."""
    return _ctx_from_config(getattr(info, "config", None) or {})


_CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"


def auth_headers(user_agent):
    """Request headers, carrying a bearer token when one is configured.

    WHY THIS EXISTS: gated repos — every meta-llama/* and google/gemma-* row
    this tracker carries — serve README.md publicly but answer 401 for
    config.json and tokenizer_config.json, and the API's `config` expand
    returns no context field for them either. Ten discovered rows sat at
    context_window: 0 with no source the pipeline could legally read. A token
    belonging to an account that has accepted those licences turns each of
    them into an ordinary authoritative config.json read.

    A blank token is treated as absent. An unset GitHub secret interpolates
    to the empty string, and sending `Authorization: Bearer ` on every
    request is strictly worse than sending nothing: it can turn a public 200
    into a 401 on repos that never needed auth.

    Everything degrades to today's behaviour when no token is set — the
    gated rows simply stay in the review queue, exactly as they do now.
    """
    headers = {"User-Agent": user_agent}
    for var in _TOKEN_ENV_VARS:
        token = (os.environ.get(var) or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            break
    return headers


def _http_get_json(url):
    req = urllib.request.Request(url, headers=auth_headers("owlt-puller/1.0"))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _plausible_ctx(value):
    """True for an int that could really be a context window.

    bool is excluded because it is a subclass of int; MAX_PLAUSIBLE_CTX
    screens the 1e30 model_max_length sentinel (see that constant).
    """
    return (isinstance(value, int) and not isinstance(value, bool)
            and 0 < value <= MAX_PLAUSIBLE_CTX)


def _ctx_from_config(cfg):
    if not isinstance(cfg, dict):
        return None
    for k in CTX_KEYS:
        if _plausible_ctx(cfg.get(k)):
            return cfg[k]
    for sub in ("text_config", "llm_config"):
        inner = cfg.get(sub) or {}
        if isinstance(inner, dict):
            for k in CTX_KEYS:
                if _plausible_ctx(inner.get(k)):
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

    architecture comes back None when neither the API expand's config nor a
    fetched config.json ever yielded a usable dict — i.e. we never saw a
    config at all, as distinct from seeing one that genuinely declares no
    experts (a legitimate "dense"). fetch_config swallows every exception
    (gated repo, timeout, 404), so without this distinction a transient
    network failure was indistinguishable from a confirmed-dense config and
    both silently became "dense" — a genuinely MoE model could then promote
    with architecture=dense and params_active_b force-equal to
    params_total_b, which validate.py's dense-equality rule waves through
    cleanly since it only checks the two figures agree, not that "dense" was
    ever actually earned. validate.row_errors rejects architecture not in
    ARCH, so a None here routes the row to review instead of promoting a
    guess, with no new gate needed.
    """
    api_cfg = getattr(info, "config", None) or {}
    saw_config = isinstance(api_cfg, dict) and bool(api_cfg)
    ctx = _ctx_from_config(api_cfg)
    arch = architecture_from_config(api_cfg) if saw_config else None
    if ctx and arch == "moe":
        return ctx, arch

    cfg = fetch_config(info.id, get_json)
    if cfg is not None:
        saw_config = True
        ctx = ctx or _ctx_from_config(cfg)
        if arch != "moe":
            arch = architecture_from_config(cfg)
    return ctx or 0, (arch if saw_config else None)


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

    modality is derived here from `info` directly via modality_of, unlike
    architecture below: architecture needs resolve_facts's own config.json
    fetch to resolve, so callers must run that first and thread the result
    in as an explicit keyword. modality's signals (pipeline_tag, tags) are
    already present on `info` -- tags is part of EXPAND -- so there is
    nothing to fetch and no reason to push the computation out to callers;
    computing it here is the same shape as license_of/params_b_of below,
    which also derive a field from `info` without the caller doing it first.

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
        "modality": modality_of(info),
        "license": lic,
        "commercial_use": COMMERCIAL_GUESS.get(lic, "conditional"),
        "license_notes": "AUTO-DISCOVERED — verify license terms.",
        "weights_url": f"https://huggingface.co/{repo}",
        "downloads": getattr(info, "downloads", None),
        "discovered_via": list(discovered_via),
        "notes": AUTO_DISCOVERY_NOTE,
    }
    if arena_rank is not None:
        candidate["arena_rank"] = arena_rank
    if needs_hf_repo is not None:
        candidate["needs_hf_repo"] = needs_hf_repo
    if resolution_confidence is not None:
        candidate["resolution_confidence"] = resolution_confidence
    return candidate
