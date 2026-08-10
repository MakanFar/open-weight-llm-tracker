import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import classify
import hf_meta
import names


class FakeInfo:
    """Stand-in for huggingface_hub.ModelInfo — only the attributes we read."""

    def __init__(self, id, total=None, license=None, ctx=None,
                 created_at=None, downloads=0, pipeline_tag=None, tags=None):
        self.id = id
        self.safetensors = {"total": total} if total is not None else None
        self.card_data = {"license": license} if license is not None else {}
        self.config = {"max_position_embeddings": ctx} if ctx is not None else {}
        self.created_at = created_at
        self.downloads = downloads
        # Default None/[] rather than "text-generation"/[] so a FakeInfo
        # built without an opinion on modality yields modality_of(info) ==
        # None (undetermined), not a silently-assumed "text" — the same
        # honesty modality_of itself insists on. Every existing call site
        # that doesn't pass these two kwargs is therefore unaffected in what
        # it was actually asserting (none of them read modality before this
        # change existed).
        self.pipeline_tag = pipeline_tag
        self.tags = tags if tags is not None else []


@pytest.mark.parametrize("repo_id", [
    "zai-org/GLM-5.2-FP8",
    "MiniMaxAI/MiniMax-M3-MXFP8",
    "google/gemma-4-12B-it-qat-w4a16-ct",
    "google/gemma-4-12B-it-qat-q4_0-gguf",
    "deepseek-ai/eagle3_qwen3_8b_ttt7",
    "deepseek-ai/dflash_gemma4_12b_block7",
    "nvidia/Nemotron-3-Embed-8B-BF16",
    "TheBloke/Llama-2-70B-AWQ",
])
def test_derivatives_are_excluded(repo_id):
    assert hf_meta.is_derivative(repo_id) is True


@pytest.mark.parametrize("repo_id", [
    "zai-org/GLM-5.2",
    "MiniMaxAI/MiniMax-M3",
    "moonshotai/Kimi-K2.6",
    "moonshotai/Kimi-K2.7-Code",
    "google/gemma-4-31b-it",
    "deepseek-ai/DeepSeek-V4",
])
def test_real_models_are_kept(repo_id):
    assert hf_meta.is_derivative(repo_id) is False


def test_should_track_rejects_small_model():
    info = FakeInfo("org/tiny-1b", total=1_000_000_000, license="apache-2.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "small"


def test_should_track_rejects_zero_param_artifact():
    info = FakeInfo("google/tabfm-1.0.0-jax", total=None, license="apache-2.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "no_params"


def test_should_track_rejects_unaccepted_license():
    info = FakeInfo("org/model-8b", total=8_000_000_000, license="cc-by-nc-nd-4.0")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is False
    assert reason == "license"


def test_should_track_accepts_real_model():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    keep, reason = hf_meta.should_track(info, min_params=3.0)
    assert keep is True
    assert reason is None


def test_candidate_from_repo_shape():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit",
                    ctx=131072, created_at="2026-06-16T07:39:20+00:00",
                    downloads=42)
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])

    assert c["name"] == "GLM-5.2"
    assert c["hf_repo"] == "zai-org/GLM-5.2"
    assert c["developer"] == "zai-org"
    assert c["release_date"] == date(2026, 6, 16)
    assert c["params_total_b"] == 753.3
    assert c["params_active_b"] == 753.3
    assert c["context_window"] == 131072
    assert c["license"] == "mit"
    assert c["commercial_use"] is True
    assert c["discovered_via"] == ["org-sweep"]
    assert "arena_rank" not in c
    assert "needs_hf_repo" not in c
    assert "resolution_confidence" not in c


def test_candidate_from_repo_has_no_benchmark_field():
    """models.yaml has no benchmark field; the AA Index is joined at render
    time from aa_scores.yaml, keyed by hf_repo, and is never stored per-row.
    A candidate row must not mint a field that doesn't exist downstream.
    """
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])
    assert "benchmark" not in c


def test_candidate_from_repo_carries_arena_rank():
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["arena"], arena_rank=10)
    assert c["arena_rank"] == 10
    assert c["discovered_via"] == ["arena"]


def test_candidate_from_repo_carries_verification_flag():
    """An inexact arena match must reach the review queue flagged as such."""
    info = FakeInfo("deepseek-ai/DeepSeek-V4", total=680_000_000_000,
                    license="mit")
    c = hf_meta.candidate_from_repo(
        info, discovered_via=["arena"], arena_rank=24,
        needs_hf_repo=True, resolution_confidence="medium")

    assert c["needs_hf_repo"] is True
    assert c["resolution_confidence"] == "medium"


def test_candidate_from_repo_keeps_exact_match_unflagged():
    """needs_hf_repo=False is recorded, not dropped — it means 'exact match'."""
    info = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit")
    c = hf_meta.candidate_from_repo(
        info, discovered_via=["arena"], arena_rank=10,
        needs_hf_repo=False, resolution_confidence="high")

    assert c["needs_hf_repo"] is False
    assert c["resolution_confidence"] == "high"


def test_fetch_config_returns_the_body():
    cfg = {"max_position_embeddings": 131072}
    assert hf_meta.fetch_config("org/model", get_json=lambda url: cfg) == cfg


def test_fetch_config_swallows_fetch_errors():
    def boom(url):
        raise RuntimeError("gated repo")
    assert hf_meta.fetch_config("org/model", get_json=boom) is None


def test_fetch_config_rejects_a_non_mapping_body():
    assert hf_meta.fetch_config("org/model", get_json=lambda url: ["nope"]) is None


def test_resolve_facts_reads_nested_context_from_config_json():
    info = FakeInfo("org/m")
    cfg = {"text_config": {"max_sequence_length": 8192}}
    assert hf_meta.resolve_facts(info, get_json=lambda url: cfg) == (8192, "dense")


def test_resolve_facts_zero_when_config_has_no_context_key():
    info = FakeInfo("org/m")
    assert hf_meta.resolve_facts(info, get_json=lambda url: {"foo": 1}) \
        == (0, "dense")


@pytest.mark.parametrize("cfg", [
    {"n_routed_experts": 256},
    {"num_local_experts": 8},
    {"num_experts": 128},
    {"moe_num_experts": 64},
])
def test_architecture_from_config_flags_an_expert_count(cfg):
    assert hf_meta.architecture_from_config(cfg) == "moe"


def test_architecture_from_config_reads_nested_text_config():
    """Multimodal wrappers (…ForConditionalGeneration) nest the LM config."""
    assert hf_meta.architecture_from_config(
        {"text_config": {"n_routed_experts": 128}}) == "moe"


def test_architecture_from_config_ignores_a_single_expert():
    """One expert is not a mixture."""
    assert hf_meta.architecture_from_config({"num_experts": 1}) == "dense"


@pytest.mark.parametrize("cfg", [{}, None, {"num_attention_heads": 32}, "junk"])
def test_architecture_from_config_defaults_to_dense(cfg):
    assert hf_meta.architecture_from_config(cfg) == "dense"


def test_resolve_facts_prefers_api_expand():
    info = FakeInfo("org/m", ctx=4096)
    info.config = {"max_position_embeddings": 4096, "n_routed_experts": 16}
    assert hf_meta.resolve_facts(info, get_json=lambda url: {"n_positions": 999}) \
        == (4096, "moe")


def test_resolve_facts_falls_back_to_config_json():
    info = FakeInfo("org/m")  # API expand empty
    assert hf_meta.resolve_facts(
        info, get_json=lambda url: {"max_position_embeddings": 32768}) \
        == (32768, "dense")


def test_resolve_facts_detects_moe_from_config_json():
    info = FakeInfo("org/m")
    assert hf_meta.resolve_facts(
        info, get_json=lambda url: {"max_position_embeddings": 163840,
                                    "n_routed_experts": 384}) \
        == (163840, "moe")


def test_resolve_facts_zero_and_dense_when_nothing_resolves():
    info = FakeInfo("org/m")
    assert hf_meta.resolve_facts(info, get_json=lambda url: {}) == (0, "dense")


def test_resolve_facts_fetches_config_json_at_most_once():
    """Context and architecture come from the same body — not two requests."""
    calls = []

    def counting_get(url):
        calls.append(url)
        return {"max_position_embeddings": 8192, "num_local_experts": 8}

    info = FakeInfo("org/m")
    assert hf_meta.resolve_facts(info, get_json=counting_get) == (8192, "moe")
    assert len(calls) == 1


def test_resolve_facts_survives_a_failed_config_fetch():
    """The API expand already carried a usable config (ctx=2048 means
    info.config is a real non-empty dict) before the config.json fetch
    failed, so this is "config read, no experts declared" -- a legitimate
    dense, not a case of never having seen a config at all."""
    def boom(url):
        raise RuntimeError("gated repo")

    info = FakeInfo("org/m", ctx=2048)
    assert hf_meta.resolve_facts(info, get_json=boom) == (2048, "dense")


def test_resolve_facts_none_architecture_when_no_config_ever_resolves():
    """Finding 2 (pre-merge review): fetch_config swallows every exception,
    so a transient network failure (gated repo, timeout, 404) and a config
    that genuinely declares zero experts were indistinguishable -- both
    yielded 'dense'. A genuinely MoE model could then promote with
    architecture=dense and params_active_b force-equal to params_total_b,
    which validate.py's dense-equality rule then waves through cleanly.

    Here the API expand carries no config at all (no ctx= given to FakeInfo,
    so info.config is {}) AND the config.json fetch raises -- we never saw a
    config from either source, so architecture must come back None rather
    than default to a fact we don't actually know. validate.row_errors
    rejects architecture not in ARCH, so classify.schema_errors routes such
    a row to review instead of promoting a guess.
    """
    def boom(url):
        raise RuntimeError("gated repo")

    info = FakeInfo("org/m")  # no ctx => API expand's config is empty
    assert hf_meta.resolve_facts(info, get_json=boom) == (0, None)


def test_resolve_facts_dense_when_fetched_config_genuinely_has_no_experts():
    """The other half of the distinction: a config.json body WAS read (the
    fetch succeeded, even if the body is empty/minimal) and it simply has no
    expert keys. That is a confirmed dense, not an unknown, so this must
    stay 'dense' and must NOT regress to None alongside the fetch-failure
    case above."""
    info = FakeInfo("org/m")  # API expand empty
    assert hf_meta.resolve_facts(info, get_json=lambda url: {"foo": 1}) \
        == (0, "dense")


def test_candidate_from_repo_uses_explicit_architecture():
    info = FakeInfo("org/m", total=100_000_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"],
                                    architecture="moe")
    assert c["architecture"] == "moe"


def test_candidate_from_repo_defaults_architecture_to_dense():
    info = FakeInfo("org/m", total=7_000_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])
    assert c["architecture"] == "dense"


def test_candidate_from_repo_preserves_none_architecture():
    """resolve_facts can now return architecture=None (config never resolved
    at all -- see test_resolve_facts_none_architecture_when_no_config_ever_resolves)
    and both discover.py call sites pass that straight through as an explicit
    keyword argument. candidate_from_repo's `architecture="dense"` default
    must not silently paper over an explicit None with a guessed fact --
    an explicit argument, even a falsy one, has to win over the default."""
    info = FakeInfo("org/m", total=7_000_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"],
                                    architecture=None)
    assert c["architecture"] is None


def test_candidate_uses_explicit_context_window():
    info = FakeInfo("org/m")
    row = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"], context_window=65536)
    assert row["context_window"] == 65536


def test_modality_of_reads_multimodal_pipeline_tag():
    info = FakeInfo("allenai/Molmo2-8B", pipeline_tag="image-text-to-text")
    assert hf_meta.modality_of(info) == "multimodal"


def test_modality_of_reads_text_pipeline_tag():
    info = FakeInfo("allenai/Olmo-3.1-32B-Instruct", pipeline_tag="text-generation")
    assert hf_meta.modality_of(info) == "text"


def test_modality_of_falls_back_to_a_multimodal_tag():
    """Real repos usually carry pipeline_tag AND the same signal duplicated
    in tags (allenai/Molmo2-8B has both "image-text-to-text" and
    "multimodal" as tags). This exercises the case pipeline_tag is silent
    but a tag still says multimodal -- the fallback path, not the primary
    one."""
    info = FakeInfo("org/some-vlm", pipeline_tag=None, tags=["multimodal", "transformers"])
    assert hf_meta.modality_of(info) == "multimodal"


def test_modality_of_is_none_when_neither_signal_is_present():
    """No pipeline_tag and no multimodal-shaped tag is an honest "we don't
    know", not "assume text". See modality_of's docstring for why guessing
    text here is worse than leaving the row for a human."""
    info = FakeInfo("org/mystery-model", pipeline_tag=None, tags=["transformers", "safetensors"])
    assert hf_meta.modality_of(info) is None


def test_candidate_from_repo_carries_multimodal_modality():
    """The regression itself: allenai/Molmo2-8B promoted in a dry run
    asserting modality: text (candidate_from_repo hardcoded the literal)
    while Hugging Face's own metadata says pipeline_tag=image-text-to-text
    and tags include "multimodal". A row built from that info must now
    carry the true value."""
    info = FakeInfo("allenai/Molmo2-8B", total=8_000_000_000, license="apache-2.0",
                    pipeline_tag="image-text-to-text",
                    tags=["image-text-to-text", "multimodal", "transformers"])
    c = hf_meta.candidate_from_repo(info, discovered_via=["arena"])
    assert c["modality"] == "multimodal"


def test_candidate_with_undetermined_modality_is_not_defaulted_to_text():
    info = FakeInfo("org/mystery-8b", total=8_000_000_000, license="mit")
    c = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])
    assert c["modality"] is None


def test_candidate_with_undetermined_modality_routes_to_review_not_promote():
    """Confirms Finding-closing claim 3: validate.py already rejects a
    modality outside MODALITY, and classify.schema_errors reuses
    validate.row_errors, so a None modality needs no new gate -- it demotes
    the row to review by itself, the same mechanism that already catches a
    bad release_date (see test_discover_queue's
    test_schema_invalid_carried_forward_row_is_demoted_not_promoted).

    downloads=600_000 and the default release_date (today, from
    candidate_from_repo when no created_at is given) clear is_notable's
    downloads-recency floor on their own, independent of aa_index/arena_rank
    -- so this row would otherwise be a clean, unassisted PROMOTE, isolating
    modality as the only reason it lands in review.
    """
    info = FakeInfo("org/mystery-8b", total=8_000_000_000, license="mit",
                    downloads=600_000, ctx=131072)
    row = hf_meta.candidate_from_repo(info, discovered_via=["org-sweep"])
    assert row["modality"] is None
    assert classify.missing_vitals(row, set()) == []

    reasons = classify.missing_vitals(row, set())
    reasons += [f"schema-invalid: {e}" for e in classify.schema_errors(row)]
    assert any("schema-invalid" in r and "modality" in r for r in reasons), reasons
    assert classify.route(row, set()) == "review"


def test_every_quant_format_is_excluded_by_hf_meta():
    """Two overlapping vocabularies, held together by a test not by coupling.

    names.QUANT_FORMATS says "this token is a quantized re-release";
    hf_meta.EXCLUDE_PATTERNS says "reject this repo". Every quant format must
    appear in both, or pull_arena would de-prioritise a repo that discover.py
    then happily stages. Same technique as
    test_author_hints_stay_in_step_with_org_allowlist.

    Tokens are probed with all three separators repo ids actually use ("-",
    "_", "."): EXCLUDE_PATTERNS anchors several of them on a leading hyphen
    (-int4, -fp8, -4bit), but names.repo_identity splits repo ids on
    [-_.] — an underscore- or dot-separated quantized mirror
    ("org/Model_INT4", "org/Model.int4") must be excluded too, or it slips
    into the review queue and then inherits the primary repo's AA score via
    repo_identity.
    """
    missing = sorted(
        f"{sep}{t}" for t in names.QUANT_FORMATS for sep in ("-", "_", ".")
        if not hf_meta.EXCLUDE_PATTERNS.search(f"model{sep}{t}"))
    assert not missing, (
        f"names.QUANT_FORMATS entries not caught by EXCLUDE_PATTERNS: {missing}")


def test_native_formats_are_deliberately_not_excluded():
    """BF16 is a release format, not a quantization. Do not 'fix' this gap.

    NVIDIA publishes Nemotron 3 Ultra ONLY as -BF16 repos — there is no bare
    nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B. Adding bf16 to EXCLUDE_PATTERNS
    would delete the model from the tracker entirely. The token still belongs
    in names.PRECISION_TOKENS, because it IS noise when comparing two names.
    """
    for fmt in names.NATIVE_FORMATS:
        assert not hf_meta.EXCLUDE_PATTERNS.search(f"model-{fmt}"), (
            f"{fmt} is a primary release format and must stay excludable-free")
