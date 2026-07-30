import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hf_meta


class FakeInfo:
    """Stand-in for huggingface_hub.ModelInfo — only the attributes we read."""

    def __init__(self, id, total=None, license=None, ctx=None,
                 created_at=None, downloads=0):
        self.id = id
        self.safetensors = {"total": total} if total is not None else None
        self.card_data = {"license": license} if license is not None else {}
        self.config = {"max_position_embeddings": ctx} if ctx is not None else {}
        self.created_at = created_at
        self.downloads = downloads


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
