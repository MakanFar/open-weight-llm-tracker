import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import names


def test_slug_lowercases_and_drops_punctuation():
    assert names.slug("GLM-5.2 (Max)") == "glm52max"
    assert names.slug("Llama 3.3 70B") == "llama3370b"
    assert names.slug("") == ""


def test_strip_variant_suffix_handles_hyphen_and_space():
    """Repos write 'Llama-3.3-70B-Instruct'; AA writes 'Llama 3.3 70B'."""
    assert names.strip_variant_suffix("Llama-3.3-70B-Instruct") == "Llama-3.3-70B"
    assert names.strip_variant_suffix("Llama 3.3 70B Instruct") == "Llama 3.3 70B"
    assert names.strip_variant_suffix("gemma-3-27b-it") == "gemma-3-27b"
    assert names.strip_variant_suffix("Qwen2.5-7B-Chat") == "Qwen2.5-7B"


def test_strip_variant_suffix_leaves_other_names_alone():
    assert names.strip_variant_suffix("GLM-5.2") == "GLM-5.2"
    assert names.strip_variant_suffix("Kimi-K3") == "Kimi-K3"


def test_strip_variant_suffix_does_not_eat_a_word_ending_in_it():
    """'Summit' ends in 'it' but is not a variant suffix."""
    assert names.strip_variant_suffix("Model-Summit") == "Model-Summit"


# --- repo_identity: the render-time join key ---------------------------------

def test_repo_identity_keeps_size_tokens():
    """Size is identity: 405B and 8B are different models, not variants.

    This is the join key, used for an unguarded equality match across a whole
    file. strip_repo_decorations drops size (right for its pairwise ratio
    matching) and is fatally wrong here — both would key to 'llama31'.
    Mirrors tests/test_pull_aa.py::test_best_by_slug_keeps_different_sizes_apart.
    """
    assert names.repo_identity("meta-llama/Llama-3.1-405B-Instruct") == "llama31405b"
    assert names.repo_identity("meta-llama/Llama-3.1-8B-Instruct") == "llama318b"
    assert names.repo_identity("openai/gpt-oss-120b") == "gptoss120b"
    assert names.repo_identity("openai/gpt-oss-20b") == "gptoss20b"
    assert names.repo_identity("ibm-granite/granite-4.1-30b-base") == "granite4130b"
    assert names.repo_identity("ibm-granite/granite-4.1-3b-base") == "granite413b"


def test_repo_identity_drops_a_dated_snapshot_suffix():
    """The DeepSeek V4 Flash split: arena resolved the bare repo, AA the dated one."""
    assert names.repo_identity("deepseek-ai/DeepSeek-V4-Flash-0731") == "deepseekv4flash"
    assert names.repo_identity("deepseek-ai/DeepSeek-V4-Flash") == "deepseekv4flash"


def test_repo_identity_keeps_a_meaningful_suffix():
    """DSpark is a distinct model, not a decoration."""
    assert names.repo_identity(
        "deepseek-ai/DeepSeek-V4-Flash-DSpark") == "deepseekv4flashdspark"


def test_repo_identity_drops_precision_but_not_size():
    """The Nemotron case: NVFP4 and BF16 are the same weights, 550B/A55B are not."""
    assert names.repo_identity(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nemotron3ultra550ba55b"
    assert names.repo_identity(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16") == "nemotron3ultra550ba55b"


def test_repo_identity_drops_a_duplicated_vendor_prefix():
    assert names.repo_identity("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16") \
        == "nemotron3super120ba12b"


def test_repo_identity_never_empties_a_name():
    """A single-digit tail is not a date; gemma-2 must stay 'gemma2'."""
    assert names.repo_identity("google/gemma-2-27b-it") == "gemma227b"
    assert names.repo_identity("microsoft/phi-4") == "phi4"


def test_repo_identity_strips_a_two_token_date():
    assert names.repo_identity(
        "CohereForAI/c4ai-command-r-plus-08-2024") == "c4aicommandrplus"


# --- display_identity: ordered candidate keys --------------------------------

def test_display_identity_strips_leaderboard_chrome():
    assert names.display_identity("GLM 5.2 (Max) Z.ai · MIT") == ("glm52",)


def test_display_identity_offers_a_vendor_stripped_fallback():
    """Arena writes 'Tencent Hy3'; models.yaml calls it 'Hy3'.

    Both keys are returned, full name FIRST. The full name must win when it
    matches, because the vendor-stripped form is lossy — 'Mistral Small 3'
    reduces to 'small3', which could collide with another vendor's Small 3.
    """
    assert names.display_identity("Tencent Hy3 Tencent · Apache 2.0") \
        == ("tencenthy3", "hy3")
    assert names.display_identity("Thinking Machines Inkling Thinky · Apache 2.0") \
        == ("thinkingmachinesinkling", "inkling")
    assert names.display_identity("Mistral Small 3") == ("mistralsmall3", "small3")


def test_display_identity_returns_one_key_when_there_is_no_leading_vendor():
    assert names.display_identity("Kimi K3") == ("kimik3",)
    assert names.display_identity("Nemotron 3 Ultra Nvidia · OpenMDW-1.1") \
        == ("nemotron3ultra",)


def test_display_identity_drops_a_variant_suffix():
    assert names.display_identity("Llama 3.3 70B Instruct") == ("llama3370b",)


def test_precision_tokens_is_the_union_of_native_and_quant():
    assert names.PRECISION_TOKENS == names.NATIVE_FORMATS | names.QUANT_FORMATS
    assert "bf16" in names.NATIVE_FORMATS
    assert "nvfp4" in names.QUANT_FORMATS
    assert not names.NATIVE_FORMATS & names.QUANT_FORMATS
