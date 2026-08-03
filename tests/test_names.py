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
