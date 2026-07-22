import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena


@pytest.mark.parametrize("display,expected", [
    ("Anthropic Claude Fable 5 (High) Anthropic · Proprietary",
     "Anthropic Claude Fable 5"),
    ("GLM 5.2 (Max) Z.ai · MIT · SiliconFlow", "GLM 5.2"),
    ("GLM 5.1 Z.ai · MIT · SiliconFlow", "GLM 5.1"),
    ("Kimi K3 Moonshot · Proprietary", "Kimi K3"),
    ("Kimi K2.7 Code Moonshot · Modified MIT", "Kimi K2.7 Code"),
    ("DeepSeek V4 Pro DeepSeek · MIT", "DeepSeek V4 Pro"),
    ("Minimax M3 MiniMax · MiniMax Community License", "Minimax M3"),
    ("Nemotron 3 Ultra Nvidia · OpenMDW-1.1", "Nemotron 3 Ultra"),
    ("Gemma 4 31B Google · Apache 2.0", "Gemma 4 31B"),
    ("Qwen3.7 Max Alibaba · Proprietary", "Qwen3.7 Max"),
    ("Meta Muse Spark 1.1 Meta · Proprietary", "Meta Muse Spark 1.1"),
    ("Tencent Hy3 Tencent · Apache 2.0", "Tencent Hy3"),
    ("Mimo V2.5 Pro Xiaomi · MIT", "Mimo V2.5 Pro"),
    ("Thinking Machines Inkling Thinky · Apache 2.0",
     "Thinking Machines Inkling"),
])
def test_normalize_model_name(display, expected):
    assert pull_arena.normalize_model_name(display) == expected


def test_derive_org_returns_two_tuple():
    org, kw = pull_arena.derive_org("Kimi K3")
    assert org == "Moonshot AI"
    assert kw == "kimi"


def test_derive_org_unknown():
    org, kw = pull_arena.derive_org("Some Unheard Of Model")
    assert org is None
    assert kw is None


@pytest.mark.parametrize("query,repo_id", [
    ("GLM 5.2", "zai-org/GLM-5.2"),
    ("Kimi K2.7 Code", "moonshotai/Kimi-K2.7-Code"),
    ("Minimax M3", "MiniMaxAI/MiniMax-M3"),
    ("Gemma 4 31B", "google/gemma-4-31b-it"),
    ("Kimi K2.6", "moonshotai/Kimi-K2.6"),
])
def test_score_match_high(query, repo_id):
    assert pull_arena.score_match(query, repo_id) == "high"


def test_score_match_medium_on_prefix():
    assert pull_arena.score_match("DeepSeek V4 Pro", "deepseek-ai/DeepSeek-V4") == "medium"


def test_score_match_low_on_mismatch():
    assert pull_arena.score_match("GLM 5.2", "meta-llama/Llama-3.1-8B") == "low"
