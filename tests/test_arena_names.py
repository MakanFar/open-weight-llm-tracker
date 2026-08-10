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


# --- new text-leaderboard trailing effort token -----------------------------
# The Agent board wrote reasoning effort in parentheses ("GLM 5.2 (Max)"),
# already handled above. The text leaderboard's open-source view instead
# glues it onto the slug with a hyphen ("glm-5.2-max"), which the old code
# never stripped, so the two highest-ranked open-weight rows on the new board
# (kimi-k3-max, glm-5.2-max) failed to resolve. See names.EFFORT_TOKENS.

@pytest.mark.parametrize("token", ["max", "high", "medium", "low", "minimal", "xhigh"])
def test_trailing_effort_token_is_stripped(token):
    """Each of the six inference-time reasoning-effort settings must be
    stripped when hyphen-glued onto the slug, the same way the old board's
    parenthetical form already was.
    """
    assert pull_arena.normalize_model_name(f"glm-5.2-{token}") == "glm-5.2"


def test_thinking_suffix_is_not_stripped():
    """moonshotai/Kimi-K2-Thinking is a real, separate repo from the
    non-thinking weights. Stripping 'thinking' like an effort token would
    make 'kimi-k2.5-thinking' resolve to Kimi-K2.5 -- the WRONG weights --
    instead of correctly staying unresolved. See test_arena_resolve.py for
    the score_match side of this guard.
    """
    assert pull_arena.normalize_model_name("kimi-k2.5-thinking") \
        == "kimi-k2.5-thinking"


def test_preview_suffix_is_not_stripped():
    """'preview' can denote a genuinely distinct release --
    deepseek-v4-pro-high-preview is on the board -- so it must survive even
    though it follows an effort word ('high'). Stripping only the true
    trailing token (never a word in the middle) is what keeps this row
    correctly unresolved rather than guessed down to a different repo.
    """
    assert pull_arena.normalize_model_name("deepseek-v4-pro-high-preview") \
        == "deepseek-v4-pro-high-preview"


def test_space_separated_effort_word_is_a_real_product_name_not_a_tag():
    """Alibaba's Qwen-Max is a distinct, real proprietary tier (vs
    Qwen-Plus/Qwen-Turbo) that legitimately ends in the word 'Max' -- it is
    not a reasoning-effort knob, and no HF repo will ever resolve for it
    either way. The old board renders it as a separate, space-delimited word
    ("Qwen3.7 Max Alibaba"), never hyphen-glued onto the slug the way the new
    board writes effort tags. So the strip must key off the hyphen/underscore
    attachment and leave a plain space-separated trailing word alone --
    already covered by the "Qwen3.7 Max" row in test_normalize_model_name
    above; this test names the reasoning it protects.
    """
    assert pull_arena.normalize_model_name(
        "Qwen3.7 Max Alibaba · Proprietary") == "Qwen3.7 Max"


def test_old_parenthetical_effort_tag_still_works():
    """The trailing-token strip added for the new board is additive: it must
    not change how the old board's parenthetical effort tag was already
    handled.
    """
    assert pull_arena.normalize_model_name(
        "GLM 5.2 (Max) Z.ai · MIT · SiliconFlow") == "GLM 5.2"


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


@pytest.mark.parametrize("query,repo_id", [
    ("Qwen3.7 Max", "Qwen/Qwen3"),      # proprietary; "Qwen3" is only a brand prefix
    ("Grok 4.5", "someone/grok"),
    ("GPT 5.6 Sol", "foo/gpt"),
    ("Claude Fable 5", "bar/claude"),
])
def test_score_match_rejects_brand_prefix_only(query, repo_id):
    """A prefix relation between very different-length names is not a match.

    Every one of these scored "medium" under the old prefix-only rule, and
    "medium" is above the open-weight threshold in resolve_row — so each of
    these proprietary models was minted open_weight: true.
    """
    assert pull_arena.score_match(query, repo_id) == "low"
