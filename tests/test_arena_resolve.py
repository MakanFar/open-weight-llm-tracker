import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena


def fake_search(index):
    """Build a search_fn backed by a dict of {author_or_None: [repo_ids]}.

    The fake HONORS `query`, the way the real HF search endpoint does: a repo
    only comes back if the query text matches its name. We approximate that
    with the query's first token (slugged), which is close enough to keep the
    result sets realistic.

    This matters. An earlier version ignored `query` entirely and returned a
    canned list keyed only on author, so every resolution test was handed a
    pre-filtered, noise-free answer and could never observe a wrong repo being
    picked. Tests that build the fixture the answer wants prove nothing.
    """
    def _search(query, author):
        pool = index.get(author, index.get(None, []))
        tokens = query.split()
        key = pull_arena.slug(tokens[0]) if tokens else ""
        return [r for r in pool if key in pull_arena.slug(r.split("/")[-1])]
    return _search


def test_resolves_exact_match_high_confidence():
    row = {"model": "GLM 5.2 (Max) Z.ai · MIT · SiliconFlow",
           "org": "Zhipu AI", "matched_keyword": "glm"}
    search = fake_search({"zai-org": ["zai-org/GLM-5.2", "zai-org/GLM-5.1"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "zai-org/GLM-5.2"
    assert out["resolution_confidence"] == "high"
    assert out["open_weight"] is True
    assert out["needs_hf_repo"] is False


def test_unresolvable_model_is_not_open_weight():
    """Meta Muse Spark is proprietary — no weights repo exists."""
    row = {"model": "Meta Muse Spark 1.1 Meta · Proprietary",
           "org": "Meta", "matched_keyword": "muse"}
    search = fake_search({"meta-llama": []})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] is None
    assert out["open_weight"] is False
    assert out["resolution_confidence"] is None


def test_low_confidence_flags_for_human():
    row = {"model": "Kimi K3 Moonshot · Proprietary",
           "org": "Moonshot AI", "matched_keyword": "kimi"}
    search = fake_search({"moonshotai": ["moonshotai/Kimi-Linear-48B-A3B-Base"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolution_confidence"] == "low"
    assert out["needs_hf_repo"] is True
    assert out["open_weight"] is False


def test_medium_confidence_resolves_and_flags():
    row = {"model": "DeepSeek V4 Pro DeepSeek · MIT",
           "org": "DeepSeek", "matched_keyword": "deepseek"}
    search = fake_search({"deepseek-ai": ["deepseek-ai/DeepSeek-V4"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "deepseek-ai/DeepSeek-V4"
    assert out["resolution_confidence"] == "medium"
    assert out["open_weight"] is True
    assert out["needs_hf_repo"] is True


def test_prefers_highest_scoring_candidate():
    row = {"model": "Kimi K2.6 Moonshot · Modified MIT",
           "org": "Moonshot AI", "matched_keyword": "kimi"}
    search = fake_search({"moonshotai": [
        "moonshotai/Kimi-K2-Thinking",   # low
        "moonshotai/Kimi-K2.6",          # high, listed second
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "moonshotai/Kimi-K2.6"
    assert out["resolution_confidence"] == "high"


def test_unknown_org_searches_without_author():
    row = {"model": "Tencent Hy3 Tencent · Apache 2.0",
           "org": "Tencent", "matched_keyword": "hy3"}
    search = fake_search({None: ["tencent/Tencent-Hy3"]})

    out = pull_arena.resolve_row(row, search)

    assert out["resolved_repo"] == "tencent/Tencent-Hy3"
    assert out["open_weight"] is True


# --- proprietary models against realistic, noisy result sets -----------------
#
# Each of these used to come back open_weight: true. The search returns real
# repos that share the vendor's brand prefix, and the old prefix-only
# score_match rated "Grok 4.5" vs "grok" (etc.) as medium — good enough to
# mint open weights for a closed model.

def test_proprietary_grok_does_not_resolve_against_brand_lookalikes():
    row = {"model": "Grok 4.5 xAI · Proprietary",
           "org": "xAI", "matched_keyword": "grok"}
    search = fake_search({"xai-org": [
        "xai-org/grok",            # brand prefix of "grok45" — must not match
        "xai-org/grok-1",
        "xai-org/grok-2-mini",
        "xai-org/grok-1-hf",
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["open_weight"] is False
    assert out["resolved_repo"] is None


def test_proprietary_gpt_does_not_resolve_against_open_weight_siblings():
    """OpenAI publishes gpt-oss; that must not make GPT 5.6 Sol open-weight."""
    row = {"model": "GPT 5.6 Sol (xHigh) OpenAI · Proprietary",
           "org": "OpenAI", "matched_keyword": "gpt"}
    search = fake_search({"openai": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt2",
        "openai/whisper-large-v3",   # noise the query itself filters out
    ]})

    out = pull_arena.resolve_row(row, search)

    assert out["open_weight"] is False
    assert out["resolved_repo"] is None


def test_anthropic_never_searches_all_of_hugging_face():
    """Anthropic maps to None: no namespace, so no search at all.

    The index is keyed on None (the unscoped bucket) and stuffed with repos a
    real unscoped search for "Claude" would return. If resolve_row ever falls
    back to an unscoped search, this test sees them and fails.
    """
    calls = []

    def tracking_search(query, author):
        calls.append((query, author))
        return fake_search({None: [
            "some-user/claude",
            "some-user/claude-3-replica",
            "another/Claude-Fable-5",   # an outright impostor repo
        ]})(query, author)

    row = {"model": "Anthropic Claude Fable 5 (High) Anthropic · Proprietary",
           "org": "Anthropic", "matched_keyword": "claude"}

    out = pull_arena.resolve_row(row, tracking_search)

    assert calls == [], f"searched HF for a vendor with no namespace: {calls}"
    assert out["open_weight"] is False
    assert out["resolved_repo"] is None


def test_search_failure_is_not_fatal():
    def boom(query, author):
        raise RuntimeError("HF rate limited")

    row = {"model": "GLM 5.2 Z.ai · MIT", "org": "Zhipu AI",
           "matched_keyword": "glm"}

    out = pull_arena.resolve_row(row, boom)

    assert out["resolved_repo"] is None
    assert out["open_weight"] is False
    assert out["needs_hf_repo"] is True
