import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pull_arena


def fake_search(index):
    """Build a search_fn backed by a dict of {author_or_None: [repo_ids]}."""
    def _search(query, author):
        return index.get(author, index.get(None, []))
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


def test_search_failure_is_not_fatal():
    def boom(query, author):
        raise RuntimeError("HF rate limited")

    row = {"model": "GLM 5.2 Z.ai · MIT", "org": "Zhipu AI",
           "matched_keyword": "glm"}

    out = pull_arena.resolve_row(row, boom)

    assert out["resolved_repo"] is None
    assert out["open_weight"] is False
    assert out["needs_hf_repo"] is True
