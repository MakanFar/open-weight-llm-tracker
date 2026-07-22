import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover


def cand(repo, via, rank=None, released="2026-01-01"):
    y, m, d = (int(x) for x in released.split("-"))
    c = {"hf_repo": repo, "name": repo.split("/")[-1],
         "discovered_via": list(via), "release_date": date(y, m, d)}
    if rank is not None:
        c["arena_rank"] = rank
    return c


def test_merge_dedups_on_repo_case_insensitively():
    org_rows = [cand("zai-org/GLM-5.2", ["org-sweep"])]
    arena_rows = [cand("ZAI-ORG/glm-5.2", ["arena"], rank=10)]

    merged = discover.merge_candidates(org_rows, arena_rows)

    assert len(merged) == 1
    assert sorted(merged[0]["discovered_via"]) == ["arena", "org-sweep"]
    assert merged[0]["arena_rank"] == 10


def test_arena_ranked_models_sort_first():
    org_rows = [cand("org/New-Model", ["org-sweep"], released="2026-07-01")]
    arena_rows = [cand("zai-org/GLM-5.2", ["arena"], rank=10,
                       released="2026-06-16")]

    merged = discover.merge_candidates(org_rows, arena_rows)

    assert [c["hf_repo"] for c in merged] == [
        "zai-org/GLM-5.2", "org/New-Model"]


def test_ranked_models_sort_by_rank():
    arena_rows = [
        cand("a/Model-B", ["arena"], rank=27),
        cand("a/Model-A", ["arena"], rank=10),
    ]
    merged = discover.merge_candidates([], arena_rows)
    assert [c["arena_rank"] for c in merged] == [10, 27]


def test_unranked_models_sort_by_recency():
    org_rows = [
        cand("a/Older", ["org-sweep"], released="2026-01-01"),
        cand("a/Newer", ["org-sweep"], released="2026-07-01"),
    ]
    merged = discover.merge_candidates(org_rows, [])
    assert [c["name"] for c in merged] == ["Newer", "Older"]


def test_load_arena_missing_file_is_not_fatal(tmp_path):
    rows, new_orgs = discover.load_arena(tmp_path / "nope.yaml")
    assert rows == []
    assert new_orgs == []


def test_load_arena_malformed_file_is_not_fatal(tmp_path):
    bad = tmp_path / "arena.yaml"
    bad.write_text("{{{ not yaml at all")
    rows, new_orgs = discover.load_arena(bad)
    assert rows == []
    assert new_orgs == []


def test_load_arena_returns_only_resolved_rows(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(
        "arena_agent:\n"
        "- rank: 10\n"
        "  model: GLM 5.2\n"
        "  resolved_repo: zai-org/GLM-5.2\n"
        "  open_weight: true\n"
        "- rank: 18\n"
        "  model: Meta Muse Spark 1.1\n"
        "  resolved_repo: null\n"
        "  open_weight: false\n"
        "new_orgs: [Tencent, Xiaomi]\n"
    )
    rows, new_orgs = discover.load_arena(f)

    assert [r["resolved_repo"] for r in rows] == ["zai-org/GLM-5.2"]
    assert new_orgs == ["Tencent", "Xiaomi"]
