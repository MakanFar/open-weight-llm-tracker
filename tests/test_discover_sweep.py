import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
from test_hf_meta import FakeInfo


class FakeApi:
    """Returns a canned model list per author; raises for authors in `errors`."""

    def __init__(self, by_author, errors=(), non_text=()):
        self.by_author = by_author
        self.errors = set(errors)
        # Authors whose repos exist but carry no text-generation tag, so a
        # filtered query returns nothing and an unfiltered one does not.
        # thinkingmachines is the real case: six repos, none tagged
        # text-generation, because Inkling is multimodal.
        self.non_text = set(non_text)
        self.calls = []

    def list_models(self, **kwargs):
        # Real HfApi.list_models is a generator function: calling it runs
        # no code, and the HTTP request (here, the simulated raise) only
        # fires once the caller starts iterating. Mirror that laziness so
        # tests fail if discover.py stops forcing the generator inside its
        # try/except.
        author = kwargs.get("author")
        self.calls.append(author)
        if author in self.errors:
            raise RuntimeError("HF 429")
        if author in self.non_text and kwargs.get("pipeline_tag"):
            return
        yield from self.by_author.get(author, [])


def test_sweep_queries_each_org_once():
    """One list_models call per org — the sweep must not paginate or retry.

    Both orgs return a model on purpose. An org that returns NOTHING is
    re-queried once, unfiltered, to tell a dead org from a live one with no
    text-generation repos (see sweep_orgs); that second call is deliberate
    and is covered by its own tests, so this one uses non-empty orgs to keep
    testing the ordinary path it was written for.
    """
    api = FakeApi({
        "zai-org": [FakeInfo("zai-org/GLM-5.2", total=753_300_000_000,
                             license="mit")],
        "moonshotai": [FakeInfo("moonshotai/Kimi-K3", total=1_000_000_000_000,
                                license="mit")],
    })
    discover.sweep_orgs(api, ["zai-org", "moonshotai"], 3.0, set(),
                        get_json=lambda url: {})
    assert sorted(api.calls) == ["moonshotai", "zai-org"]


def test_sweep_collects_real_models():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set(),
                                            get_json=lambda url: {})

    assert len(candidates) == 1
    assert candidates[0]["hf_repo"] == "zai-org/GLM-5.2"
    assert candidates[0]["discovered_via"] == ["org-sweep"]


def test_sweep_drops_quantizations():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
        FakeInfo("zai-org/GLM-5.2-FP8", total=753_400_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set(),
                                            get_json=lambda url: {})

    assert [c["hf_repo"] for c in candidates] == ["zai-org/GLM-5.2"]
    assert skips["derivative"] == 1


def test_sweep_skips_already_known_repos():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org"], 3.0, {"zai-org/glm-5.2"}, get_json=lambda url: {})

    assert candidates == []
    assert skips["known"] == 1


def test_one_failing_org_does_not_abort_sweep():
    api = FakeApi(
        by_author={"moonshotai": [
            FakeInfo("moonshotai/Kimi-K2.6", total=1_058_600_000_000, license="mit"),
        ]},
        errors=["zai-org"],
    )
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org", "moonshotai"], 3.0, set(), get_json=lambda url: {})

    assert len(candidates) == 1
    assert skips["org_error"] == 1


def test_sweep_drops_models_older_than_the_window():
    """`limit=50` per org is not a recency filter.

    An org that publishes rarely has 50-newest reaching years back, so the
    sweep must reject by age or it stages the whole back-catalogue.
    """
    api = FakeApi({"allenai": [
        FakeInfo("allenai/OLMo-2-13B", total=13_000_000_000,
                 license="apache-2.0", created_at="2023-07-11T00:00:00+00:00"),
        FakeInfo("allenai/OLMo-4-32B", total=32_000_000_000,
                 license="apache-2.0", created_at="2026-07-15T00:00:00+00:00"),
    ]})

    candidates, skips = discover.sweep_orgs(
        api, ["allenai"], 3.0, set(), get_json=lambda url: {},
        max_age_days=180, today=date(2026, 7, 30))

    assert [c["hf_repo"] for c in candidates] == ["allenai/OLMo-4-32B"]
    assert skips["stale"] == 1


def test_sweep_keeps_models_with_no_creation_date():
    """An unknown created_at cannot prove staleness — keep it for review."""
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})

    candidates, skips = discover.sweep_orgs(
        api, ["zai-org"], 3.0, set(), get_json=lambda url: {},
        max_age_days=180, today=date(2026, 7, 30))

    assert [c["hf_repo"] for c in candidates] == ["zai-org/GLM-5.2"]
    assert skips["stale"] == 0


def test_sweep_age_window_can_be_disabled():
    api = FakeApi({"allenai": [
        FakeInfo("allenai/OLMo-2-13B", total=13_000_000_000,
                 license="apache-2.0", created_at="2023-07-11T00:00:00+00:00"),
    ]})

    candidates, _ = discover.sweep_orgs(
        api, ["allenai"], 3.0, set(), get_json=lambda url: {},
        max_age_days=0, today=date(2026, 7, 30))

    assert [c["hf_repo"] for c in candidates] == ["allenai/OLMo-2-13B"]


def test_context_window_filled_from_config_json():
    api = FakeApi({"Qwen": [
        FakeInfo("Qwen/Qwen9-30B", total=30_000_000_000, license="apache-2.0"),
    ]})
    cands, _ = discover.sweep_orgs(
        api, ["Qwen"], 3.0, set(),
        get_json=lambda url: {"max_position_embeddings": 40960})
    assert cands[0]["context_window"] == 40960


class FakeRepoApi:
    """Serves model_info(repo) from a canned {repo_id: FakeInfo} map."""

    def __init__(self, by_repo):
        self.by_repo = by_repo

    def model_info(self, repo, **kwargs):
        return self.by_repo[repo]


def test_arena_candidates_propagate_verification_flag():
    """SCHEMA.md promises needs_hf_repo reaches candidates.yaml.

    Without it, a possibly-wrong "medium" match is indistinguishable from an
    exact one in the review queue.
    """
    api = FakeRepoApi({"deepseek-ai/DeepSeek-V4": FakeInfo(
        "deepseek-ai/DeepSeek-V4", total=680_000_000_000, license="mit")})
    rows = [{"rank": 24, "resolved_repo": "deepseek-ai/DeepSeek-V4",
             "needs_hf_repo": True, "resolution_confidence": "medium"}]

    out = discover.arena_candidates(api, rows, 3.0, set(), get_json=lambda url: {})

    assert len(out) == 1
    assert out[0]["arena_rank"] == 24
    assert out[0]["needs_hf_repo"] is True
    assert out[0]["resolution_confidence"] == "medium"


def test_arena_candidates_mark_exact_matches_unflagged():
    api = FakeRepoApi({"zai-org/GLM-5.2": FakeInfo(
        "zai-org/GLM-5.2", total=753_300_000_000, license="mit")})
    rows = [{"rank": 10, "resolved_repo": "zai-org/GLM-5.2",
             "needs_hf_repo": False, "resolution_confidence": "high"}]

    out = discover.arena_candidates(api, rows, 3.0, set(), get_json=lambda url: {})

    assert out[0]["needs_hf_repo"] is False
    assert out[0]["resolution_confidence"] == "high"


def test_dead_code_is_gone():
    """The follower-probe machinery existed only for the unbounded query."""
    for name in ("get_org_overview", "is_organization", "build_query",
                 "AUTHOR_BLOCKLIST"):
        assert not hasattr(discover, name), f"{name} should have been deleted"


def test_sweep_reports_an_org_that_returns_nothing(capsys):
    """An allowlist entry HF does not recognise must not fail silently.

    ORG_ALLOWLIST carried "Tencent" for a long time while HF's namespace is
    lowercase "tencent"; list_models(author="Tencent") returns an empty list
    rather than raising, so the org_error path never fired and the sweep
    reported a perfectly normal run while covering nothing. Three more
    entries -- CohereForAI, THUDM, databricks -- were dead the same way.
    An org with zero repos is always a configuration bug: allowlisted orgs
    publish models, that is why they are on the list.
    """
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org", "Tencent"], 3.0, set(), get_json=lambda url: {})

    assert skips["empty_org"] == 1
    assert "Tencent" in capsys.readouterr().out
    assert len(candidates) == 1


def test_sweep_distinguishes_a_live_org_with_no_text_generation_repos(capsys):
    """thinkingmachines publishes six repos and none are tagged
    text-generation -- Inkling is multimodal -- so the sweep's filtered query
    returns nothing while the org is very much alive. Reporting that as "dead
    or misspelled" sends a maintainer to look for a typo that is not there,
    and hides the real finding: the sweep cannot see this vendor's models at
    all.
    """
    api = FakeApi({"thinkingmachines": [
        FakeInfo("thinkingmachines/Inkling", total=1_000_000_000_000,
                 license="apache-2.0"),
    ]}, non_text=["thinkingmachines"])
    _, skips = discover.sweep_orgs(api, ["thinkingmachines"], 3.0, set(),
                                   get_json=lambda url: {})

    out = capsys.readouterr().out
    assert skips["empty_org"] == 1
    assert "no text-generation repos" in out
    assert "dead or misspelled" not in out


def test_sweep_still_calls_a_dead_org_dead(capsys):
    api = FakeApi({})
    _, skips = discover.sweep_orgs(api, ["CohereForAI"], 3.0, set(),
                                   get_json=lambda url: {})
    assert skips["empty_org"] == 1
    assert "no repos on HF" in capsys.readouterr().out


def test_sweep_does_not_report_an_org_whose_models_were_all_filtered():
    """Zero CANDIDATES is normal -- everything was known, small or a quant.
    Zero REPOS is the bug. Only the second is worth a warning."""
    api = FakeApi({"zai-org": [FakeInfo("zai-org/GLM-5.2-GGUF", license="mit")]})
    _, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set(),
                                   get_json=lambda url: {})
    assert skips["empty_org"] == 0
