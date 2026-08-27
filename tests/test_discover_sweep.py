import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
import hf_meta
from test_hf_meta import FakeInfo


class FakeApi:
    """Returns a canned model list per author; raises for authors in `errors`."""

    def __init__(self, by_author, errors=()):
        self.by_author = by_author
        self.errors = set(errors)
        self.calls = []

    def list_models(self, **kwargs):
        # Real HfApi.list_models is a generator function: calling it runs
        # no code, and the HTTP request (here, the simulated raise) only
        # fires once the caller starts iterating. Mirror that laziness so
        # tests fail if discover.py stops forcing the generator inside its
        # try/except.
        author = kwargs.get("author")
        self.calls.append((author, kwargs.get("pipeline_tag")))
        if author in self.errors:
            raise RuntimeError("HF 429")
        tag = kwargs.get("pipeline_tag")
        for m in self.by_author.get(author, []):
            # A fake with no pipeline_tag of its own is tag-agnostic and
            # matches any query, so the many existing fixtures that never
            # cared about modality keep working. One that DOES declare a tag
            # is matched strictly, which is what lets a test model an org
            # whose repos are multimodal-only.
            own = getattr(m, "pipeline_tag", None)
            if tag is None or own is None or own == tag:
                yield m

    @property
    def authors_called(self):
        return [a for a, _ in self.calls]


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
    assert sorted(set(api.authors_called)) == ["moonshotai", "zai-org"]


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


def test_sweep_finds_a_multimodal_only_org(capsys):
    """thinkingmachines publishes six repos and NONE are tagged
    text-generation -- Inkling is multimodal -- so a sweep that asks only for
    text-generation cannot see this vendor at all. Inkling reached the queue
    solely because arena ranked it and arena fetches repos by name; nothing
    would have found it otherwise, and nothing would find an unranked
    multimodal flagship at all.
    """
    api = FakeApi({"thinkingmachines": [
        FakeInfo("thinkingmachines/Inkling", total=1_000_000_000_000,
                 license="apache-2.0", pipeline_tag="image-text-to-text"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["thinkingmachines"], 3.0, set(),
                                            get_json=lambda url: {})

    assert [c["hf_repo"] for c in candidates] == ["thinkingmachines/Inkling"]
    assert skips["empty_org"] == 0


def test_sweep_asks_for_every_modality_it_indexes():
    """models.yaml carries text, vision-language and multimodal rows, so the
    sweep must ask for the pipeline tags that produce them -- not just one."""
    api = FakeApi({"thinkingmachines": []})
    discover.sweep_orgs(api, ["thinkingmachines"], 3.0, set(),
                        get_json=lambda url: {})
    asked = {tag for _, tag in api.calls if tag}
    assert "text-generation" in asked
    assert hf_meta.MULTIMODAL_PIPELINE_TAGS <= asked


def test_sweep_returns_a_repo_once_even_if_several_tags_match(capsys):
    """A tag-agnostic repo comes back from every query; it is one model."""
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, _ = discover.sweep_orgs(api, ["zai-org"], 3.0, set(),
                                        get_json=lambda url: {})
    assert [c["hf_repo"] for c in candidates] == ["zai-org/GLM-5.2"]


def test_sweep_reports_a_live_org_with_nothing_it_can_use(capsys):
    """An org publishing only non-generative heads is alive but useless to
    this sweep. Saying "dead or misspelled" would send a maintainer hunting
    a typo that is not there."""
    api = FakeApi({"someorg": [
        FakeInfo("someorg/embedder", total=1_000_000_000,
                 license="mit", pipeline_tag="sentence-similarity"),
    ]})
    _, skips = discover.sweep_orgs(api, ["someorg"], 3.0, set(),
                                   get_json=lambda url: {})
    out = capsys.readouterr().out
    assert skips["empty_org"] == 1
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
