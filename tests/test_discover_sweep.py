import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
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
        self.calls.append(author)
        if author in self.errors:
            raise RuntimeError("HF 429")
        yield from self.by_author.get(author, [])


def test_sweep_queries_each_org_once():
    api = FakeApi({"zai-org": [], "moonshotai": []})
    discover.sweep_orgs(api, ["zai-org", "moonshotai"], 3.0, set())
    assert sorted(api.calls) == ["moonshotai", "zai-org"]


def test_sweep_collects_real_models():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set())

    assert len(candidates) == 1
    assert candidates[0]["hf_repo"] == "zai-org/GLM-5.2"
    assert candidates[0]["discovered_via"] == ["org-sweep"]


def test_sweep_drops_quantizations():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
        FakeInfo("zai-org/GLM-5.2-FP8", total=753_400_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(api, ["zai-org"], 3.0, set())

    assert [c["hf_repo"] for c in candidates] == ["zai-org/GLM-5.2"]
    assert skips["derivative"] == 1


def test_sweep_skips_already_known_repos():
    api = FakeApi({"zai-org": [
        FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit"),
    ]})
    candidates, skips = discover.sweep_orgs(
        api, ["zai-org"], 3.0, {"zai-org/glm-5.2"})

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
        api, ["zai-org", "moonshotai"], 3.0, set())

    assert len(candidates) == 1
    assert skips["org_error"] == 1


def test_dead_code_is_gone():
    """The follower-probe machinery existed only for the unbounded query."""
    for name in ("get_org_overview", "is_organization", "build_query",
                 "AUTHOR_BLOCKLIST"):
        assert not hasattr(discover, name), f"{name} should have been deleted"
