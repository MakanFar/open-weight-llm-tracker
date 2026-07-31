"""The review queue in candidates.yaml must survive repeated discovery runs.

discover.py rewrites candidates.yaml wholesale on every run. If already-staged
rows also count as "already known" they are skipped by the sweep, produce no
replacement row, and the rewrite silently empties the queue — so merging a
candidates PR made the next run propose deleting everything it had just staged.
"""
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover
from test_discover_sweep import FakeApi
from test_hf_meta import FakeInfo


GLM = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit",
               created_at="2026-07-01T00:00:00+00:00")
KIMI = FakeInfo("moonshotai/Kimi-K3", total=1_058_600_000_000, license="mit",
                created_at="2026-07-20T00:00:00+00:00")

TODAY = date(2026, 7, 30)


def _refresh(api, tmp_path, **kw):
    return discover.refresh(
        api, 3.0,
        data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        use_arena=False, get_json=lambda url: {}, today=TODAY, **kw)


def _seed(tmp_path, models="models: []\n", candidates="models: []\n"):
    (tmp_path / "models.yaml").write_text(models)
    (tmp_path / "candidates.yaml").write_text(candidates)


def test_second_run_preserves_staged_candidates(tmp_path):
    """The regression: run twice, the queue must not empty itself."""
    _seed(tmp_path)
    api = FakeApi({"zai-org": [GLM]})

    first, _, _ = _refresh(api, tmp_path)
    assert [c["hf_repo"] for c in first] == ["zai-org/GLM-5.2"]

    discover.write_candidates(tmp_path / "candidates.yaml", first)

    second, _, _ = _refresh(api, tmp_path)
    assert [c["hf_repo"] for c in second] == ["zai-org/GLM-5.2"]


def test_second_run_appends_new_findings_to_the_queue(tmp_path):
    _seed(tmp_path)
    api = FakeApi({"zai-org": [GLM]})

    first, _, _ = _refresh(api, tmp_path)
    discover.write_candidates(tmp_path / "candidates.yaml", first)

    api = FakeApi({"zai-org": [GLM], "moonshotai": [KIMI]})
    second, _, _ = discover.refresh(
        api, 3.0,
        data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        orgs=["zai-org", "moonshotai"],
        use_arena=False, get_json=lambda url: {}, today=TODAY)

    assert sorted(c["hf_repo"] for c in second) == [
        "moonshotai/Kimi-K3", "zai-org/GLM-5.2"]


def test_promoted_candidates_leave_the_queue(tmp_path):
    """Once a row lands in models.yaml it must not be re-staged or carried."""
    _seed(tmp_path,
          models="models:\n  - name: GLM-5.2\n    hf_repo: zai-org/GLM-5.2\n",
          candidates=yaml.safe_dump({"models": [
              {"name": "GLM-5.2", "hf_repo": "zai-org/GLM-5.2",
               "release_date": date(2026, 7, 1)}]}))
    api = FakeApi({"zai-org": [GLM]})

    candidates, _, _ = _refresh(api, tmp_path)

    assert candidates == []


def test_staged_rows_are_kept_even_when_older_than_the_age_window(tmp_path):
    """Carry-forward outranks recency: a pending review is not stale."""
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "OLMo-2-13B", "hf_repo": "allenai/OLMo-2-13B",
         "release_date": date(2023, 7, 11)}]}))
    api = FakeApi({"allenai": []})

    candidates, _, _ = _refresh(api, tmp_path, max_age_days=180)

    assert [c["hf_repo"] for c in candidates] == ["allenai/OLMo-2-13B"]


def test_staged_row_with_unusable_release_date_survives(tmp_path):
    """candidates.yaml is edited by hand — a bad date must not kill the run.

    Carrying staged rows forward feeds human-edited YAML into the ranking
    sort, which reads release_date; a missing or non-date value there would
    otherwise take down the whole refresh.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "NoDate", "hf_repo": "org/no-date"},
        {"name": "StrDate", "hf_repo": "org/str-date",
         "release_date": "sometime in 2025"},
    ]}))
    api = FakeApi({})

    candidates, _, _ = _refresh(api, tmp_path)

    assert sorted(c["hf_repo"] for c in candidates) == [
        "org/no-date", "org/str-date"]


def test_write_candidates_round_trips(tmp_path):
    path = tmp_path / "candidates.yaml"
    rows = [{"name": "GLM-5.2", "hf_repo": "zai-org/GLM-5.2",
             "release_date": date(2026, 7, 1)}]

    discover.write_candidates(path, rows)

    assert path.read_text().startswith("# AUTO-GENERATED")
    assert yaml.safe_load(path.read_text())["models"] == rows
