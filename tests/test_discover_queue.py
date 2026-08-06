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


# downloads=600_000 clears classify.NOTABILITY_DOWNLOADS so these fixtures are
# notable on their own — independent of whatever aa_scores.yaml happens to
# contain — which is what keeps this file's carry-forward tests from silently
# depending on production data.
GLM = FakeInfo("zai-org/GLM-5.2", total=753_300_000_000, license="mit",
               created_at="2026-07-01T00:00:00+00:00", downloads=600_000)
KIMI = FakeInfo("moonshotai/Kimi-K3", total=1_058_600_000_000, license="mit",
                created_at="2026-07-20T00:00:00+00:00", downloads=600_000)

TODAY = date(2026, 7, 30)


def _refresh(api, tmp_path, **kw):
    # aa_path defaults to a sidecar that does not exist, not the real
    # aa_scores.yaml — these tests are about carry-forward mechanics, not
    # about whatever AA currently rates zai-org/GLM-5.2.
    kw.setdefault("aa_path", tmp_path / "nope.yaml")
    kw.setdefault("get_text", lambda url: "")
    return discover.refresh(
        api, 3.0,
        data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        use_arena=False, get_json=lambda url: {}, today=TODAY, **kw)


def _seed(tmp_path, models="models: []\n", candidates="models: []\n"):
    (tmp_path / "models.yaml").write_text(models)
    (tmp_path / "candidates.yaml").write_text(candidates)


def test_second_run_preserves_staged_candidates(tmp_path):
    """The regression: run twice, the queue must not empty itself.

    GLM has no context_window in this fixture (the fake get_json/get_text
    fetchers answer nothing), so it is notable-but-incomplete and lands in
    the review queue rather than promoting — that incompleteness is
    incidental to this test, which only cares that the row survives.
    """
    _seed(tmp_path)
    api = FakeApi({"zai-org": [GLM]})

    promoted, first_queue, _, _ = _refresh(api, tmp_path)
    assert promoted == []
    assert [c["hf_repo"] for c in first_queue] == ["zai-org/GLM-5.2"]

    discover.write_candidates(tmp_path / "candidates.yaml", first_queue)

    _, second_queue, _, _ = _refresh(api, tmp_path)
    assert [c["hf_repo"] for c in second_queue] == ["zai-org/GLM-5.2"]


def test_second_run_appends_new_findings_to_the_queue(tmp_path):
    _seed(tmp_path)
    api = FakeApi({"zai-org": [GLM]})

    _, first_queue, _, _ = _refresh(api, tmp_path)
    discover.write_candidates(tmp_path / "candidates.yaml", first_queue)

    api = FakeApi({"zai-org": [GLM], "moonshotai": [KIMI]})
    _, second_queue, _, _ = _refresh(api, tmp_path,
                                     orgs=["zai-org", "moonshotai"])

    assert sorted(c["hf_repo"] for c in second_queue) == [
        "moonshotai/Kimi-K3", "zai-org/GLM-5.2"]


def test_promoted_candidates_leave_the_queue(tmp_path):
    """Once a row lands in models.yaml it must not be re-staged or carried."""
    _seed(tmp_path,
          models="models:\n  - name: GLM-5.2\n    hf_repo: zai-org/GLM-5.2\n",
          candidates=yaml.safe_dump({"models": [
              {"name": "GLM-5.2", "hf_repo": "zai-org/GLM-5.2",
               "release_date": date(2026, 7, 1)}]}))
    api = FakeApi({"zai-org": [GLM]})

    promoted, queue, _, _ = _refresh(api, tmp_path)

    assert promoted == []
    assert queue == []


def test_staged_rows_are_kept_even_when_older_than_the_age_window(tmp_path):
    """Carry-forward outranks recency: a pending review is not stale."""
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "OLMo-2-13B", "hf_repo": "allenai/OLMo-2-13B",
         "release_date": date(2023, 7, 11), "downloads": 600_000}]}))
    api = FakeApi({"allenai": []})

    _, queue, _, _ = _refresh(api, tmp_path, max_age_days=180)

    assert [c["hf_repo"] for c in queue] == ["allenai/OLMo-2-13B"]


def test_staged_row_with_unusable_release_date_survives(tmp_path):
    """candidates.yaml is edited by hand — a bad date must not kill the run.

    Carrying staged rows forward feeds human-edited YAML into the ranking
    sort, which reads release_date; a missing or non-date value there would
    otherwise take down the whole refresh. downloads is set on both rows so
    the routing loop (added in this task) does not drop them before the
    ranking sort ever runs — that gate is not what this test is about.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "NoDate", "hf_repo": "org/no-date", "downloads": 600_000},
        {"name": "StrDate", "hf_repo": "org/str-date",
         "release_date": "sometime in 2025", "downloads": 600_000},
    ]}))
    api = FakeApi({})

    _, queue, _, _ = _refresh(api, tmp_path)

    assert sorted(c["hf_repo"] for c in queue) == [
        "org/no-date", "org/str-date"]


def test_write_candidates_round_trips(tmp_path):
    path = tmp_path / "candidates.yaml"
    rows = [{"name": "GLM-5.2", "hf_repo": "zai-org/GLM-5.2",
             "release_date": date(2026, 7, 1)}]

    discover.write_candidates(path, rows)

    assert path.read_text().startswith("# AUTO-GENERATED")
    assert yaml.safe_load(path.read_text())["models"] == rows


# --- AA index merged onto candidate rows, mirroring arena_rank --------------

def test_load_aa_index_reads_the_sidecar(tmp_path):
    f = tmp_path / "aa.yaml"
    f.write_text("scores:\n  Moonshot/Kimi-K3:\n    intelligence_index: 57\n"
                 "    variant: max\n")
    assert discover.load_aa_index(f) == {"moonshot/kimi-k3": 57}


def test_load_aa_index_degrades_on_a_missing_or_broken_file(tmp_path):
    assert discover.load_aa_index(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("scores: [not, a, mapping]\n")
    assert discover.load_aa_index(bad) == {}


def test_annotate_aa_sets_the_index_on_matching_rows():
    rows = [{"hf_repo": "moonshotai/Kimi-K3"}, {"hf_repo": "org/unrated"}]
    discover.annotate_aa(rows, {"moonshotai/kimi-k3": 57})
    assert rows[0]["aa_index"] == 57
    assert "aa_index" not in rows[1], "absent, not null — matches arena_rank"


def test_annotate_aa_refreshes_a_stale_index():
    rows = [{"hf_repo": "org/m", "aa_index": 10}]
    discover.annotate_aa(rows, {"org/m": 42})
    assert rows[0]["aa_index"] == 42


def test_annotate_aa_drops_an_index_the_sidecar_no_longer_has():
    """AA delists older models; a carried-forward row must not keep a dead score."""
    rows = [{"hf_repo": "org/m", "aa_index": 10}]
    discover.annotate_aa(rows, {})
    assert "aa_index" not in rows[0]


def test_refresh_annotates_carried_forward_candidates(tmp_path):
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3",
         "release_date": date(2026, 7, 1)}]}))
    (tmp_path / "aa.yaml").write_text(
        "scores:\n  moonshotai/Kimi-K3:\n    intelligence_index: 57\n")
    api = FakeApi({})

    _, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "aa.yaml",
        use_arena=False, get_json=lambda url: {}, get_text=lambda url: "",
        today=TODAY)

    assert queue[0]["aa_index"] == 57


def test_refresh_splits_promotable_from_reviewable(tmp_path):
    """The whole point: complete+notable leaves, incomplete+notable waits,
    unremarkable never appears.

    Ready's seeded aa_index (5) deliberately disagrees with the sidecar's (40):
    annotate_aa refreshes every row from aa_scores.yaml before classify.route
    ever sees it (see annotate_aa's docstring — AA delists models, so a stale
    carried-forward score must not survive), so the row that reaches promotion
    proves the sidecar decided it, not the stale seed.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Ready", "hf_repo": "org/ready", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 70.0,
         "params_active_b": 70.0, "architecture": "dense",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 5},
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 45},
        {"name": "Noise", "hf_repo": "org/noise", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 7.0,
         "params_active_b": 7.0, "architecture": "dense",
         "context_window": 4096, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 12},
    ]}))
    (tmp_path / "aa.yaml").write_text(
        "scores:\n"
        "  org/ready:\n    intelligence_index: 40\n"
        "  org/gappy:\n    intelligence_index: 45\n")
    api = FakeApi({})

    promoted, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "aa.yaml",
        use_arena=False, get_json=lambda url: {},
        get_text=lambda url: "", today=TODAY)

    assert [r["hf_repo"] for r in promoted] == ["org/ready"]
    assert promoted[0]["aa_index"] == 40, \
        "the sidecar is the source of truth, not the stale candidates.yaml seed"
    assert [r["hf_repo"] for r in queue] == ["org/gappy"]


def test_refresh_records_why_a_queued_row_is_queued(tmp_path):
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 0, "modality": "text", "license": "mit",
         "commercial_use": True, "aa_index": 45}]}))
    (tmp_path / "aa.yaml").write_text(
        "scores:\n  org/gappy:\n    intelligence_index: 45\n")

    _, queue, _, _ = discover.refresh(
        FakeApi({}), 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "aa.yaml", use_arena=False,
        get_json=lambda url: {}, get_text=lambda url: "", today=TODAY)

    assert set(queue[0]["needs_review"]) >= {"moe-active-params-unknown",
                                             "no-context-window"}


def test_a_skipped_arena_repo_reports_the_rank_it_takes_with_it(capsys):
    """A dropped quantization also drops its leaderboard rank.

    hf_meta.EXCLUDE_PATTERNS is right to reject a quantized repo, but the rank
    is real and the model may deserve a row under its primary repo. Say so,
    rather than dropping it silently — this is how Nemotron 3 Ultra's rank 42
    vanished from the review queue.
    """
    class _Api:
        def model_info(self, repo, expand=None):
            return FakeInfo(id=repo)

    rows = [{"resolved_repo": "nvidia/Some-Model-NVFP4", "rank": 42}]
    discover.arena_candidates(_Api(), rows, min_params=0, known=set())

    out = capsys.readouterr().out
    assert "nvidia/Some-Model-NVFP4" in out
    assert "42" in out
