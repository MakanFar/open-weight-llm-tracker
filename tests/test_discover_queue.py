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
    # aa_path and arena_path both default to sidecars that do not exist, not
    # the real aa_scores.yaml / arena_agent_rankings.yaml — these tests are
    # about carry-forward mechanics, not about whatever AA or arena currently
    # says about zai-org/GLM-5.2. arena_path matters as much as aa_path now
    # that annotate_arena_rank re-stamps ranks every run: the production file
    # ranks GLM-5.2, which is this module's main fixture, so leaving it to the
    # default would silently bind these assertions to a scraped leaderboard.
    kw.setdefault("aa_path", tmp_path / "nope.yaml")
    kw.setdefault("arena_path", tmp_path / "no-arena.yaml")
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
    """Carry-forward outranks the org-sweep age window: a pending review is
    not stale merely because sweep_orgs' max_age_days would have filtered it
    had it been freshly discovered today.

    release_date is set older than the 180-day max_age_days used below (so
    the point above is actually exercised) but still inside
    classify.NOTABILITY_DOWNLOADS_MAX_AGE_DAYS (365) of TODAY, so the row
    stays notable via downloads — that separate, newer recency gate is not
    what this test is about (see test_downloads_recency_* in
    test_classify.py for that).
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "OLMo-2-13B", "hf_repo": "allenai/OLMo-2-13B",
         "release_date": date(2026, 1, 1), "downloads": 600_000}]}))
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


def test_schema_invalid_carried_forward_row_is_demoted_not_promoted(tmp_path):
    """A hand-edited candidates.yaml row can clear every vitals check
    (classify.missing_vitals) and still fail validate.py's own rules —
    release_date: "sometime in 2025" is exactly the garbage
    test_staged_row_with_unusable_release_date_survives proves a human can
    type. missing_vitals alone would wave this row through to models.yaml,
    where render_readme's date sort then raises TypeError comparing a
    datetime.date to a str, killing the whole weekly PR. The promotion gate
    must also run validate.py's per-row checks and demote a failure to
    review — never append it — carrying the specific complaint so a human
    can see what is wrong.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "BadDate", "hf_repo": "org/bad-date", "developer": "org",
         "release_date": "sometime in 2025", "params_total_b": 70.0,
         "params_active_b": 70.0, "architecture": "dense",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 600_000}]}))
    api = FakeApi({})

    promoted, queue, _, _ = _refresh(api, tmp_path)

    assert promoted == []
    assert [c["hf_repo"] for c in queue] == ["org/bad-date"]
    assert any("schema-invalid" in r for r in queue[0]["needs_review"]), \
        queue[0]["needs_review"]


def test_schema_valid_carried_forward_row_still_promotes(tmp_path):
    """Same shape as the row above but with a real date: the new schema gate
    must not reject a genuinely clean row, only a broken one.

    release_date is within classify.NOTABILITY_DOWNLOADS_MAX_AGE_DAYS of
    TODAY so this row stays notable via downloads — the schema gate is what
    this test is about, not the (separate, newer) downloads recency gate.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "GoodDate", "hf_repo": "org/good-date", "developer": "org",
         "release_date": date(2026, 6, 1), "params_total_b": 70.0,
         "params_active_b": 70.0, "architecture": "dense",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 600_000}]}))
    api = FakeApi({})

    promoted, queue, _, _ = _refresh(api, tmp_path)

    assert [c["hf_repo"] for c in promoted] == ["org/good-date"]
    assert queue == []


# --- re-enrichment of carried-forward rows ----------------------------------
# enrich_row previously only ran inside sweep_orgs/arena_candidates, i.e. only
# when a row was newly built. A row already staged in candidates.yaml goes
# through surviving_staged() untouched, so every one of the 358 real staged
# rows got exactly one enrichment attempt, at first discovery, and never
# again — enrich.py was permanently inert against the existing queue.

def test_refresh_reenriches_a_carried_forward_row_with_unmet_vitals(tmp_path):
    """A carried-forward MoE row still missing its active-params figure must
    get a fresh enrichment attempt on every refresh(), not just at the run
    that first discovered it — a vendor can publish the activation figure on
    the card later, and the row should unblock automatically once they do.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 600_000}]}))
    api = FakeApi({})
    card = "It has ~700B parameters and ~35B activated parameters."

    promoted, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "nope.yaml",
        use_arena=False, get_json=lambda url: {},
        get_text=lambda url: card, today=TODAY)

    assert [r["hf_repo"] for r in promoted] == ["org/gappy"]
    assert promoted[0]["params_active_b"] == 35.0


def test_refresh_does_not_reenrich_a_carried_forward_row_that_is_already_complete(tmp_path):
    """Only rows classify.missing_vitals still flags should get a re-fetch
    attempt — a fully complete row must not pay for a network round trip on
    every single weekly run forever."""
    def boom(url):
        raise AssertionError(f"should not have fetched {url} for a complete row")

    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Ready", "hf_repo": "org/ready", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 70.0,
         "params_active_b": 70.0, "architecture": "dense",
         "context_window": 131072, "modality": "text", "license": "mit",
         "commercial_use": True, "downloads": 600_000}]}))
    api = FakeApi({})

    promoted, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "nope.yaml",
        use_arena=False, get_json=lambda url: {},
        get_text=boom, today=TODAY)

    assert [r["hf_repo"] for r in promoted] == ["org/ready"]


def test_refresh_reenrichment_survives_a_carried_forward_row_having_no_model_info(tmp_path):
    """Carried-forward rows have no ModelInfo (they were never fetched this
    run) — enrich_row's licence half needs one for license_string. Passing
    None must not raise; it just means the licence half is a no-op for a
    carried row, honestly reflecting that no fresher licence data exists."""
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "Gappy", "hf_repo": "org/gappy", "developer": "org",
         "release_date": date(2026, 7, 1), "params_total_b": 700.0,
         "params_active_b": 700.0, "architecture": "moe",
         "context_window": 0, "modality": "text", "license": "other",
         "commercial_use": True, "downloads": 600_000}]}))
    api = FakeApi({})

    # Must not raise even though there is no ModelInfo to read a licence from.
    promoted, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        aa_path=tmp_path / "nope.yaml",
        use_arena=False, get_json=lambda url: {"model_max_length": 65536},
        get_text=lambda url: "", today=TODAY)

    assert queue[0]["context_window"] == 65536
    assert queue[0]["license"] == "other"  # licence half is a no-op for carried rows


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


# --- arena rank re-annotated on every run, mirroring aa_index ---------------
#
# arena_rank has the exact same freeze problem aa_index had: a staged row is
# in `known`, so sweep_orgs/arena_candidates skip it and merge_candidates can
# only merge arena fields onto a row appearing in BOTH lists -- impossible for
# a row that is only carried forward. Without re-annotation the rank a row
# was first staged with is the rank it keeps forever, even after the model
# falls off the leaderboard entirely.

def _arena_yaml(rows):
    return yaml.safe_dump({"arena_agent": rows})


def test_load_arena_index_reads_the_sidecar(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(_arena_yaml([
        {"rank": 2, "resolved_repo": "zai-org/GLM-5.2"}]))
    assert discover.load_arena_index(f) == {"zai-org/glm-5.2": 2}


def test_load_arena_index_ignores_rows_with_no_resolved_repo(tmp_path):
    f = tmp_path / "arena.yaml"
    f.write_text(_arena_yaml([
        {"rank": 1, "resolved_repo": None},
        {"rank": 2, "model": "some-unresolved-model"}]))
    assert discover.load_arena_index(f) == {}


def test_load_arena_index_ignores_non_int_ranks(tmp_path):
    """discover.load_arena does not type-check rank; this loader must,
    because a rank is compared/sorted, not just displayed -- matching
    render_readme.load_arena_ranks_from_rows's isinstance(rank, int) check.
    """
    f = tmp_path / "arena.yaml"
    f.write_text(_arena_yaml([
        {"rank": "n/a", "resolved_repo": "org/unranked"}]))
    assert discover.load_arena_index(f) == {}


def test_load_arena_index_keeps_the_best_rank_across_reasoning_efforts(tmp_path):
    """The same model can appear at several reasoning efforts (e.g. -max,
    -thinking); the row that matters for notability/promotion is the best
    (numerically lowest) one it achieved.
    """
    f = tmp_path / "arena.yaml"
    f.write_text(_arena_yaml([
        {"rank": 12, "resolved_repo": "zai-org/GLM-5.2"},
        {"rank": 2, "resolved_repo": "zai-org/GLM-5.2"},
        {"rank": 5, "resolved_repo": "zai-org/GLM-5.2"}]))
    assert discover.load_arena_index(f) == {"zai-org/glm-5.2": 2}


def test_load_arena_index_degrades_on_a_missing_or_broken_file(tmp_path):
    assert discover.load_arena_index(tmp_path / "nope.yaml") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("arena_agent: [not, a, mapping]\n")
    assert discover.load_arena_index(bad) == {}


def test_annotate_arena_rank_sets_the_rank_on_matching_rows():
    rows = [{"hf_repo": "zai-org/GLM-5.2"}, {"hf_repo": "org/unranked"}]
    discover.annotate_arena_rank(rows, {"zai-org/glm-5.2": 2})
    assert rows[0]["arena_rank"] == 2
    assert "arena_rank" not in rows[1], "absent, not null"


def test_annotate_arena_rank_refreshes_a_changed_rank():
    """The GLM-5.2 case from the switch to the text leaderboard: staged at
    12, now actually 2.
    """
    rows = [{"hf_repo": "zai-org/GLM-5.2", "arena_rank": 12}]
    discover.annotate_arena_rank(rows, {"zai-org/glm-5.2": 2})
    assert rows[0]["arena_rank"] == 2


def test_annotate_arena_rank_drops_a_rank_the_board_no_longer_lists():
    """The Kimi-K2.7-Code case: staged at rank 23, then the model fell off
    the leaderboard entirely. arena_rank is not just informational -- both
    classify.is_notable and the promotion-floor check treat
    `arena_rank is not None` as sufficient on its own, so a row that keeps a
    dead rank keeps being treated as ranked (and promotable) indefinitely.
    The field must be removed, not left stale.
    """
    rows = [{"hf_repo": "moonshotai/Kimi-K2.7-Code", "arena_rank": 23}]
    discover.annotate_arena_rank(rows, {})
    assert "arena_rank" not in rows[0]


def test_refresh_corrects_a_carried_forward_stale_arena_rank(tmp_path):
    """End-to-end: a row staged from a previous run's leaderboard snapshot
    must come out of refresh() with the CURRENT rank, not the one it was
    first staged with -- the file's rank changed underneath it and no other
    code path in refresh() ever touches a carried-forward row's arena_rank.
    """
    _seed(tmp_path, candidates=yaml.safe_dump({"models": [
        {"name": "GLM-5.2", "hf_repo": "zai-org/GLM-5.2",
         "release_date": date(2026, 7, 1), "arena_rank": 12}]}))
    (tmp_path / "arena.yaml").write_text(_arena_yaml([
        {"rank": 2, "resolved_repo": "zai-org/GLM-5.2"}]))
    api = FakeApi({})

    _, queue, _, _ = discover.refresh(
        api, 3.0, data_path=tmp_path / "models.yaml",
        candidates_path=tmp_path / "candidates.yaml",
        arena_path=tmp_path / "arena.yaml", aa_path=tmp_path / "nope.yaml",
        use_arena=False, get_json=lambda url: {}, get_text=lambda url: "",
        today=TODAY)

    assert queue[0]["arena_rank"] == 2


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


# --- reviewed family collisions -------------------------------------------

TRACKED_GLM51 = """\
models:
  - name: GLM-5.1
    hf_repo: zai-org/GLM-5.1
    developer: zai-org
    release_date: 2026-03-01
    params_total_b: 744
    params_active_b: 32
    architecture: moe
    context_window: 1048576
    modality: text
    license: mit
    commercial_use: true
"""

# Complete, notable, schema-clean. Its ONLY blocker is that family_stem
# collapses it onto the tracked GLM-5.1 — the shape of allenai/
# Olmo-3.1-32B-Think, the real row that motivated the marker.
STAGED_GLM52 = """\
models:
  - name: GLM-5.2
    hf_repo: zai-org/GLM-5.2
    developer: zai-org
    release_date: 2026-07-01
    params_total_b: 753.3
    params_active_b: 32.0
    architecture: moe
    context_window: 1048576
    modality: text
    license: mit
    commercial_use: true
    discovered_via: [org-sweep]
    downloads: 600000
{marker}"""


def test_collision_row_without_the_marker_stays_queued_forever(tmp_path):
    """The defect the marker exists to fix, pinned as a test.

    candidates.yaml is rebuilt each run and models.yaml is append-only, so
    without a way to record the reviewer's decision this row regenerates
    identically on every run with nothing a human can do about it.
    """
    _seed(tmp_path, models=TRACKED_GLM51,
          candidates=STAGED_GLM52.format(marker=""))
    promoted, queue, _, _ = _refresh(FakeApi({}), tmp_path)

    assert promoted == []
    assert [c["hf_repo"] for c in queue] == ["zai-org/GLM-5.2"]
    assert queue[0]["needs_review"] == ["family-already-tracked"]


def test_reviewed_collision_promotes_and_leaves_the_queue(tmp_path):
    """A human sets the marker; the next run publishes the row and it is gone.

    This is the whole point of the field — it must work end to end through
    carry-forward, not just in missing_vitals.
    """
    _seed(tmp_path, models=TRACKED_GLM51,
          candidates=STAGED_GLM52.format(
              marker="    family_collision_reviewed: true\n"))
    promoted, queue, _, _ = _refresh(FakeApi({}), tmp_path)

    assert [c["hf_repo"] for c in promoted] == ["zai-org/GLM-5.2"]
    assert queue == []


def test_promoted_reviewed_collision_carries_no_marker_into_models_yaml(tmp_path):
    """The marker answers a candidates.yaml question and must not be published."""
    _seed(tmp_path, models=TRACKED_GLM51,
          candidates=STAGED_GLM52.format(
              marker="    family_collision_reviewed: true\n"))
    promoted, _, _, _ = _refresh(FakeApi({}), tmp_path)

    discover.append_models(tmp_path / "models.yaml",
                           [discover.promotion_row(r) for r in promoted])
    rows = yaml.safe_load((tmp_path / "models.yaml").read_text())["models"]
    published = [r for r in rows if r["hf_repo"] == "zai-org/GLM-5.2"]
    assert published and "family_collision_reviewed" not in published[0]
