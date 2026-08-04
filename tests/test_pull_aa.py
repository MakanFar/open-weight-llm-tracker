import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import yaml
import pull_aa

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "aa_leaderboard.html").read_text()

TRACKED = [
    {"name": "Kimi K3", "hf_repo": "moonshotai/Kimi-K3"},
    {"name": "Llama 3.3 70B Instruct", "hf_repo": "meta-llama/Llama-3.3-70B-Instruct"},
    {"name": "Some Model", "hf_repo": "zai-org/GLM-5.2"},
]


def test_parse_reads_every_scored_row():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert [r["model"] for r in rows] == [
        "Claude Opus 5 (max)", "Kimi K3 (max)", "Kimi K3 (low)",
        "GLM-5.2", "Llama 3.3 70B",
    ]


def test_parse_reads_creator_and_index():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    kimi = next(r for r in rows if r["model"] == "Kimi K3 (max)")
    assert kimi["creator"] == "Moonshot AI"
    assert kimi["intelligence_index"] == 57


def test_parse_drops_rows_with_no_numeric_index():
    """Header rows and unrated models both fail the integer check."""
    rows = pull_aa.parse_leaderboard(FIXTURE)
    assert all(r["model"] != "Unrated Model" for r in rows)
    assert all(r["model"] != "Model" for r in rows)


def test_parse_returns_empty_for_markup_with_no_table():
    assert pull_aa.parse_leaderboard("<html><body><p>nope</p></body></html>") == []


def test_best_by_slug_keeps_the_highest_variant():
    rows = pull_aa.parse_leaderboard(FIXTURE)
    best = pull_aa.best_by_slug(rows)
    kimi = best["kimik3"]
    assert kimi["intelligence_index"] == 57
    assert kimi["variant"] == "max"
    assert kimi["aa_model"] == "Kimi K3 (max)"


def test_best_by_slug_labels_a_row_with_no_parenthetical():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert best["glm52"]["variant"] == "default"
    assert best["glm52"]["intelligence_index"] == 34


def test_best_by_slug_collapses_variants_to_one_entry_per_model():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    assert sorted(best) == ["claudeopus5", "glm52", "kimik3", "llama3370b"]


def test_best_by_slug_is_order_independent():
    """The winner must not depend on which variant the page lists first."""
    rows = [
        {"model": "M (low)", "creator": "C", "intelligence_index": 10},
        {"model": "M (max)", "creator": "C", "intelligence_index": 20},
    ]
    assert pull_aa.best_by_slug(rows)["m"]["intelligence_index"] == 20
    assert pull_aa.best_by_slug(rows[::-1])["m"]["intelligence_index"] == 20


def test_best_by_slug_keeps_different_sizes_apart():
    """Size is identity: 72B and 7B are different models, not variants.

    names.strip_decorations drops size tokens, which is right for pull_arena's
    pairwise ratio matching and fatally wrong here — both would key to 'qwen25'
    and one would silently overwrite the other.
    """
    rows = [
        {"model": "Qwen2.5 72B", "creator": "Alibaba", "intelligence_index": 30},
        {"model": "Qwen2.5 7B", "creator": "Alibaba", "intelligence_index": 12},
    ]
    best = pull_aa.best_by_slug(rows)
    assert sorted(best) == ["qwen2572b", "qwen257b"]
    assert best["qwen2572b"]["intelligence_index"] == 30
    assert best["qwen257b"]["intelligence_index"] == 12


def test_match_joins_on_the_model_name():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["moonshotai/Kimi-K3"]["intelligence_index"] == 57
    assert scores["moonshotai/Kimi-K3"]["variant"] == "max"


def test_match_tolerates_an_instruct_suffix_on_our_side():
    """AA says 'Llama 3.3 70B'; models.yaml says 'Llama 3.3 70B Instruct'."""
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["meta-llama/Llama-3.3-70B-Instruct"]["intelligence_index"] == 9


def test_match_falls_back_to_the_repo_tail():
    """'Some Model' does not match, but the repo tail GLM-5.2 does."""
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["zai-org/GLM-5.2"]["intelligence_index"] == 34


def test_match_reports_aa_rows_that_hit_nothing():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    _, unmatched = pull_aa.match_to_tracked(best, TRACKED)
    assert unmatched == ["Claude Opus 5 (max)"]


def test_match_records_the_source_url():
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    scores, _ = pull_aa.match_to_tracked(best, TRACKED)
    assert scores["moonshotai/Kimi-K3"]["source"] == pull_aa.LEADERBOARD_URL


def test_tracked_models_reads_models_yaml(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(yaml.safe_dump({"models": [
        {"name": "M", "hf_repo": "org/m"},
        {"name": "No repo"},
    ]}))
    assert pull_aa.tracked_models(f) == [{"name": "M", "hf_repo": "org/m"}]


def test_keys_for_orders_name_before_repo_tail():
    """name is checked before the hf_repo tail, so precedence must be explicit.

    A set has no order, so returning one made the winner depend on Python's
    hash-randomized iteration when the two sources resolve to different AA
    entries (see test_match_prefers_name_key_over_repo_tail_key below). The
    fix is an ordered, deduplicated sequence: name first, repo tail second.
    """
    model = {"name": "Name Key", "hf_repo": "org/Repo-Key"}
    assert pull_aa._keys_for(model) == ["namekey", "repokey"]


def test_keys_for_dedupes_when_name_and_repo_tail_match():
    """The two sources often agree; that must yield one key, not a repeat."""
    model = {"name": "Same", "hf_repo": "org/same"}
    assert pull_aa._keys_for(model) == ["same"]


def test_match_prefers_name_key_over_repo_tail_key():
    """When name and repo-tail keys resolve to different AA rows, name wins.

    Before the fix this depended on set iteration order (hash-randomized),
    so the same inputs could pick either row across runs.
    """
    best = {
        "namekey": {
            "model_slug": "namekey", "aa_model": "Name Entry",
            "variant": "default", "intelligence_index": 10,
        },
        "repokey": {
            "model_slug": "repokey", "aa_model": "Repo Entry",
            "variant": "default", "intelligence_index": 99,
        },
    }
    tracked = [{"name": "Name Key", "hf_repo": "org/Repo-Key"}]
    scores, unmatched = pull_aa.match_to_tracked(best, tracked)
    assert scores["org/Repo-Key"]["intelligence_index"] == 10
    assert scores["org/Repo-Key"]["aa_model"] == "Name Entry"
    assert unmatched == ["Repo Entry"]


def test_match_does_not_double_claim_an_aa_entry(capsys):
    """An AA row may score at most one tracked model; first claim wins.

    Two tracked rows can produce overlapping keys (a near-duplicate entry, or
    one row's name-key colliding with another row's repo-tail-key). Silently
    giving both the same AA measurement would double-count one data point as
    two models' scores with no trace of it happening.
    """
    best = {
        "dupe": {
            "model_slug": "dupe", "aa_model": "Dupe Model",
            "variant": "default", "intelligence_index": 50,
        },
    }
    tracked = [
        {"name": "Dupe", "hf_repo": "org/first-repo"},
        {"name": "Something Else", "hf_repo": "org/Dupe"},
    ]
    scores, unmatched = pull_aa.match_to_tracked(best, tracked)
    assert scores["org/first-repo"]["intelligence_index"] == 50
    assert "org/Dupe" not in scores
    assert unmatched == []

    warning = capsys.readouterr().out
    assert "org/first-repo" in warning
    assert "org/Dupe" in warning
    assert "Dupe Model" in warning


def test_refresh_writes_scores_and_unmatched(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    n = pull_aa.refresh(out, FIXTURE, TRACKED)
    assert n == 3
    doc = yaml.safe_load(out.read_text())
    assert doc["scores"]["moonshotai/Kimi-K3"]["intelligence_index"] == 57
    assert doc["unmatched"] == ["Claude Opus 5 (max)"]


def test_refresh_leaves_the_sidecar_untouched_when_the_fetch_failed(tmp_path):
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  org/m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, None, TRACKED) is None
    assert out.read_text() == before


def test_refresh_treats_a_zero_row_parse_as_failure(tmp_path):
    """Empty parse means AA's markup changed — do not erase good data."""
    out = tmp_path / "aa_scores.yaml"
    out.write_text("scores:\n  org/m:\n    intelligence_index: 42\n")
    before = out.read_text()

    assert pull_aa.refresh(out, "<html><body>redesigned</body></html>", TRACKED) is None
    assert out.read_text() == before


def test_refresh_writes_an_empty_score_set_when_rows_parsed_but_matched_nothing(tmp_path):
    """Parsed fine, matched nobody: that is a real answer, not a failure."""
    out = tmp_path / "aa_scores.yaml"
    assert pull_aa.refresh(out, FIXTURE, []) == 0
    assert yaml.safe_load(out.read_text())["scores"] == {}


def test_fetch_html_returns_none_on_error():
    def boom(url, **kw):
        raise RuntimeError("HTTP Error 429: Too Many Requests")
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=boom) is None


def test_fetch_html_returns_none_on_bad_status():
    class Resp:
        status_code = 503
        text = "nope"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) is None


def test_fetch_html_returns_the_body_on_success():
    class Resp:
        status_code = 200
        text = "<html>ok</html>"
    assert pull_aa.fetch_html(pull_aa.LEADERBOARD_URL, get=lambda u, **kw: Resp()) == "<html>ok</html>"


# --- the join covers the review queue, not just the published index ---------

def test_staged_models_reads_candidates(tmp_path):
    f = tmp_path / "candidates.yaml"
    f.write_text(yaml.safe_dump({"models": [
        {"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"},
        {"name": "No repo"},
    ]}))
    assert pull_aa.staged_models(f) == [
        {"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"}]


def test_staged_models_tolerates_a_missing_file(tmp_path):
    assert pull_aa.staged_models(tmp_path / "nope.yaml") == []


def test_joinable_prefers_models_yaml_on_duplicate_repo(tmp_path):
    """A row mid-promotion can briefly sit in both files; score it once."""
    data = tmp_path / "models.yaml"
    data.write_text(yaml.safe_dump({"models": [
        {"name": "Tracked Name", "hf_repo": "org/m"}]}))
    cands = tmp_path / "candidates.yaml"
    cands.write_text(yaml.safe_dump({"models": [
        {"name": "Staged Name", "hf_repo": "org/m"},
        {"name": "Other", "hf_repo": "org/n"}]}))

    rows = pull_aa.joinable_models(data, cands)

    assert [r["hf_repo"] for r in rows] == ["org/m", "org/n"]
    assert rows[0]["name"] == "Tracked Name"


def test_staged_candidates_get_scored(tmp_path):
    """A model in the review queue must not be reported as unmatched."""
    best = pull_aa.best_by_slug(pull_aa.parse_leaderboard(FIXTURE))
    staged = [{"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"}]

    scores, unmatched = pull_aa.match_to_tracked(best, staged)

    assert scores["moonshotai/Kimi-K3"]["intelligence_index"] == 57
    assert "Kimi K3 (max)" not in unmatched
