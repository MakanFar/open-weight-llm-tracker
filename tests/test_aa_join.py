"""The AA name->repo join, which used to happen inside pull_aa's write.

Most of these tests moved here from tests/test_pull_aa.py with
match_to_tracked renamed to claims(); the ones about freshness are new, and
are the reason the module exists.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import yaml
import aa_join
import pull_aa

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "aa_leaderboard.html").read_text()

TRACKED = [
    {"name": "Kimi K3", "hf_repo": "moonshotai/Kimi-K3"},
    {"name": "Llama 3.3 70B Instruct", "hf_repo": "meta-llama/Llama-3.3-70B-Instruct"},
    {"name": "Some Model", "hf_repo": "zai-org/GLM-5.2"},
]


def scraped(tmp_path):
    """A real sidecar, written the way pull_aa writes one."""
    out = tmp_path / "aa_scores.yaml"
    pull_aa.refresh(out, FIXTURE)
    return out


# --- loading ---------------------------------------------------------------

def test_load_entries_reads_every_row_the_scraper_wrote(tmp_path):
    entries = aa_join.load_entries(scraped(tmp_path))
    assert sorted(entries) == ["claudeopus5", "glm52", "kimik3", "llama3370b"]
    assert entries["kimik3"]["intelligence_index"] == 57
    assert entries["kimik3"]["variant"] == "max"
    assert entries["kimik3"]["aa_model"] == "Kimi K3 (max)"


def _entries_from(doc, tmp_path):
    """load_entries over a document written to disk."""
    f = tmp_path / "aa_scores.yaml"
    f.write_text(yaml.safe_dump(doc))
    return aa_join.load_entries(f)


def test_load_entries_rederives_the_key_rather_than_trusting_the_file(tmp_path):
    """names.slug stays the single authority on the key.

    A key read back from disk is frozen against whatever normalisation wrote
    it: change names.slug and every key in the committed file silently stops
    matching, with no error and no diff to point at.
    """
    doc = {"scores": {"stale-key-from-an-old-run": {
        "aa_model": "Kimi K3 (max)", "intelligence_index": 57, "variant": "max"}}}
    assert set(_entries_from(doc, tmp_path)) == {"kimik3"}


def test_load_entries_drops_rows_with_no_usable_score(tmp_path):
    doc = {"scores": {
        "a": {"aa_model": "No Index"},
        "b": {"aa_model": "Bool Index", "intelligence_index": True},
        "c": {"aa_model": "Text Index", "intelligence_index": "57"},
        "d": {"intelligence_index": 40},
        "e": {"aa_model": "Good One", "intelligence_index": 40},
    }}
    assert set(_entries_from(doc, tmp_path)) == {"goodone"}


def test_load_entries_tolerates_a_missing_file(tmp_path):
    assert aa_join.load_entries(tmp_path / "nope.yaml") == {}


def test_load_entries_tolerates_a_malformed_file(tmp_path):
    for text in ("", "scores: nope\n", "- not\n- a mapping\n", "{{{\n"):
        f = tmp_path / "aa_scores.yaml"
        f.write_text(text)
        assert aa_join.load_entries(f) == {}


# --- keys ------------------------------------------------------------------

def test_keys_for_orders_name_before_repo_tail():
    """name is checked before the hf_repo tail, so precedence must be explicit.

    A set has no order, so returning one made the winner depend on Python's
    hash-randomized iteration when the two sources resolve to different AA
    entries (see test_claims_prefers_name_key_over_repo_tail_key below).
    """
    assert aa_join.keys_for({"name": "Name Key", "hf_repo": "org/Repo-Key"}) \
        == ["namekey", "repokey"]


def test_keys_for_dedupes_when_name_and_repo_tail_match():
    """The two sources often agree; that must yield one key, not a repeat."""
    assert aa_join.keys_for({"name": "Same", "hf_repo": "org/same"}) == ["same"]


def test_keys_for_tolerates_a_row_with_no_name():
    """models.yaml always carries one; candidates.yaml is hand-edited."""
    assert aa_join.keys_for({"hf_repo": "org/GLM-5.2"}) == ["glm52"]
    assert aa_join.keys_for({}) == []
    assert aa_join.keys_for("not a row") == []


# --- joining ---------------------------------------------------------------

def test_join_matches_on_the_model_name(tmp_path):
    joined = aa_join.join(aa_join.load_entries(scraped(tmp_path)), TRACKED)
    assert joined["moonshotai/kimi-k3"]["intelligence_index"] == 57
    assert joined["moonshotai/kimi-k3"]["variant"] == "max"


def test_join_tolerates_an_instruct_suffix_on_our_side(tmp_path):
    """AA says 'Llama 3.3 70B'; models.yaml says 'Llama 3.3 70B Instruct'."""
    joined = aa_join.join(aa_join.load_entries(scraped(tmp_path)), TRACKED)
    assert joined["meta-llama/llama-3.3-70b-instruct"]["intelligence_index"] == 9


def test_join_falls_back_to_the_repo_tail(tmp_path):
    """'Some Model' does not match, but the repo tail GLM-5.2 does."""
    joined = aa_join.join(aa_join.load_entries(scraped(tmp_path)), TRACKED)
    assert joined["zai-org/glm-5.2"]["intelligence_index"] == 34


def test_join_is_keyed_by_the_rows_own_repo(tmp_path):
    """Which is why no caller needs a repo_identity fallback.

    The old sidecar stored whichever repo string the scrape-time join landed
    on, and aa_cell reconciled the two spellings. There is one spelling now.
    """
    entries = aa_join.load_entries(scraped(tmp_path))
    joined = aa_join.join(entries, [{"name": "Kimi K3",
                                     "hf_repo": "some-mirror/Kimi-K3"}])
    assert list(joined) == ["some-mirror/kimi-k3"]


def test_join_falls_back_to_repo_identity_for_a_dated_snapshot():
    """AA names the snapshot it measured; models.yaml names the release.

    "DeepSeek V4 Flash 0731" against deepseek-ai/DeepSeek-V4-Flash is a real
    published score on a real tracked row, and neither exact key bridges it.
    This is the last resort, tried only after name and repo tail both miss.
    """
    entries = {"deepseekv4flash0731": {"aa_model": "DeepSeek V4 Flash 0731 (max)",
                                       "intelligence_index": 52, "variant": "max"}}
    rows = [{"name": "DeepSeek-V4-Flash",
             "hf_repo": "deepseek-ai/DeepSeek-V4-Flash"}]
    joined = aa_join.join(entries, rows)
    assert joined["deepseek-ai/deepseek-v4-flash"]["intelligence_index"] == 52


def test_an_exact_key_beats_the_identity_fallback():
    entries = {
        "deepseekv4flash": {"aa_model": "DeepSeek V4 Flash",
                            "intelligence_index": 44, "variant": "default"},
        "deepseekv4flash0731": {"aa_model": "DeepSeek V4 Flash 0731 (max)",
                                "intelligence_index": 52, "variant": "max"},
    }
    rows = [{"name": "DeepSeek-V4-Flash",
             "hf_repo": "deepseek-ai/DeepSeek-V4-Flash"}]
    joined = aa_join.join(entries, rows)
    assert joined["deepseek-ai/deepseek-v4-flash"]["intelligence_index"] == 44


def test_an_identity_two_entries_claim_is_dropped_not_guessed(capsys):
    """Two snapshots of one model: we cannot know which the row means.

    This is live data — AA lists DeepSeek V4 Pro and DeepSeek V4 Pro 0813.
    A wrong number is worse than no number.
    """
    entries = {
        "deepseekv4pro0813": {"aa_model": "DeepSeek V4 Pro 0813",
                              "intelligence_index": 60, "variant": "default"},
        "deepseekv4pro0901": {"aa_model": "DeepSeek V4 Pro 0901",
                              "intelligence_index": 62, "variant": "default"},
    }
    rows = [{"name": "DeepSeek-V4-Pro", "hf_repo": "deepseek-ai/DeepSeek-V4-Pro"}]

    assert aa_join.join(entries, rows) == {}
    assert "deepseekv4pro" in capsys.readouterr().out


def test_the_identity_fallback_does_not_collapse_distinct_sizes():
    """405B and 8B are different models, and repo_identity keeps size tokens."""
    entries = {
        "llama31405b": {"aa_model": "Llama 3.1 405B",
                        "intelligence_index": 30, "variant": "default"},
        "llama318b": {"aa_model": "Llama 3.1 8B",
                      "intelligence_index": 12, "variant": "default"},
    }
    rows = [{"name": "x", "hf_repo": "meta-llama/Llama-3.1-405B-Instruct"}]
    joined = aa_join.join(entries, rows)
    assert joined["meta-llama/llama-3.1-405b-instruct"]["intelligence_index"] == 30


def test_join_skips_rows_with_no_repo(tmp_path):
    entries = aa_join.load_entries(scraped(tmp_path))
    assert aa_join.join(entries, [{"name": "Kimi K3"}, None, "junk"]) == {}


def test_claims_prefers_name_key_over_repo_tail_key():
    """When name and repo-tail keys resolve to different AA rows, name wins."""
    entries = {
        "namekey": {"aa_model": "Name Entry", "variant": "default",
                    "intelligence_index": 10},
        "repokey": {"aa_model": "Repo Entry", "variant": "default",
                    "intelligence_index": 99},
    }
    joined, claimed = aa_join.claims(entries, [{"name": "Name Key",
                                                "hf_repo": "org/Repo-Key"}])
    assert joined["org/repo-key"]["aa_model"] == "Name Entry"
    assert aa_join.unclaimed(entries, claimed) == ["Repo Entry"]


def test_claims_does_not_double_claim_an_aa_entry(capsys):
    """An AA row may score at most one row; first claim wins.

    Two rows can produce overlapping keys (a near-duplicate models.yaml
    entry, or one row's name-key colliding with another's repo-tail-key).
    Silently giving both the same measurement would double-count one data
    point as two models' scores with no trace of it happening.
    """
    entries = {"dupe": {"aa_model": "Dupe Model", "variant": "default",
                        "intelligence_index": 50}}
    rows = [{"name": "Dupe", "hf_repo": "org/first-repo"},
            {"name": "Something Else", "hf_repo": "org/Dupe"}]

    joined, claimed = aa_join.claims(entries, rows)

    assert joined["org/first-repo"]["intelligence_index"] == 50
    assert "org/dupe" not in joined
    assert aa_join.unclaimed(entries, claimed) == []

    warning = capsys.readouterr().out
    assert "org/first-repo" in warning
    assert "org/Dupe" in warning
    assert "Dupe Model" in warning


def test_unclaimed_reports_aa_rows_that_hit_nothing(tmp_path):
    entries = aa_join.load_entries(scraped(tmp_path))
    _, claimed = aa_join.claims(entries, TRACKED)
    assert aa_join.unclaimed(entries, claimed) == ["Claude Opus 5 (max)"]


# --- freshness: the reason this module exists ------------------------------

def test_a_row_added_after_the_scrape_is_scored_immediately(tmp_path):
    """The GLM-5.3-Flash case.

    discover.yml scrapes before it discovers. Under the old design the join
    was frozen into the file at scrape time, so a model promoted this run was
    joined against an index that predated it and rendered an em dash for a
    week — exactly the release the tracker exists to surface. Joining at the
    point of use means the row is scored the moment it exists, with no
    re-scrape.
    """
    entries = aa_join.load_entries(scraped(tmp_path))

    before = aa_join.join(entries, [])
    after = aa_join.join(entries, TRACKED)

    assert before == {}
    assert after["zai-org/glm-5.2"]["intelligence_index"] == 34


def test_a_deleted_row_releases_its_claim(tmp_path):
    entries = aa_join.load_entries(scraped(tmp_path))
    _, claimed = aa_join.claims(entries, [TRACKED[0]])
    assert "GLM-5.2" in aa_join.unclaimed(entries, claimed)


def test_a_renamed_row_loses_its_score_in_the_same_run(tmp_path):
    """Not a week later. The failure shows up in the diff that caused it."""
    entries = aa_join.load_entries(scraped(tmp_path))
    renamed = [{"name": "Something Unrecognisable",
                "hf_repo": "moonshotai/Renamed"}]
    assert aa_join.join(entries, renamed) == {}


# --- row loading -----------------------------------------------------------

def test_tracked_models_reads_models_yaml(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text(yaml.safe_dump({"models": [
        {"name": "M", "hf_repo": "org/m"},
        {"name": "No repo"},
    ]}))
    assert aa_join.tracked_models(f) == [{"name": "M", "hf_repo": "org/m"}]


def test_staged_models_reads_candidates(tmp_path):
    f = tmp_path / "candidates.yaml"
    f.write_text(yaml.safe_dump({"models": [
        {"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"},
        {"name": "No repo"},
    ]}))
    assert aa_join.staged_models(f) == [
        {"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"}]


def test_staged_models_tolerates_a_missing_file(tmp_path):
    assert aa_join.staged_models(tmp_path / "nope.yaml") == []


def test_union_prefers_the_first_list_on_a_duplicate_repo():
    """A row mid-promotion can briefly sit in both files; join it once."""
    rows = aa_join.union([{"name": "Tracked Name", "hf_repo": "org/m"}],
                         [{"name": "Staged Name", "hf_repo": "org/M"},
                          {"name": "Other", "hf_repo": "org/n"}])
    assert [r["hf_repo"] for r in rows] == ["org/m", "org/n"]
    assert rows[0]["name"] == "Tracked Name"


def test_joinable_models_spans_both_files(tmp_path):
    data = tmp_path / "models.yaml"
    data.write_text(yaml.safe_dump({"models": [
        {"name": "Tracked Name", "hf_repo": "org/m"}]}))
    cands = tmp_path / "candidates.yaml"
    cands.write_text(yaml.safe_dump({"models": [
        {"name": "Staged Name", "hf_repo": "org/m"},
        {"name": "Other", "hf_repo": "org/n"}]}))

    rows = aa_join.joinable_models(data, cands)

    assert [r["hf_repo"] for r in rows] == ["org/m", "org/n"]
    assert rows[0]["name"] == "Tracked Name"


def test_a_staged_candidate_can_be_scored(tmp_path):
    """A model in the review queue is scored before promotion, not after.

    A reviewer deciding whether to take a model wants its index in hand.
    """
    entries = aa_join.load_entries(scraped(tmp_path))
    staged = [{"name": "Kimi-K3", "hf_repo": "moonshotai/Kimi-K3"}]

    joined, claimed = aa_join.claims(entries, staged)

    assert joined["moonshotai/kimi-k3"]["intelligence_index"] == 57
    assert "Kimi K3 (max)" not in aa_join.unclaimed(entries, claimed)
