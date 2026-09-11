"""refresh_review.py re-derives needs_review after something edits the queue.

The pipeline does not need this — the next run rebuilds the file and
classify.route() re-derives its own verdict rather than trusting the field.
It exists for the human reading the PR, who would otherwise see a row
carrying a filled params_active_b AND moe-active-params-unknown at once.
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import yaml
import refresh_review


def moe(**over):
    row = {"name": "M", "hf_repo": "acme/M-9B", "developer": "acme",
           "release_date": date(2026, 9, 1), "params_total_b": 100.0,
           "params_active_b": 100.0, "architecture": "moe",
           "context_window": 131072, "modality": "text", "license": "mit",
           "commercial_use": True, "aa_index": 40,
           "weights_url": "https://huggingface.co/acme/M-9B"}
    row.update(over)
    return row


# --- the contradiction it exists to remove --------------------------------

def test_a_filled_activation_clears_its_own_reason():
    row = moe(params_active_b=16.0, needs_review=["moe-active-params-unknown"])
    changed = refresh_review.restate([row], set())
    assert len(changed) == 1
    assert "moe-active-params-unknown" not in (row.get("needs_review") or [])


def test_a_row_with_nothing_left_loses_the_key_entirely():
    """`needs_review: []` reads as 'reviewed and cleared', which is a claim
    this script is in no position to make. Absent means absent."""
    row = moe(params_active_b=16.0, needs_review=["moe-active-params-unknown"])
    refresh_review.restate([row], set())
    assert "needs_review" not in row


def test_an_unfilled_row_keeps_its_reason():
    row = moe(needs_review=["moe-active-params-unknown"])
    assert refresh_review.restate([row], set()) == []
    assert row["needs_review"] == ["moe-active-params-unknown"]


# --- it must not launder away the reasons the agent may not touch ---------

def test_a_duplicate_flag_survives_a_restate():
    """The supersede-or-coexist decision is a human's. Re-deriving the list
    must not be a back door to clearing it."""
    row = moe(params_active_b=16.0,
              needs_review=["moe-active-params-unknown", "duplicates-tracked-row"])
    refresh_review.restate([row], {"m9b"})
    assert row["needs_review"] == ["duplicates-tracked-row"]


def test_a_reviewed_collision_marker_still_works_through_restate():
    row = moe(params_active_b=16.0, duplicate_reviewed=True,
              needs_review=["duplicates-tracked-row"])
    refresh_review.restate([row], {"m9b"})
    assert "needs_review" not in row


def test_a_new_gap_introduced_by_an_edit_is_reported():
    """It re-derives; it does not only subtract. An edit that breaks a field
    must show up rather than be silently kept off the list."""
    row = moe(context_window=0)
    changed = refresh_review.restate([row], set())
    assert "no-context-window" in row["needs_review"]
    assert changed


# --- scope --------------------------------------------------------------

def test_it_touches_no_field_but_needs_review():
    row = moe(params_active_b=16.0, needs_review=["moe-active-params-unknown"])
    before = {k: v for k, v in row.items() if k != "needs_review"}
    refresh_review.restate([row], set())
    assert {k: v for k, v in row.items() if k != "needs_review"} == before


def test_a_non_dict_row_is_skipped_not_crashed_on():
    assert refresh_review.restate(["junk", None], set()) == []


# --- --count-with, which gates the agent step ----------------------------

def test_count_with_finds_rows_carrying_any_named_reason():
    rows = [moe(needs_review=["moe-active-params-unknown"]),
            moe(needs_review=["license-not-allowlisted"]),
            moe(needs_review=["duplicates-tracked-row"]),
            moe()]
    hits = refresh_review.count_with(
        rows, ["moe-active-params-unknown", "license-not-allowlisted"])
    assert len(hits) == 2


def test_count_with_prints_a_bare_integer_the_workflow_can_read(capsys, tmp_path):
    path = tmp_path / "candidates.yaml"
    path.write_text(yaml.safe_dump(
        {"generated": "2026-09-10",
         "models": [moe(needs_review=["license-not-allowlisted"])]}))
    refresh_review.main(["--candidates", str(path),
                         "--count-with", "license-not-allowlisted"])
    assert capsys.readouterr().out.strip() == "1"


# --- the file it writes ---------------------------------------------------

def test_write_preserves_the_run_stamp(tmp_path):
    """If this bumped the stamp it would be indistinguishable from a real
    change to pr_gate.py, and every run would look PR-worthy again."""
    path = tmp_path / "candidates.yaml"
    data = tmp_path / "models.yaml"
    data.write_text(yaml.safe_dump({"models": []}))
    path.write_text(yaml.safe_dump(
        {"generated": "2026-01-01",
         "models": [moe(params_active_b=16.0,
                        needs_review=["moe-active-params-unknown"])]}))

    refresh_review.main(["--write", "--candidates", str(path),
                         "--data", str(data)])
    assert yaml.safe_load(path.read_text())["generated"] == "2026-01-01"


def test_a_dry_run_leaves_the_file_alone(tmp_path):
    path = tmp_path / "candidates.yaml"
    data = tmp_path / "models.yaml"
    data.write_text(yaml.safe_dump({"models": []}))
    path.write_text(yaml.safe_dump(
        {"generated": "2026-01-01",
         "models": [moe(params_active_b=16.0,
                        needs_review=["moe-active-params-unknown"])]}))
    before = path.read_text()

    refresh_review.main(["--candidates", str(path), "--data", str(data)])
    assert path.read_text() == before


def test_it_never_writes_models_yaml(tmp_path):
    path = tmp_path / "candidates.yaml"
    data = tmp_path / "models.yaml"
    data.write_text(yaml.safe_dump({"models": []}))
    before = data.read_text()
    path.write_text(yaml.safe_dump(
        {"generated": "2026-01-01",
         "models": [moe(params_active_b=16.0,
                        needs_review=["moe-active-params-unknown"])]}))

    refresh_review.main(["--write", "--candidates", str(path),
                         "--data", str(data)])
    assert data.read_text() == before
