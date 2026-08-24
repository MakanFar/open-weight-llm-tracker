import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import discover

EXISTING = """\
# Open-weight tracker — SOURCE OF TRUTH
# Hand-curated. Do not reformat.

models:
  - name: Llama 3.3 70B Instruct
    hf_repo: meta-llama/Llama-3.3-70B-Instruct
    developer: Meta
    release_date: 2024-12-06
    params_total_b: 70
    params_active_b: 70
    architecture: dense
    context_window: 131072
    modality: text
    license: llama-3.3-community
    commercial_use: conditional
    license_notes: "Hand-written note that must survive."
"""


def _candidate(**kw):
    base = dict(name="GLM-5.2", hf_repo="zai-org/GLM-5.2", developer="zai-org",
                release_date="2026-06-16", params_total_b=753.3,
                params_active_b=32.0, architecture="moe", context_window=1048576,
                modality="text", license="mit", commercial_use=True,
                weights_url="https://huggingface.co/zai-org/GLM-5.2",
                discovered_via=["arena"], arena_rank=12, aa_index=51,
                downloads=1651533, needs_hf_repo=False,
                resolution_confidence="high", params_active_source="32B activated")
    base.update(kw)
    return base


def test_tracked_stems_reads_models_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    assert discover.tracked_stems(p) == {"llama70b"}


def test_promotion_row_strips_discovery_only_fields():
    row = discover.promotion_row(_candidate())
    for field in ("discovered_via", "arena_rank", "aa_index", "downloads",
                  "needs_hf_repo", "resolution_confidence"):
        assert field not in row, f"{field} must not leak into models.yaml"


def test_promotion_row_strips_needs_review():
    """needs_review is classify.route's own output, not a fact about the model.

    A row only ever reaches promotion_row after classify.route judged it
    complete, so any needs_review it still carries (e.g. from a prior run
    where it was staged, then a human filled the gap by hand) is stale and
    must not leak into models.yaml the way arena_rank and friends must not.
    """
    row = discover.promotion_row(_candidate(needs_review=["no-context-window"]))
    assert "needs_review" not in row


def test_promotion_row_strips_the_family_collision_marker():
    """family_collision_reviewed is a review-workflow field, not a model fact.

    It records that a human cleared a candidates.yaml collision; once the row
    is in models.yaml the collision is resolved and the marker is meaningless
    there. validate.py checks models.yaml only, so it would otherwise leak in
    the way arena_rank and needs_review must not.
    """
    row = discover.promotion_row(_candidate(family_collision_reviewed=True))
    assert "family_collision_reviewed" not in row


def test_promotion_row_keeps_the_activation_provenance():
    """A reviewer must be able to check where 32B came from."""
    assert discover.promotion_row(_candidate())["params_active_source"] == "32B activated"


def test_promotion_row_marks_commercial_use_unverified():
    assert discover.promotion_row(_candidate())["commercial_use_verified"] is False


def test_promotion_row_stamps_false_even_if_candidate_claims_verified():
    """A human hand-editing candidates.yaml could set this true from a licence
    tag, but the value was never actually read off a licence — it must be
    forced to False regardless of what the candidate carries in."""
    row = discover.promotion_row(_candidate(commercial_use_verified=True))
    assert row["commercial_use_verified"] is False


def test_promotion_row_drops_the_auto_discovery_boilerplate_note():
    row = discover.promotion_row(_candidate(
        notes="Auto-discovered candidate; review before merging into models.yaml."))
    assert "notes" not in row


def test_promotion_row_keeps_a_human_written_note():
    row = discover.promotion_row(_candidate(notes="Confirmed with vendor blog post."))
    assert row["notes"] == "Confirmed with vendor blog post."


def test_append_leaves_existing_rows_byte_identical(tmp_path):
    """The invariant. A human's hand-edited row must survive untouched."""
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)

    discover.append_models(p, [discover.promotion_row(_candidate())])

    text = p.read_text()
    assert EXISTING.rstrip("\n") in text, "existing content was rewritten"
    assert "Hand-written note that must survive." in text
    assert text.startswith("# Open-weight tracker"), "comment header lost"


def test_append_adds_the_new_row(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)

    n = discover.append_models(p, [discover.promotion_row(_candidate())])

    assert n == 1
    doc = yaml.safe_load(p.read_text())
    assert [m["hf_repo"] for m in doc["models"]] == [
        "meta-llama/Llama-3.3-70B-Instruct", "zai-org/GLM-5.2"]


def test_append_of_nothing_leaves_the_file_untouched(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    before = p.read_text()

    assert discover.append_models(p, []) == 0
    assert p.read_text() == before


def test_appended_file_still_parses_and_validates_shape(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    discover.append_models(p, [discover.promotion_row(_candidate())])

    doc = yaml.safe_load(p.read_text())
    new = doc["models"][-1]
    for field in ("name", "developer", "release_date", "params_total_b",
                  "params_active_b", "architecture", "context_window",
                  "modality", "license", "commercial_use"):
        assert field in new


def test_append_separates_the_new_row_with_a_blank_line(tmp_path):
    """Every row pair in the real file has a blank line between them; an
    appended row must match that hand-formatting instead of butting directly
    against the previous row's last line."""
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    discover.append_models(p, [discover.promotion_row(_candidate())])

    text = p.read_text()
    assert '"Hand-written note that must survive."\n\n  - name: GLM-5.2' in text


def test_append_separates_multiple_new_rows_from_each_other(tmp_path):
    """Blank-line separation is the file's convention between EVERY row
    pair, including two rows landing in the same append batch — not just
    between the last existing row and the first new one."""
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)
    second = _candidate(name="Kimi K2.7", hf_repo="moonshotai/Kimi-K2.7")
    discover.append_models(p, [discover.promotion_row(_candidate()),
                                discover.promotion_row(second)])

    text = p.read_text()
    assert "  - name: GLM-5.2" in text and "  - name: Kimi K2.7" in text
    between = text.split("  - name: GLM-5.2", 1)[1]
    assert "\n\n  - name: Kimi K2.7" in between


def test_append_refuses_when_last_top_level_key_is_not_models(tmp_path):
    """If models.yaml ever ends with a second top-level key (e.g. a `sources:`
    block), naive EOF-append would land the new row under that key instead of
    under `models:` — the file would still parse, `models` would be
    unchanged, and the promoted model would silently vanish. Refuse instead
    of writing."""
    p = tmp_path / "models.yaml"
    bad = EXISTING + "\nsources:\n  - https://example.com\n"
    p.write_text(bad)

    with pytest.raises(Exception):
        discover.append_models(p, [discover.promotion_row(_candidate())])

    assert p.read_text() == bad


def test_append_refuses_on_empty_file(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text("")

    with pytest.raises(Exception):
        discover.append_models(p, [discover.promotion_row(_candidate())])

    assert p.read_text() == ""


def test_append_refuses_when_no_models_key_present(tmp_path):
    p = tmp_path / "models.yaml"
    header_only = "# Open-weight tracker — SOURCE OF TRUTH\n# Hand-curated. Do not reformat.\n"
    p.write_text(header_only)

    with pytest.raises(Exception):
        discover.append_models(p, [discover.promotion_row(_candidate())])

    assert p.read_text() == header_only


def test_append_writes_atomically(tmp_path, monkeypatch):
    """A crash mid-write must not leave models.yaml truncated. append_models
    should write to a temp file in the same directory and os.replace() it
    onto the target, never truncate the target in place."""
    p = tmp_path / "models.yaml"
    p.write_text(EXISTING)

    real_replace = os.replace
    called = {}

    def spy_replace(src, dst):
        called["src"] = Path(src)
        called["dst"] = Path(dst)
        # At the moment of replace, the temp file must already hold the full
        # new content and the target must still hold the OLD content
        # untouched — proof the write happened off to the side first.
        assert Path(src).read_text() != EXISTING
        assert Path(dst).read_text() == EXISTING
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    discover.append_models(p, [discover.promotion_row(_candidate())])

    assert called, "os.replace was never called — write is not atomic"
    assert called["dst"] == p
    assert called["src"].parent == p.parent
