import sys
from pathlib import Path

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


def test_promotion_row_keeps_the_activation_provenance():
    """A reviewer must be able to check where 32B came from."""
    assert discover.promotion_row(_candidate())["params_active_source"] == "32B activated"


def test_promotion_row_marks_commercial_use_unverified():
    assert discover.promotion_row(_candidate())["commercial_use_verified"] is False


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
