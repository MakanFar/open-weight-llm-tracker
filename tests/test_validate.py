import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate


def _valid_row(**kw):
    row = dict(name="M", developer="d", release_date=date(2025, 6, 1),
               params_total_b=70.0, params_active_b=70.0, architecture="dense",
               context_window=131072, modality="text", license="mit",
               commercial_use=True)
    row.update(kw)
    return row


def test_row_errors_empty_for_a_valid_row():
    assert validate.row_errors(_valid_row()) == []


def test_row_errors_flags_a_non_date_release_date():
    """A hand-edited candidates.yaml row can carry a string like "sometime in
    2025" — validate.py's own check (isinstance(rd, date)) must catch it, and
    classify.py's promotion gate reuses exactly this function."""
    errors = validate.row_errors(_valid_row(release_date="sometime in 2025"))
    assert any("release_date" in e for e in errors)


def test_row_errors_flags_dense_active_not_equal_total():
    errors = validate.row_errors(
        _valid_row(architecture="dense", params_total_b=70.0, params_active_b=8.0))
    assert any("params_active_b" in e for e in errors)


def test_identity_errors_flags_two_rows_naming_the_same_weights():
    """A dated snapshot and its bare repo are one model, so one row.

    CLAUDE.md: 'one row per model — a family flagship or distinct sizes, never
    both.' Two rows sharing an identity would also both claim the same AA
    sidecar entry at render time.
    """
    models = [
        {"name": "DeepSeek V4 Flash", "hf_repo": "deepseek-ai/DeepSeek-V4-Flash"},
        {"name": "DeepSeek V4 Flash 0731",
         "hf_repo": "deepseek-ai/DeepSeek-V4-Flash-0731"},
    ]
    errors = validate.identity_errors(models)
    assert len(errors) == 1
    assert "deepseekv4flash" in errors[0]


def test_identity_errors_allows_distinct_sizes():
    """405B and 8B are separate rows by design."""
    models = [
        {"name": "Llama 3.1 405B", "hf_repo": "meta-llama/Llama-3.1-405B-Instruct"},
        {"name": "Llama 3.1 8B", "hf_repo": "meta-llama/Llama-3.1-8B-Instruct"},
    ]
    assert validate.identity_errors(models) == []


def test_identity_errors_ignores_rows_without_a_repo():
    assert validate.identity_errors([{"name": "M"}, {"name": "N", "hf_repo": ""}]) == []


def test_the_committed_models_file_has_no_identity_collisions():
    import yaml
    doc = yaml.safe_load(validate.DATA.read_text())
    assert validate.identity_errors(doc["models"]) == []
