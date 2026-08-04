import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate


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
