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


def test_row_errors_flags_a_string_params_total_b():
    """CONTRIBUTING.md tells reviewers to type params_active_b into
    candidates.yaml by hand, so a string like "700B" (units left on, or a
    stray quote) is a very plausible hand-edit. Before this check,
    row_errors let it through, classify.route() promoted it, and
    render_readme.human_params blew up with `f"{total:g}"` on a str -- with
    no continue-on-error on that CI step, the whole weekly PR was lost.

    architecture="moe" and a numeric params_active_b keep the pre-existing
    dense-equality check (whose message text happens to mention both field
    names) from firing, so this isolates the new type check specifically.
    """
    errors = validate.row_errors(
        _valid_row(architecture="moe", params_total_b="700B", params_active_b=22.0))
    assert any("params_total_b must be a number" in e for e in errors)


def test_row_errors_flags_a_string_params_active_b():
    errors = validate.row_errors(
        _valid_row(architecture="moe", params_total_b=700.0, params_active_b="22B"))
    assert any("params_active_b must be a number" in e for e in errors)


def test_row_errors_rejects_bool_for_params_fields():
    """bool is a subclass of int in Python, so isinstance(True, int) is True
    -- a plain isinstance(x, (int, float)) check would silently accept it."""
    errors = validate.row_errors(
        _valid_row(architecture="moe", params_total_b=True, params_active_b=22.0))
    assert any("params_total_b must be a number" in e for e in errors)


def test_row_errors_allows_int_params():
    """Integers are a legitimate way to write a whole-number param count."""
    assert validate.row_errors(_valid_row(params_total_b=70, params_active_b=70)) == []


def test_row_errors_flags_a_non_bool_commercial_use_verified():
    """commercial_use_verified: "no" is truthy in Python, so
    render_readme.commercial_badge would render it as verified -- publishing
    an unverified legal claim as a checked one, exactly what the trailing
    `?` marker exists to prevent."""
    errors = validate.row_errors(_valid_row(commercial_use_verified="no"))
    assert any("commercial_use_verified" in e for e in errors)


def test_row_errors_allows_a_missing_commercial_use_verified():
    """Not every row carries this optional field; absence is not an error."""
    assert validate.row_errors(_valid_row()) == []


def test_row_errors_allows_real_bool_commercial_use_verified():
    assert validate.row_errors(_valid_row(commercial_use_verified=True)) == []
    assert validate.row_errors(_valid_row(commercial_use_verified=False)) == []


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
