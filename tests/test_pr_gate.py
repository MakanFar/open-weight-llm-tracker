"""The daily schedule's PR gate.

Two fields in discover.yml's output move on every single run by design: the
`generated:` stamp in candidates.yaml and the README badge that renders it.
pr_gate decides whether anything ELSE moved. The tests that matter most are
the ones proving it does not over-suppress — a swallowed frontier release
leaves no PR and no error, and would be found only by someone noticing the
model missing weeks later.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pr_gate

STAMPED = "# header\ngenerated: '{date}'\nmodels:\n{rows}"
README = (
    "# Tracker\n"
    "[![validate](https://img.shields.io/badge/validate-pass-green)](x)\n"
    "[![index updated](https://img.shields.io/badge/index%20updated-{date}-1f6feb)](y)\n"
    "\n| Model | Params |\n{rows}"
)


def queue(date, rows="- {name: A}\n"):
    return STAMPED.format(date=date, rows=rows)


def readme(date, rows="| A | 7 |\n"):
    return README.format(date=date, rows=rows)


# --- the whole point: a run that learned nothing opens nothing -------------

def test_a_stamp_only_run_is_not_worth_a_pr():
    before = {"candidates.yaml": queue("2026-09-09"),
              "README.md": readme("2026-09-09")}
    after = {"candidates.yaml": queue("2026-09-10"),
             "README.md": readme("2026-09-10")}
    assert pr_gate.changed_paths(before, after) == []


def test_an_utterly_unchanged_run_is_not_worth_a_pr():
    same = {p: "content" for p in pr_gate.WATCHED}
    assert pr_gate.changed_paths(same, dict(same)) == []


# --- over-suppression: every one of these MUST open a PR -------------------

def test_a_new_queue_row_is_worth_a_pr_even_though_the_stamp_also_moved():
    """The regression that would matter: the stamp always moves, so a gate
    that keyed on 'did candidates.yaml change' inverted would hide the row."""
    before = {"candidates.yaml": queue("2026-09-09")}
    after = {"candidates.yaml": queue("2026-09-10", "- {name: A}\n- {name: B}\n")}
    assert pr_gate.changed_paths(before, after) == ["candidates.yaml"]


def test_a_readme_table_change_is_worth_a_pr_even_though_the_badge_also_moved():
    before = {"README.md": readme("2026-09-09")}
    after = {"README.md": readme("2026-09-10", "| A | 7 |\n| B | 9 |\n")}
    assert pr_gate.changed_paths(before, after) == ["README.md"]


def test_a_promoted_model_is_worth_a_pr():
    assert pr_gate.changed_paths({"models.yaml": "a"}, {"models.yaml": "b"}) \
        == ["models.yaml"]


def test_the_models_json_stamp_alone_is_not_worth_a_pr():
    """models.json republishes the same run date as a top-level field.

    Missed on the first pass: the gate then fired on a simulated no-op run
    and named models.json as the substantive change. Wrong in the safe
    direction — a spurious PR, not a swallowed release — but it defeats the
    whole gate, which is why all three stamp carriers are now pinned below.
    """
    before = {"models.json": '{\n  "generated": "2026-09-09",\n  "count": 55\n}'}
    after = {"models.json": '{\n  "generated": "2026-09-10",\n  "count": 55\n}'}
    assert pr_gate.changed_paths(before, after) == []


def test_the_json_stamp_mask_leaves_the_rest_of_the_document():
    text = '{\n  "generated": "2026-09-10",\n  "count": 55\n}'
    masked = pr_gate.mask("models.json", text)
    assert "2026-09-10" not in masked and '"count": 55' in masked


def test_the_json_stamp_mask_matches_what_render_json_actually_writes(tmp_path):
    """The third pin. render_json builds the stamp through the same
    load_generated() the README badge uses."""
    import render_json
    doc = render_json.build([], {}, {}, "2026-09-10")
    import json
    line = [l for l in json.dumps(doc, indent=2).splitlines() if '"generated"' in l]
    assert len(line) == 1
    assert pr_gate.mask("models.json", line[0]).strip() == '"generated": <run date>'


def test_a_models_json_content_change_is_worth_a_pr():
    """models.json carries the AA index and arena rank the README rounds off,
    so it can move on its own."""
    assert pr_gate.changed_paths({"models.json": "[]"}, {"models.json": '[{}]'}) \
        == ["models.json"]


def test_a_file_that_did_not_exist_before_is_a_change():
    assert pr_gate.changed_paths({}, {"models.json": "[]"}) == ["models.json"]


def test_a_deleted_file_is_a_change():
    assert pr_gate.changed_paths({"models.json": "[]"}, {}) == ["models.json"]


def test_every_watched_path_is_reported_not_just_the_first():
    before = {p: "a" for p in pr_gate.WATCHED}
    after = {p: "b" for p in pr_gate.WATCHED}
    assert pr_gate.changed_paths(before, after) == list(pr_gate.WATCHED)


# --- what is deliberately NOT watched --------------------------------------

def test_the_scraped_sidecars_are_not_watched():
    """arena rows carry vote counts and CIs that differ on every scrape.

    Watching them would gate on nothing. Their churn reaches a PR exactly
    when it changes a rendered file, which is what the watched set covers.
    """
    assert "arena_agent_rankings.yaml" not in pr_gate.WATCHED
    assert "aa_scores.yaml" not in pr_gate.WATCHED


# --- the masks must not eat more than their own line -----------------------

def test_the_stamp_mask_takes_the_stamp_line_and_nothing_else():
    masked = pr_gate.mask("candidates.yaml", queue("2026-09-10"))
    assert "2026-09-10" not in masked
    assert "# header" in masked and "- {name: A}" in masked


def test_the_badge_mask_takes_the_badge_line_and_nothing_else():
    masked = pr_gate.mask("README.md", readme("2026-09-10"))
    assert "2026-09-10" not in masked
    assert "validate" in masked and "| A | 7 |" in masked


def test_the_badge_mask_does_not_touch_the_other_shields():
    """It is anchored on the badge's own markdown, not on 'img.shields.io'."""
    masked = pr_gate.mask("README.md", readme("2026-09-10"))
    assert masked.count("img.shields.io") == 1


def test_a_generated_key_nested_under_a_row_is_not_mistaken_for_the_stamp():
    """The mask is anchored at column 0; an indented key is row data."""
    text = "generated: '2026-09-10'\nmodels:\n- name: A\n  generated: '2026-01-01'\n"
    assert "2026-01-01" in pr_gate.mask("candidates.yaml", text)


def test_models_yaml_is_compared_verbatim():
    """Only the two files carrying a run stamp get masked at all."""
    text = "generated: '2026-09-10'\n"
    assert pr_gate.mask("models.yaml", text) == text


# --- the label the mask is pinned to must be the one the renderer writes ---

def test_the_badge_mask_matches_what_render_readme_actually_emits():
    """If the shield is ever relabelled and this pattern is not, the gate
    silently starts treating the daily date bump as substantive again."""
    import render_readme
    line = [l for l in render_readme.badges(7, "2026-09-10").splitlines()
            if "index%20updated" in l]
    assert len(line) == 1
    assert pr_gate.mask("README.md", line[0]) == "<freshness badge>"


def test_the_stamp_mask_matches_what_write_candidates_actually_writes(tmp_path):
    """Same pin on the other side: discover.write_candidates owns the key."""
    import discover
    out = tmp_path / "candidates.yaml"
    discover.write_candidates(out, [{"name": "A"}])
    masked = pr_gate.mask("candidates.yaml", out.read_text())
    assert "generated: <run date>" in masked


# --- the CLI the workflow calls --------------------------------------------

def test_cli_prints_a_bare_boolean_on_stdout():
    """The workflow reads stdout straight into a step output, so the
    explanatory line has to go to stderr or it would corrupt the value."""
    proc = subprocess.run(
        [sys.executable, str(Path(pr_gate.__file__))],
        capture_output=True, text=True,
        cwd=str(Path(pr_gate.__file__).resolve().parent.parent))
    assert proc.returncode == 0
    assert proc.stdout.strip() in ("true", "false")


# --- the workflow step that consumes it ------------------------------------

def _gate_step():
    import yaml
    wf = yaml.safe_load(
        (Path(__file__).resolve().parent.parent
         / ".github" / "workflows" / "discover.yml").read_text())
    steps = wf["jobs"]["discover"]["steps"]
    return next(s for s in steps if s.get("id") == "worth"), steps


def test_the_pr_step_is_gated_on_the_gate():
    _, steps = _gate_step()
    pr = next(s for s in steps
              if str(s.get("uses", "")).startswith("peter-evans/create-pull-request"))
    assert pr["if"] == "steps.worth.outputs.changed == 'true'"


def test_the_gate_step_assigns_on_its_own_line_so_a_crash_fails_the_job():
    """`echo "changed=$(cmd)"` swallows cmd's exit status even under bash -e.

    Written that way, a crashed gate writes a bare `changed=`, the step
    exits 0, and the PR step's `if` reads it as false — silently suppressing
    a release, which is the one failure this whole gate exists to prevent.
    A bare assignment propagates the status; keep it on its own line.
    """
    step, _ = _gate_step()
    run = step["run"]
    assert "changed=$(python scripts/pr_gate.py)" in run
    assert 'echo "changed=$(python' not in run


def test_the_gate_step_rejects_a_value_that_is_not_true_or_false():
    """Belt to the assignment's braces: garbage on stdout must not read as
    false. Anything unrecognised fails the job instead."""
    step, _ = _gate_step()
    assert "exit 1" in step["run"]
    assert "true|false)" in step["run"]
