#!/usr/bin/env python3
"""
Validate models.yaml against the schema. Run in CI on every PR.
Exits non-zero if anything is wrong.

  python scripts/validate.py
"""
import re
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names  # noqa: E402  (names.py imports only `re`)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"

# How far the vendor's published total may sit from the measured tensor count
# and still plausibly describe the same model. Every real gap observed is under
# 4%; 25% leaves enormous room while still catching a figure copied off another
# variant of the family. Mirrors enrich._TOTAL_MATCH_TOLERANCE, which is the
# tighter capture-time guard — this one only has to catch hand edits.
STATED_TOTAL_TOLERANCE = 0.25

REQUIRED = [
    "name", "developer", "release_date", "params_total_b", "params_active_b",
    "architecture", "context_window", "modality", "license", "commercial_use",
]

# Extend this allowlist as you add models — keeps license strings consistent.
LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama-2-community", "llama-3-community",
    "llama-3.1-community", "llama-3.2-community",
    "llama-3.3-community", "llama-4-community",
    "qwen", "gemma", "deepseek",
    "cc-by-nc-4.0", "cc-by-4.0", "kimi-k3", "minimax-community",
}
ARCH = {"dense", "moe"}
MODALITY = {"text", "vision-language", "multimodal"}
COMMERCIAL = {True, False, "conditional"}


class SchemaError(str):
    """A validator complaint that remembers which field it is about.

    A plain `str` subclass on purpose: every consumer treats these as
    strings — main()'s print loop, the `any("release_date" in e ...)` checks
    in tests, classify's f-string prefix — and subclassing keeps all of that
    working untouched while adding one attribute.

    The attribute exists so classify.review_reasons can suppress a validator
    complaint that duplicates a reason missing_vitals already gave, WITHOUT
    matching on message text. Ten gated rows carried both
    "gated-repo-no-access" and "schema-invalid: context_window must be a
    positive integer"; message matching would have re-broken the moment
    anyone reworded a message.

    Do not let one of these reach yaml.safe_dump — a str subclass raises
    RepresenterError. review_reasons returns plain strs for exactly that
    reason.
    """

    def __new__(cls, field, message):
        obj = super().__new__(cls, message)
        obj.field = field
        return obj


def identity_errors(models):
    """Reject two rows whose hf_repos name the same weights.

    render_readme joins AA scores and arena ranks by repo identity when the
    exact repo string misses. Two rows sharing an identity would both claim the
    same sidecar entry, so the renderer's guard would drop it and BOTH rows
    would lose their number. It is also a plain duplicate: CLAUDE.md allows one
    row per model, a family flagship or distinct sizes, never both.
    """
    seen = {}
    errors = []
    for m in models:
        repo = m.get("hf_repo")
        if not repo:
            continue
        identity = names.repo_identity(repo)
        if not identity:
            continue
        if identity in seen:
            errors.append(
                f"[{m.get('name', repo)}] hf_repo '{repo}' names the same model "
                f"as '{seen[identity]}' (shared identity '{identity}') — "
                f"keep one row per model")
        else:
            seen[identity] = repo
    return errors


def row_errors(m, tag=None):
    """Every schema problem with a single row. [] means the row is clean.

    This is the per-row subset of what main() checks on every entry in
    models.yaml — everything except the duplicate-entry check, which needs
    the `seen` state main() accumulates across rows and so cannot live in a
    single-row function. Kept separate from main() so other code (classify.py's
    promotion gate) can run the identical checks on a row that is not yet, and
    may never be, in models.yaml — without reimplementing the rules and
    risking them drifting apart from what CI actually enforces.
    """
    if tag is None:
        tag = m.get("name") or m.get("hf_repo") or "<unnamed>"
    errors = []

    # Absent fields are reported ONCE. Every check below this loop asks
    # "is the value acceptable", which is a question with no meaning for a
    # field nobody set: a null modality used to collect both "missing
    # required field: modality" and "modality must be one of {...}", and the
    # second told a reviewer nothing. `in (None, "")` rather than a
    # truthiness test on purpose — commercial_use: false and
    # params_total_b: 0 are falsy but present, and must still be judged on
    # their merits below.
    missing = {f for f in REQUIRED if m.get(f) in (None, "")}
    for f in REQUIRED:
        if f in missing:
            errors.append(SchemaError(f, f"[{tag}] missing required field: {f}"))

    if "architecture" not in missing and m.get("architecture") not in ARCH:
        errors.append(SchemaError(
            "architecture", f"[{tag}] architecture must be one of {ARCH}"))
    if "modality" not in missing and m.get("modality") not in MODALITY:
        errors.append(SchemaError(
            "modality", f"[{tag}] modality must be one of {MODALITY}"))
    if "commercial_use" not in missing and m.get("commercial_use") not in COMMERCIAL:
        errors.append(SchemaError(
            "commercial_use",
            f"[{tag}] commercial_use must be true/false/conditional"))

    # Optional (not in REQUIRED — plenty of legitimate rows omit it), but
    # when present it gates render_readme.commercial_badge's trailing `?`
    # marker: that marker is the only visible signal that commercial_use was
    # a licence-tag guess, never actually read by a human. A string like
    # "no" is truthy, so an unchecked isinstance-free `if` would render it
    # as verified — publishing an unverified legal claim as a checked one.
    cuv = m.get("commercial_use_verified")
    if cuv is not None and not isinstance(cuv, bool):
        errors.append(SchemaError(
            "commercial_use_verified",
            f"[{tag}] commercial_use_verified must be true/false, got {cuv!r}"))
    if "license" not in missing and m.get("license") not in LICENSES:
        errors.append(SchemaError(
            "license",
            f"[{tag}] license '{m.get('license')}' not in allowlist "
            f"(add it to validate.py if intentional)"))

    # Numeric fields must actually BE numbers, not merely present. Presence
    # is covered by the REQUIRED loop above; a hand-edited candidates.yaml
    # row can carry a present-but-wrong-typed value like "700B" (units left
    # on) that sails through a bare `m.get(f) in (None, "")` presence check.
    # Every downstream numeric comparison below already guards with
    # isinstance and so silently no-ops on a string — nothing here would
    # ever complain — and render_readme.human_params does f"{total:g}",
    # which raises ValueError on a str. bool is excluded because it is a
    # subclass of int in Python (isinstance(True, int) is True), so a bare
    # isinstance(x, (int, float)) would wave a stray `true`/`false` through.
    pt, pa = m.get("params_total_b"), m.get("params_active_b")
    for field, val in (("params_total_b", pt), ("params_active_b", pa)):
        if val not in (None, "") and (isinstance(val, bool)
                                       or not isinstance(val, (int, float))):
            errors.append(SchemaError(
                field, f"[{tag}] {field} must be a number, got {val!r}"))

    # params_total_stated_b is the figure the VENDOR publishes; params_total_b
    # is the measured tensor count. Two names for two quantities, so a row can
    # carry both without appearing to argue with itself. Optional, and never
    # invented: most models publish no distinct headline figure, and absence
    # is the normal case.
    #
    # They still describe the SAME model, though. Cards quote a rounded
    # headline while safetensors counts every tensor — GLM-5 says 744B against
    # 753.9B, DeepSeek-V3 says 671B against 684.5B — so a few percent apart is
    # expected and 25% is far outside anything observed. A gap that wide means
    # the figure was copied off a different variant of the family, which is
    # exactly what enrich's capture-time tie-break exists to prevent; without
    # this check a hand edit could reintroduce it with nothing to complain.
    pts = m.get("params_total_stated_b")
    if pts not in (None, ""):
        if isinstance(pts, bool) or not isinstance(pts, (int, float)):
            errors.append(SchemaError(
                "params_total_stated_b",
                f"[{tag}] params_total_stated_b must be a number, got {pts!r}"))
        elif pts <= 0:
            errors.append(SchemaError(
                "params_total_stated_b",
                f"[{tag}] params_total_stated_b must be positive, got {pts!r}"))
        elif isinstance(pt, (int, float)) and not isinstance(pt, bool) and pt > 0 \
                and abs(pts - pt) / pt > STATED_TOTAL_TOLERANCE:
            errors.append(SchemaError(
                "params_total_stated_b",
                f"[{tag}] params_total_stated_b ({pts}) is more than "
                f"{STATED_TOTAL_TOLERANCE:.0%} from params_total_b ({pt}) — "
                f"is it the figure for a different variant?"))

    # MoE sanity: active <= total
    if isinstance(pt, (int, float)) and isinstance(pa, (int, float)) and pa > pt:
        errors.append(SchemaError(
            "params_active_b",
            f"[{tag}] params_active_b ({pa}) > params_total_b ({pt})"))

    # dense models: active should equal total
    if m.get("architecture") == "dense" and pt != pa:
        errors.append(SchemaError(
            "params_active_b",
            f"[{tag}] dense model should have params_active_b == params_total_b"))

    # date format + not in the future
    rd = m.get("release_date")
    if "release_date" not in missing:
        if not isinstance(rd, date):
            errors.append(SchemaError(
                "release_date", f"[{tag}] release_date must be YYYY-MM-DD"))
        elif rd > date.today():
            errors.append(SchemaError(
                "release_date", f"[{tag}] release_date is in the future: {rd}"))

    ctx = m.get("context_window")
    if "context_window" not in missing and (not isinstance(ctx, int) or ctx <= 0):
        errors.append(SchemaError(
            "context_window", f"[{tag}] context_window must be a positive integer"))

    return errors


def main():
    doc = yaml.safe_load(DATA.read_text())
    models = doc.get("models", [])
    errors = []
    seen = set()

    for i, m in enumerate(models):
        tag = m.get("name", f"<index {i}>")

        errors.extend(row_errors(m, tag=tag))

        # no duplicate models
        key = (m.get("name"), m.get("hf_repo"))
        if key in seen:
            errors.append(f"[{tag}] duplicate entry")
        seen.add(key)

    errors.extend(identity_errors(models))

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} problem(s):\n")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    print(f"OK — {len(models)} models, no problems found.")


if __name__ == "__main__":
    main()
