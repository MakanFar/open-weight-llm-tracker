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

    for f in REQUIRED:
        if m.get(f) in (None, ""):
            errors.append(f"[{tag}] missing required field: {f}")

    if m.get("architecture") not in ARCH:
        errors.append(f"[{tag}] architecture must be one of {ARCH}")
    if m.get("modality") not in MODALITY:
        errors.append(f"[{tag}] modality must be one of {MODALITY}")
    if m.get("commercial_use") not in COMMERCIAL:
        errors.append(f"[{tag}] commercial_use must be true/false/conditional")
    if m.get("license") not in LICENSES:
        errors.append(f"[{tag}] license '{m.get('license')}' not in allowlist "
                      f"(add it to validate.py if intentional)")

    # MoE sanity: active <= total
    pt, pa = m.get("params_total_b"), m.get("params_active_b")
    if isinstance(pt, (int, float)) and isinstance(pa, (int, float)) and pa > pt:
        errors.append(f"[{tag}] params_active_b ({pa}) > params_total_b ({pt})")

    # dense models: active should equal total
    if m.get("architecture") == "dense" and pt != pa:
        errors.append(f"[{tag}] dense model should have params_active_b == params_total_b")

    # date format + not in the future
    rd = m.get("release_date")
    if not isinstance(rd, date):
        errors.append(f"[{tag}] release_date must be YYYY-MM-DD")
    elif rd > date.today():
        errors.append(f"[{tag}] release_date is in the future: {rd}")

    ctx = m.get("context_window")
    if not isinstance(ctx, int) or ctx <= 0:
        errors.append(f"[{tag}] context_window must be a positive integer")

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
