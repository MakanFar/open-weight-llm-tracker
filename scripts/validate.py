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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"

REQUIRED = [
    "name", "developer", "release_date", "params_total_b", "params_active_b",
    "architecture", "context_window", "modality", "license", "commercial_use",
]

# Extend this allowlist as you add models — keeps license strings consistent.
LICENSES = {
    "apache-2.0", "mit", "bsd-3-clause",
    "llama-3.1-community", "llama-3.3-community", "llama-4-community",
    "qwen", "gemma", "deepseek",
    "cc-by-nc-4.0", "cc-by-4.0", "kimi-k3","minimax-community"
}
ARCH = {"dense", "moe"}
MODALITY = {"text", "vision-language", "multimodal"}
COMMERCIAL = {True, False, "conditional"}


def main():
    doc = yaml.safe_load(DATA.read_text())
    models = doc.get("models", [])
    errors = []
    seen = set()

    for i, m in enumerate(models):
        tag = m.get("name", f"<index {i}>")

        for f in REQUIRED:
            if m.get(f) in (None, ""):
                errors.append(f"[{tag}] missing required field: {f}")

        # no duplicate models
        key = (m.get("name"), m.get("hf_repo"))
        if key in seen:
            errors.append(f"[{tag}] duplicate entry")
        seen.add(key)

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

    if errors:
        print(f"VALIDATION FAILED — {len(errors)} problem(s):\n")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    print(f"OK — {len(models)} models, no problems found.")


if __name__ == "__main__":
    main()
