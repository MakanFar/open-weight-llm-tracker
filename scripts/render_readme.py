#!/usr/bin/env python3
"""
Render models.yaml into the README.md table. Never hand-edit the table —
edit the YAML and re-run this.

  python scripts/render_readme.py
"""
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
README = ROOT / "README.md"

START = "<!-- MODELS_TABLE_START -->"
END = "<!-- MODELS_TABLE_END -->"


def human_params(total, active, arch):
    if arch == "moe":
        return f"{total:g}B ({active:g}B active)"
    return f"{total:g}B"


def human_ctx(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}K"
    return str(n)


def commercial_badge(v):
    return {True: "Yes", False: "No", "conditional": "Conditional"}.get(v, str(v))


def build_table(models):
    # newest first, then by size
    models = sorted(
        models,
        key=lambda m: (m.get("release_date") or date.min, m.get("params_total_b", 0)),
        reverse=True,
    )
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Benchmark | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for m in models:
        b = m.get("benchmark", {})
        name = m["name"]
        if m.get("weights_url"):
            name = f"[{name}]({m['weights_url']})"
        bench = f"{b.get('name','?')} {b.get('score','?')}"
        rows.append(
            f"| {name} | {m['developer']} | {m['release_date']} | "
            f"{human_params(m['params_total_b'], m['params_active_b'], m['architecture'])} | "
            f"{human_ctx(m['context_window'])} | {m['modality']} | {bench} | "
            f"`{m['license']}` | {commercial_badge(m['commercial_use'])} |"
        )
    return "\n".join(rows)


def main():
    doc = yaml.safe_load(DATA.read_text())
    table = build_table(doc["models"])
    n = len(doc["models"])

    body = (
        "# Open-Weight & Open-Source LLM Tracker\n\n"
        f"A curated, machine-readable index of open-weight LLMs — parameter count, "
        f"context window, an anchor benchmark, and license clarity. "
        f"Currently tracking **{n} models**.\n\n"
        "Data lives in [`models.yaml`](models.yaml) (the source of truth). This table "
        "is generated — do not edit it by hand. See [SCHEMA.md](SCHEMA.md) for fields and "
        "[CONTRIBUTING.md](CONTRIBUTING.md) to add a model.\n\n"
        "> **Benchmark caveat:** the benchmark column mixes vendor-reported and "
        "third-party numbers (see each row's `benchmark.source` in the YAML). Anchor to a "
        "single leaderboard before relying on it for comparisons.\n\n"
        f"{START}\n{table}\n{END}\n\n"
        "## Regenerate\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python scripts/validate.py        # check the data\n"
        "python scripts/pull_hf.py         # (optional) auto-fill from Hugging Face\n"
        "python scripts/render_readme.py   # rebuild this table\n"
        "```\n"
    )
    README.write_text(body)
    print(f"Rendered README.md with {n} models.")


if __name__ == "__main__":
    main()
