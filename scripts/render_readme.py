#!/usr/bin/env python3
"""
Render README.md from models.yaml.

NOTE: this rewrites the WHOLE file, not just the region between the
MODELS_TABLE markers — the surrounding prose is the `body` string in main()
below. Hand-edits to README.md anywhere are silently reverted on the next
run (and the validate workflow fails on the resulting diff), so prose
changes belong here.

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
        "python scripts/pull_hf.py         # (optional) auto-fill fields from Hugging Face\n"
        "python scripts/render_readme.py   # rebuild this table\n"
        "```\n\n"
        "## Staying current (automatic discovery)\n\n"
        "Two scripts feed the review queue in [`candidates.yaml`](candidates.yaml). "
        "Neither ever edits `models.yaml` directly.\n\n"
        "[`scripts/discover.py`](scripts/discover.py) sweeps an allowlist of "
        "organizations — one Hugging Face query per org — rather than scanning all of "
        "HF by recency. Sorting the whole Hub by upload date returns finetunes and "
        "quantizations, essentially never a frontier release. **Adding an org to "
        "`ORG_ALLOWLIST` in `scripts/discover.py` is how the tracker gains coverage.** "
        "It skips quantizations/adapters/merges, dedups against `models.yaml`, and "
        "writes new rows with fields pre-filled.\n\n"
        "[`scripts/pull_arena.py`](scripts/pull_arena.py) scrapes the "
        "[arena.ai](https://arena.ai/leaderboard/agent) leaderboard and resolves each "
        "ranked model to a Hugging Face repo. Arena rank then orders the review queue, "
        "so the models people actually use are reviewed first.\n\n"
        "**Open-weight status comes from whether weights actually resolve on Hugging "
        "Face** — not from a vendor's name and not from a leaderboard's license label. "
        "A model is open-weight if and only if a public weights repo was found for it. "
        "See [SCHEMA.md](SCHEMA.md) for the discovery-only fields, including "
        "`needs_hf_repo`, which flags an inexact name match for a human to confirm.\n\n"
        "The `discover-models` GitHub Action runs it weekly and opens a **pull request** "
        "with the new candidates — review the PR, fill the `TODO` fields (active params, "
        "benchmark, commercial-use), move approved rows into `models.yaml`, and merge.\n"
    )
    README.write_text(body)
    print(f"Rendered README.md with {n} models.")


if __name__ == "__main__":
    main()
