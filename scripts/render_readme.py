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
LEADERBOARD = ROOT / "leaderboard_scores.yaml"
ARENA = ROOT / "arena_agent_rankings.yaml"

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


def load_leaderboard(path=LEADERBOARD):
    """{lower_repo: mmlu_float}. {} on missing/malformed."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    out = {}
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if isinstance(entry, dict) and isinstance(entry.get("mmlu"), (int, float)):
                out[str(repo).lower()] = float(entry["mmlu"])
    return out


def load_arena_ranks(path=ARENA):
    """{lower_repo: rank_int}. {} on missing/malformed."""
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    out = {}
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict) and r.get("resolved_repo") and isinstance(r.get("rank"), int):
                out[str(r["resolved_repo"]).lower()] = r["rank"]
    return out


def mmlu_cell(model, lb):
    repo = (model.get("hf_repo") or "").lower()
    if repo in lb:
        return str(lb[repo])
    score = (model.get("benchmark") or {}).get("score")
    return f"{score}*" if isinstance(score, (int, float)) else "?"


def arena_cell(model, ranks):
    return str(ranks.get((model.get("hf_repo") or "").lower(), "—"))


def build_table(models, lb, ranks):
    # newest first, then by size
    models = sorted(
        models,
        key=lambda m: (m.get("release_date") or date.min, m.get("params_total_b", 0)),
        reverse=True,
    )
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Arena | MMLU | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for m in models:
        name = m["name"]
        if m.get("weights_url"):
            name = f"[{name}]({m['weights_url']})"
        rows.append(
            f"| {name} | {m['developer']} | {m['release_date']} | "
            f"{human_params(m['params_total_b'], m['params_active_b'], m['architecture'])} | "
            f"{human_ctx(m['context_window'])} | {m['modality']} | "
            f"{arena_cell(m, ranks)} | {mmlu_cell(m, lb)} | "
            f"`{m['license']}` | {commercial_badge(m['commercial_use'])} |"
        )
    return "\n".join(rows)


def main():
    doc = yaml.safe_load(DATA.read_text())
    lb = load_leaderboard()
    ranks = load_arena_ranks()
    table = build_table(doc["models"], lb, ranks)
    n = len(doc["models"])

    body = (
        "# Open-Weight & Open-Source LLM Tracker\n\n"
        f"A curated, machine-readable index of open-weight LLMs — parameter count, "
        f"context window, an anchor benchmark, and license clarity. "
        f"Currently tracking **{n} models**.\n\n"
        "Data lives in [`models.yaml`](models.yaml) (the source of truth). This table "
        "is generated — do not edit it by hand. See [SCHEMA.md](SCHEMA.md) for fields and "
        "[CONTRIBUTING.md](CONTRIBUTING.md) to add a model.\n\n"
        "> **Columns:** **MMLU** is from the HF Open LLM Leaderboard where "
        "available; values marked `*` fall back to a vendor/manual figure and are "
        "not harness-comparable. **Arena** is the Agent Arena rank (`—` = not "
        "currently ranked). See each row's `benchmark.source` in the YAML.\n\n"
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
