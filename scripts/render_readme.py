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
import re
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
    """{lower_repo: {"metric": str, "score": float}}. {} on missing/malformed.

    Both keys are required: a score whose metric is unknown cannot be placed,
    because MMLU and MMLU-PRO go in different columns.
    """
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    out = {}
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if not isinstance(entry, dict):
                continue
            metric, score = entry.get("metric"), entry.get("score")
            if isinstance(metric, str) and isinstance(score, (int, float)):
                out[str(repo).lower()] = {"metric": metric.upper(),
                                          "score": float(score)}
    return out


_ORG_TAIL_WORDS = {
    "anthropic", "openai", "google", "meta", "alibaba", "deepseek", "moonshot",
    "z.ai", "zai", "minimax", "nvidia", "xai", "microsoft", "cohere", "mistral",
    "tencent", "xiaomi", "thinky", "ibm", "baidu", "ai2", "01.ai", "tii", "zhipu",
}


def _slug(text):
    """Lowercase, strip non-alphanumerics. Mirrors pull_arena.slug.

    Duplicated rather than imported: pull_arena pulls in requests/bs4 at module
    scope, and this renderer must stay offline and dependency-light.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _arena_display_name(display):
    """Strip leaderboard chrome: 'Kimi K3 Moonshot · Proprietary' -> 'Kimi K3'."""
    name = str(display).split("·")[0]
    name = re.sub(r"\([^)]*\)", " ", name)
    tokens = re.sub(r"\s+", " ", name).strip().split(" ")
    if len(tokens) > 1 and tokens[-1].lower() in _ORG_TAIL_WORDS:
        tokens = tokens[:-1]
    return " ".join(tokens)


def load_arena_ranks(path=ARENA):
    """{"repos": {lower_repo: rank}, "names": {name_slug: rank}}.

    Two indexes, because a rank and a weights repo are separate facts. HF
    resolution fails transiently — one rate-limited search writes
    resolved_repo: null — and a rank already scraped should not vanish from the
    table because of it. Open-weight status still comes only from resolution;
    this index decides where a number is printed, nothing more.
    """
    empty = {"repos": {}, "names": {}}
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return empty
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return empty

    repos, names = {}, {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("rank"), int):
            continue
        if r.get("resolved_repo"):
            repos[str(r["resolved_repo"]).lower()] = r["rank"]
        if r.get("model"):
            key = _slug(_arena_display_name(r["model"]))
            if key:
                names.setdefault(key, r["rank"])
    return {"repos": repos, "names": names}


def _leaderboard_score(model, lb, metric):
    entry = lb.get((model.get("hf_repo") or "").lower())
    if isinstance(entry, dict) and entry.get("metric") == metric:
        return entry["score"]
    return None


def mmlu_cell(model, lb):
    """Plain MMLU: leaderboard where available, else the vendor figure with *."""
    score = _leaderboard_score(model, lb, "MMLU")
    if score is not None:
        return str(score)
    score = (model.get("benchmark") or {}).get("score")
    return f"{score}*" if isinstance(score, (int, float)) else "?"


def mmlu_pro_cell(model, lb):
    """MMLU-PRO: leaderboard only.

    Never falls back to models.yaml benchmark.score — that field is MMLU, and
    MMLU-PRO is a much harsher scale (~40 where MMLU reads ~80). Showing one
    in the other's column would invent a comparison that does not exist.
    """
    score = _leaderboard_score(model, lb, "MMLU-PRO")
    return str(score) if score is not None else "—"


def arena_cell(model, ranks):
    """Rank by resolved repo, else by model name, else '—'."""
    rank = ranks["repos"].get((model.get("hf_repo") or "").lower())
    if rank is None:
        rank = ranks["names"].get(_slug(model.get("name") or ""))
    return str(rank) if rank is not None else "—"


def build_table(models, lb, ranks):
    # newest first, then by size
    models = sorted(
        models,
        key=lambda m: (m.get("release_date") or date.min, m.get("params_total_b", 0)),
        reverse=True,
    )
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Arena | MMLU | MMLU-Pro | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for m in models:
        name = m["name"]
        if m.get("weights_url"):
            name = f"[{name}]({m['weights_url']})"
        rows.append(
            f"| {name} | {m['developer']} | {m['release_date']} | "
            f"{human_params(m['params_total_b'], m['params_active_b'], m['architecture'])} | "
            f"{human_ctx(m['context_window'])} | {m['modality']} | "
            f"{arena_cell(m, ranks)} | {mmlu_cell(m, lb)} | {mmlu_pro_cell(m, lb)} | "
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
        "> **Columns:** **MMLU** values marked `*` are vendor/manual figures from "
        "`models.yaml` and are not harness-comparable. **MMLU-Pro** comes from the "
        "HF Open LLM Leaderboard v2 (`—` = not listed there; it was archived, so "
        "most 2026 models are absent). The two benchmark columns are *not* "
        "comparable to each other — MMLU-Pro is a much harsher scale. **Arena** is "
        "the Agent Arena rank (`—` = not currently ranked). See each row's "
        "`benchmark.source` in the YAML.\n\n"
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
