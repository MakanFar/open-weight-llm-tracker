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
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import names  # noqa: E402  (names.py imports only `re` — keeps this renderer offline)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
README = ROOT / "README.md"
ARENA = ROOT / "arena_agent_rankings.yaml"
AA = ROOT / "aa_scores.yaml"
CANDIDATES = ROOT / "candidates.yaml"

START = "<!-- MODELS_TABLE_START -->"
END = "<!-- MODELS_TABLE_END -->"

REPO_URL = "https://github.com/MakanFar/open-weight-llm-tracker"
RAW_URL = "https://raw.githubusercontent.com/MakanFar/open-weight-llm-tracker/main"


def load_generated(path=CANDIDATES):
    """ISO date of the last discovery run, or None if unstamped.

    Read from candidates.yaml (discover.write_candidates puts it there), never
    from the clock. validate.yml re-renders this README and fails on any diff
    against the committed one, so a date.today() in the rendered prose would
    turn green CI red on the first day nobody ran discovery — the badge would
    be claiming freshness it does not have, and breaking the build to do it.

    Returns None on a missing, empty or malformed file, exactly like the arena
    and AA loaders: the badge is then omitted and the render still completes.
    """
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    stamp = doc.get("generated") if isinstance(doc, dict) else None
    if isinstance(stamp, date):
        return stamp.isoformat()
    return stamp.strip() if isinstance(stamp, str) and stamp.strip() else None


def badges(n, generated):
    """Shields for CI, size and freshness. Hyphens in a shields label must be
    doubled, so a date renders as 2026--08--28."""
    out = [
        f"[![validate]({REPO_URL}/actions/workflows/validate.yml/badge.svg)]"
        f"({REPO_URL}/actions/workflows/validate.yml)",
        f"[![models](https://img.shields.io/badge/models-{n}-1f6feb)](models.yaml)",
    ]
    if generated:
        stamp = generated.replace("-", "--")
        out.append(
            f"[![last discovery run](https://img.shields.io/badge/"
            f"last%20discovery%20run-{stamp}-1f6feb)]"
            f"({REPO_URL}/actions/workflows/discover.yml)")
    out.append("[![code: MIT](https://img.shields.io/badge/code-MIT-3fb950)](LICENSE)")
    out.append("[![data: CC BY 4.0](https://img.shields.io/badge/"
               "data-CC--BY--4.0-3fb950)](LICENSE-DATA)")
    return "\n".join(out)


def display_total(model):
    """The parameter count to PRINT: the vendor's published figure if there is
    one, else the measured tensor count.

    These are two different quantities and models.yaml now names both.
    params_total_b is the measured count — reproducible, and the anchor
    everything else is checked against. params_total_stated_b is what the
    vendor publishes, and it is the number a reader arrives with: DeepSeek-V3
    is "671B" to everyone even though its tensors sum to 684.5B, because the
    measured figure includes the MTP module. Printing the measured one would
    make a correct table look wrong to more people than it informed.

    Falling back is not a compromise. Most models publish no distinct headline
    figure at all, so for those rows there is no competing claim and the
    measured count IS the published number.

    Guarded rather than trusted because candidates.yaml is hand-edited and
    row_errors runs on models.yaml only — human_params does f"{total:g}",
    which raises ValueError on a string.
    """
    stated = model.get("params_total_stated_b")
    if isinstance(stated, (int, float)) and not isinstance(stated, bool) \
            and stated > 0:
        return stated
    return model["params_total_b"]


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


def commercial_badge(model):
    """Commercial-use badge. Unverified values carry a trailing ?.

    A row promoted automatically carries a value inferred from the licence
    TAG, not from anyone reading the licence. Rendering it identically to a
    checked value would publish an unverified legal claim as a settled one.
    """
    label = {True: "Yes", False: "No",
             "conditional": "Conditional"}.get(model.get("commercial_use"),
                                               str(model.get("commercial_use")))
    return label if model.get("commercial_use_verified") else f"{label}?"


def _index_by_identity(pairs, identity_of=names.repo_identity):
    """{identity: value} from (repo_id, value) pairs, ambiguous keys dropped.

    An identity claimed by two entries with DIFFERENT values cannot be resolved
    — we would be guessing which one the tracked row means — so it is dropped
    and reported. Identical values are harmless and kept: two spellings of the
    same repo naturally agree.

    Values are compared with != rather than collected into a set, because an AA
    value is a dict and dicts are unhashable.

    identity_of is a seam for testing the guard; production always uses
    names.repo_identity.
    """
    grouped = {}
    for repo_id, value in pairs:
        grouped.setdefault(identity_of(repo_id), []).append((repo_id, value))

    out = {}
    for identity, claims in grouped.items():
        if not identity:
            continue
        first = claims[0][1]
        if any(value != first for _, value in claims[1:]):
            print(f"  ! identity {identity!r} claimed by "
                  f"{sorted(r for r, _ in claims)}; not joining")
            continue
        out[identity] = first
    return out


def load_arena_ranks_from_rows(rows):
    """Build the rank indexes from already-parsed arena rows.

    Split out from load_arena_ranks so tests can exercise the indexing without
    a file on disk.
    """
    repos, name_index, identity_pairs = {}, {}, {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("rank"), int):
            continue
        if r.get("resolved_repo"):
            repos[str(r["resolved_repo"]).lower()] = r["rank"]
            # setdefault keeps the BEST rank: rows arrive rank-ordered, and the
            # same model listed at several reasoning efforts legitimately
            # resolves to the same repo (see name_index below for the display-
            # name equivalent). A genuinely ambiguous identity — two DIFFERENT
            # repo strings — is still caught by _index_by_identity below.
            identity_pairs.setdefault(str(r["resolved_repo"]), r["rank"])
        if r.get("model"):
            # Full name first, vendor-stripped second. setdefault keeps the
            # BEST rank: rows arrive rank-ordered, and the same model listed at
            # several reasoning efforts legitimately repeats a display name.
            for key in names.display_identity(str(r["model"])):
                name_index.setdefault(key, r["rank"])
    return {"repos": repos, "names": name_index,
            "identities": _index_by_identity(identity_pairs.items())}


def load_arena_ranks(path=ARENA):
    """Rank indexes from arena_agent_rankings.yaml.

    Three indexes: `repos` (exact resolved_repo, lowercased), `names` (display
    name, for when resolution never happened at all), and `identities` (repo
    identity, for when arena resolved a different spelling of the repo than the
    one models.yaml carries — e.g. arena resolves the -NVFP4 mirror while the
    tracked row is the -BF16 release). All three are built from this file alone;
    aa_scores.yaml is never read here.

    `repos` and `names` exist because a rank and a weights repo are separate
    facts — HF resolution fails transiently, one rate-limited search writes
    resolved_repo: null, and a rank already scraped should not vanish from the
    table because of it. Open-weight status still comes only from resolution;
    these indexes decide where a number is printed, nothing more.
    """
    empty = {"repos": {}, "names": {}, "identities": {}}
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return empty
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return empty
    return load_arena_ranks_from_rows(rows)


def load_aa_scores_from_dict(scores, identity_of=names.repo_identity):
    """Build the AA indexes from an already-parsed `scores:` mapping.

    Split out from load_aa_scores so tests can exercise the indexing and the
    collision guard without a file on disk.
    """
    repos, pairs = {}, []
    if isinstance(scores, dict):
        for repo, entry in scores.items():
            if not isinstance(entry, dict):
                continue
            idx = entry.get("intelligence_index")
            if isinstance(idx, int) and not isinstance(idx, bool):
                value = {"index": idx, "variant": entry.get("variant")}
                repos[str(repo).lower()] = value
                pairs.append((str(repo), value))

    return {"repos": repos,
            "identities": _index_by_identity(pairs, identity_of)}


def load_aa_scores(path=AA):
    """AA indexes keyed by lowercased repo and by repo identity.

    Returns {"repos": {}, "identities": {}} on a missing, empty, or malformed
    file — never raises — so the render always completes and every row just
    falls through to aa_cell's '—'.
    """
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {"repos": {}, "identities": {}}
    scores = doc.get("scores") if isinstance(doc, dict) else None
    return load_aa_scores_from_dict(scores)


def aa_cell(model, aa):
    """Artificial Analysis Intelligence Index, or — when AA does not rate it.

    Exact repo first, then repo identity: AA and arena disagree about which
    repo string names a model (AA scored DeepSeek-V4-Flash-0731, arena resolved
    DeepSeek-V4-Flash), and an exact-only join prints — for a model both rate.

    There is deliberately no fallback to a manual figure: models.yaml no longer
    carries one, because MMLU (~86) and the AA index (~10-57) are different
    scales and sharing a column invited a comparison that does not exist.
    """
    repo = model.get("hf_repo") or ""
    entry = aa["repos"].get(repo.lower())
    if entry is None and repo:
        entry = aa["identities"].get(names.repo_identity(repo))
    return str(entry["index"]) if entry else "—"


def arena_cell(model, ranks):
    """Rank by resolved repo, then repo identity, then display name, else '—'."""
    repo = model.get("hf_repo") or ""
    rank = ranks["repos"].get(repo.lower())
    if rank is None and repo:
        rank = ranks["identities"].get(names.repo_identity(repo))
    if rank is None:
        for key in names.display_identity(str(model.get("name") or "")):
            rank = ranks["names"].get(key)
            if rank is not None:
                break
    return str(rank) if rank is not None else "—"


def build_table(models, aa, ranks):
    # newest first, then by size
    models = sorted(
        models,
        key=lambda m: (m.get("release_date") or date.min, m.get("params_total_b", 0)),
        reverse=True,
    )
    head = ("| Model | Developer | Released | Params | Context | Modality | "
            "Arena | AA Index | License | Commercial |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for m in models:
        name = m["name"]
        if m.get("weights_url"):
            name = f"[{name}]({m['weights_url']})"
        rows.append(
            f"| {name} | {m['developer']} | {m['release_date']} | "
            f"{human_params(display_total(m), m['params_active_b'], m['architecture'])} | "
            f"{human_ctx(m['context_window'])} | {m['modality']} | "
            f"{arena_cell(m, ranks)} | {aa_cell(m, aa)} | "
            f"`{m['license']}` | {commercial_badge(m)} |"
        )
    return "\n".join(rows)


def main():
    doc = yaml.safe_load(DATA.read_text())
    aa = load_aa_scores()
    ranks = load_arena_ranks()
    table = build_table(doc["models"], aa, ranks)
    n = len(doc["models"])

    body = (
        "# Open-Weight & Open-Source LLM Tracker\n\n"
        f"{badges(n, load_generated())}\n\n"
        f"**What it is** — a curated, machine-readable index of **{n} open-weight "
        "LLMs**: parameter count, context window, modality, licence, and an anchor "
        "benchmark. One row per model, newest first.\n\n"
        "**Why it's different** — a model is listed as open-weight only if a public "
        "weights repo actually resolves on Hugging Face. Never from the vendor's "
        "name, never from a leaderboard's licence label. New releases are found "
        "automatically every week, but each one arrives as a **pull request**: "
        "nothing reaches the table without a human merging it.\n\n"
        "**How to consume it** — fetch the data, don't scrape the table:\n\n"
        "```bash\n"
        f"curl -sL {RAW_URL}/models.json    # generated, with arena + AA joined in\n"
        f"curl -sL {RAW_URL}/models.yaml    # the source of truth\n"
        "```\n\n"
        "[`models.yaml`](models.yaml) is the source of truth; "
        "[`models.json`](models.json) and this table are generated from it — do not "
        "edit either by hand. See [SCHEMA.md](SCHEMA.md) for fields and "
        "[CONTRIBUTING.md](CONTRIBUTING.md) to add a model.\n\n"
        "> **Columns:** **AA Index** is the [Artificial Analysis Intelligence "
        "Index](https://artificialanalysis.ai/leaderboards/models?weights=open) — a 0–100 "
        "composite of agentic, coding, scientific-reasoning and general "
        "evaluations. `—` means Artificial Analysis does not currently rate that "
        "model; it drops older models, so coverage skews to recent releases. The "
        "index is re-weighted between versions, so values are not comparable "
        "across time. **Arena** is the rank on arena.ai's text leaderboard among "
        "open-weight models (`—` = not currently ranked). "
        "A trailing `?` on **Commercial** marks a value inferred from the "
        "licence tag and not yet checked against the licence text. "
        "**Params** is the figure the vendor publishes where there is one "
        "(`params_total_stated_b`), otherwise the measured tensor count "
        "(`params_total_b`) — the two differ by a few percent because a "
        "checkpoint carries tensors a headline figure leaves out, and "
        "[`models.json`](models.json) carries both.\n\n"
        f"{START}\n{table}\n{END}\n\n"
        "## Regenerate\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "python scripts/validate.py        # check the data\n"
        "python scripts/pull_hf.py         # (optional) auto-fill fields from Hugging Face\n"
        "python scripts/render_readme.py   # rebuild this table\n"
        "python scripts/render_json.py     # rebuild models.json\n"
        "```\n\n"
        "## Staying current (automatic discovery)\n\n"
        "Discovery runs weekly and classifies what it finds. A model that clears "
        "the notability bar (an Artificial Analysis score, an arena rank, or "
        "≥500k Hugging Face downloads) **and** has no missing vitals is "
        "**appended** to `models.yaml` automatically — never edited, reordered, or "
        "deleted, only added to. A notable model missing something (most often a "
        "missing context window, or a family already tracked under a different "
        "repo name) waits in [`candidates.yaml`](candidates.yaml) with a "
        "`needs_review` list explaining what is missing. An unremarkable model is "
        "dropped before either file sees it.\n\n"
        "[`scripts/discover.py`](scripts/discover.py) sweeps an allowlist of "
        "organizations — one Hugging Face query per org — rather than scanning all of "
        "HF by recency. Sorting the whole Hub by upload date returns finetunes and "
        "quantizations, essentially never a frontier release. **Adding an org to "
        "`ORG_ALLOWLIST` in `scripts/discover.py` is how the tracker gains coverage.** "
        "It skips quantizations/adapters/merges, dedups against `models.yaml`, and "
        "writes new rows with fields pre-filled.\n\n"
        "[`scripts/pull_arena.py`](scripts/pull_arena.py) scrapes the "
        "[arena.ai](https://arena.ai/leaderboard/text/overall?license=open-source) "
        "leaderboard and resolves each "
        "ranked model to a Hugging Face repo. Arena rank then orders the review queue, "
        "so the models people actually use are reviewed first.\n\n"
        "**Open-weight status comes from whether weights actually resolve on Hugging "
        "Face** — not from a vendor's name and not from a leaderboard's license label. "
        "A model is open-weight if and only if a public weights repo was found for it. "
        "See [SCHEMA.md](SCHEMA.md) for the discovery-only fields, including "
        "`needs_hf_repo`, which flags an inexact name match for a human to confirm.\n\n"
        "The `discover-models` GitHub Action runs this weekly and opens a **pull "
        "request** — nothing it does ever commits straight to `main`, and that PR "
        "is the human-in-the-loop approval step. A row in the table above marked "
        "with a trailing `?` on **Commercial** was auto-promoted with "
        "`commercial_use` inferred from the licence tag: check it against the "
        "licence text before merging. For a row left in `candidates.yaml`, fill "
        "the gap named in its `needs_review` list and the next run promotes it "
        "automatically.\n\n"
        "## License\n\n"
        "Split by what the thing is. The code — `scripts/`, `tests/`, the "
        "workflows — is [MIT](LICENSE). The data — `models.yaml`, "
        "`models.json`, `candidates.yaml`, `aa_scores.yaml`, "
        "`arena_agent_rankings.yaml` and the table above — is "
        "[CC BY 4.0](LICENSE-DATA): reuse it freely, credit this repository, "
        "and say if you changed it.\n\n"
        "That covers this compilation — the selection, verification and "
        "arrangement of the facts. **It grants nothing over the model weights "
        "themselves**, which stay under their vendors' own terms; the `license` "
        "column names those, and is the whole point of the index.\n"
    )
    README.write_text(body)
    print(f"Rendered README.md with {n} models.")


if __name__ == "__main__":
    main()
