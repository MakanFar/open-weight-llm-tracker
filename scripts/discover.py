#!/usr/bin/env python3
"""
Discover NEW open-weight LLMs, classify each one, and route it accordingly.
A row that clears the notability bar (classify.is_notable) with no missing
vitals (classify.missing_vitals) is APPENDED to models.yaml; everything else
that is notable waits in candidates.yaml with a `needs_review` list; anything
unremarkable is dropped before either file sees it. This never edits, reorders
or deletes a row already in models.yaml — only append_models() writes to it,
and only by adding to the end — and it never commits to main: the Action opens
a PR, which stays the human-in-the-loop approval step.

WHY AN ORG SWEEP, NOT A GLOBAL SCAN:
    This script used to sort ALL of Hugging Face by created_at and take the
    newest N. That population is dominated by finetunes and quantizations, so
    a frontier release essentially never appears in it. A run on 2026-07-21
    scanned 300 models and produced zero candidates (110 derivative, 189
    non-org, 1 blocked — every single one filtered).

    Now we issue one list_models(author=org) call per allowlisted org. That
    surfaced GLM-5.2, Kimi-K2.7, MiniMax-M3 and Gemma 4 immediately.

    Consequence: the org-follower probe that policed the unbounded query is
    gone. The allowlist IS the query.

WHY AN AGE WINDOW ON TOP OF THAT:
    limit=50 caps how many repos an org returns, not how old they are. Orgs
    that publish rarely have their 50 newest reaching back years, so the first
    sweeps staged 362 candidates dated 2023-07-11 to 2026-07-15 — the entire
    back-catalogue of 23 orgs, not new releases. --max-age-days bounds the
    sweep by created_at; repos with no created_at are kept, since an unknown
    date cannot prove staleness. The window applies to the org sweep only:
    arena rows are gated by leaderboard rank, which is its own relevance
    signal and has nothing to do with age.

THE QUEUE IS CUMULATIVE:
    candidates.yaml is rewritten wholesale every run, so anything refresh()
    does not return is deleted. Staged rows are therefore carried forward,
    minus any a human has promoted into models.yaml. They must NOT count as
    "already known" the way tracked models do: doing that made the sweep skip
    them, emit no replacement row, and let the rewrite empty the queue — so
    merging a candidates PR made the next run propose deleting all of it.

WHERE NEW ORGS COME FROM:
    An allowlist only finds what it already knows. scripts/pull_arena.py
    scans every leaderboard row for orgs it has no HF namespace mapping for
    and writes them to the `new_orgs` key of arena_agent_rankings.yaml; this
    script reprints them so a human can widen coverage by adding the org to
    ORG_ALLOWLIST below (and its namespace to HF_AUTHOR_HINTS in pull_arena).

    For the current list, read `new_orgs` in arena_agent_rankings.yaml — it is
    regenerated on every arena run. It is deliberately not restated here: an
    org named in this docstring is by definition one someone has since added
    to ORG_ALLOWLIST, at which point the example is false. The previous
    version named three orgs, two of which were already allowlisted and so
    could never appear in new_orgs at all.

ARENA MERGE:
    Recency alone does not tell you which models matter. scripts/pull_arena.py
    resolves leaderboard entries to HF repos and writes
    arena_agent_rankings.yaml; this script merges those resolved repos into
    the same candidate queue as the org sweep and sorts the queue so ranked
    models lead (best rank first), with unranked (org-sweep-only) candidates
    following, newest first. Arena is never load-bearing: a missing, empty,
    or malformed arena_agent_rankings.yaml just means the queue falls back to
    the org sweep alone.

Usage:
  pip install -r requirements.txt
  python scripts/discover.py                  # org sweep + arena merge
  python scripts/discover.py --min-params 3
  python scripts/discover.py --max-age-days 90   # tighter recency window
  python scripts/discover.py --max-age-days 0    # no age limit (back-catalogue)
  python scripts/discover.py --no-arena       # org sweep only, skip the arena merge
  HUGGINGFACE_TOKEN=hf_xxx python scripts/discover.py   # higher rate limits

NOTES / deliberate choices (the "don'ts"):
  - We do NOT exclude gated models — Llama & Gemma are gated; excluding them
    would drop the flagships. We keep them and flag for review.
  - safetensors.total is TOTAL params. For MoE we cannot infer active params
    from the API, so params_active_b is left equal to total and flagged TODO.
    architecture IS derived — config.json declares an expert count — but that
    count does not yield active params, so detecting MoE only sharpens the
    TODO, it does not close it.
  - Candidate rows carry NO benchmark field at all — not even a TODO stub.
    models.yaml has none either: the anchor number is the Artificial Analysis
    Intelligence Index, scraped by pull_aa.py into aa_scores.yaml and joined
    onto a row by hf_repo at render time, never hand-copied or stored here.
  - license tag is uploader-supplied; commercial_use is a *guess* to be checked.
"""
import argparse
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import yaml

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit("Install deps first:  pip install -r requirements.txt")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta
import enrich
import names
import classify

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"
CANDIDATES = ROOT / "candidates.yaml"
ARENA = ROOT / "arena_agent_rankings.yaml"
AA = ROOT / "aa_scores.yaml"

# How far back the org sweep reaches. Kept generous relative to the weekly
# schedule so a run that fails or is skipped does not create a coverage hole.
DEFAULT_MAX_AGE_DAYS = 180

# The orgs we sweep. This list IS the query — adding an org here is how the
# tracker gains coverage. pull_arena.py reports leaderboard orgs missing here.
ORG_ALLOWLIST = [
    "meta-llama", "Qwen", "deepseek-ai", "mistralai", "google", "microsoft",
    "CohereForAI", "CohereLabs", "ai21labs", "allenai", "nvidia", "01-ai",
    "tiiuae", "databricks", "HuggingFaceTB", "ibm-granite", "internlm",
    "THUDM", "zai-org", "moonshotai", "openai", "xai-org", "stabilityai",
    "MiniMaxAI", "XiaomiMiMo", "poolside", "thinkingmachines", "baidu",
    # Lowercase: HF namespaces are case-sensitive here and list_models(
    # author="Tencent") returns an EMPTY LIST rather than raising, so the
    # miscased entry that sat here swept nothing and reported nothing. The
    # empty_org warning in sweep_orgs exists to make that class of typo
    # visible instead of silent.
    "tencent",
]


def _rows_of(path):
    """The `models:` list of a YAML file, skipping anything malformed."""
    path = Path(path)
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        return []
    rows = doc.get("models")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("hf_repo")]


def tracked_repos(path=DATA):
    """Lowercased hf_repos already promoted into models.yaml.

    Deliberately does NOT read candidates.yaml. Staged rows are pending
    review, not settled decisions: counting them as "known" made the sweep
    skip them, produced no replacement row, and let the wholesale rewrite in
    write_candidates() empty the queue on the very next run.
    """
    return {r["hf_repo"].lower() for r in _rows_of(path)}


# Discovery-only fields. SCHEMA.md requires these stripped on promotion —
# validate.py checks models.yaml only, so they would otherwise leak in.
# needs_review is classify.route's own output (why a row waited rather than
# promoted); a promoted row has already cleared that bar, so the reason it
# once carried is stale and must not survive into models.yaml.
# family_collision_reviewed records a REVIEWER'S DECISION about a
# candidates.yaml collision, not a fact about the model. Once the row is in
# models.yaml the collision it answered is resolved and the marker is
# meaningless there, so it is stripped like needs_review.
PROMOTION_STRIP_FIELDS = ("discovered_via", "arena_rank", "aa_index",
                          "downloads", "needs_hf_repo", "resolution_confidence",
                          "needs_review", "family_collision_reviewed",
                          "gated_no_access")


def tracked_stems(path=DATA):
    """Family stems already present in models.yaml."""
    return {names.family_stem(r["hf_repo"]) for r in _rows_of(path)}


def promotion_row(candidate):
    """A candidate reshaped for models.yaml.

    commercial_use_verified is force-stamped False (an unconditional
    assignment, not a setdefault) because the value must reflect that no
    human has read the licence — a candidate arriving with
    commercial_use_verified already True (e.g. hand-edited in
    candidates.yaml from a licence-tag guess) must NOT promote as verified.
    render_readme.py marks an unverified value with a trailing `?` on the
    Commercial column, so a reader can tell an inferred claim from a checked
    one — this row just has to carry the correct value for it to key off.

    The auto-discovery boilerplate note (stamped by every candidate at
    creation, see hf_meta.AUTO_DISCOVERY_NOTE) is dropped rather than
    promoted: it is a to-do for the reviewer, not a fact about the model, and
    is false the instant the row is promoted. Matched by exact identity so a
    human's own "notes" survives untouched — only that literal sentence is
    ever removed. license_notes is a different field with a legitimate
    standing marker ("AUTO-DISCOVERED — verify license terms.") and is left
    alone.
    """
    row = {k: v for k, v in candidate.items() if k not in PROMOTION_STRIP_FIELDS}
    row["commercial_use_verified"] = False
    if row.get("notes") == hf_meta.AUTO_DISCOVERY_NOTE:
        del row["notes"]
    return row


class ModelsYamlShapeError(Exception):
    """models.yaml is not shaped the way append_models requires.

    Raised instead of writing anything. append_models concatenates text at
    EOF with no re-parse of what it wrote — that is deliberate (see its
    docstring) — but that only works if the LAST top-level key in the file
    really is `models:`. If the file ever ends with a different top-level
    key (e.g. a hand-added `sources:` block), a naive text-append lands the
    new row underneath that key instead: the file still parses, `models` is
    unchanged, and the promoted model silently vanishes with no exception
    and no change to validate.py's reported count. Refusing loudly here is
    the only thing that turns that into a visible failure instead of a
    silent data loss.
    """


def _assert_appendable_shape(text):
    """Refuse to append unless models.yaml is a mapping ending in `models: [...]`.

    append_models writes raw text onto the end of the file, so anything
    after that point in the real file would end up appearing to belong to
    whatever key trails it. Checking that `models` is LAST (not just
    present) is what keeps the append landing in the right place.
    """
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelsYamlShapeError(f"models.yaml is not valid YAML: {exc}") from exc

    if not isinstance(doc, dict) or not doc:
        raise ModelsYamlShapeError(
            "models.yaml does not parse as a non-empty mapping; refusing to "
            "append rather than risk corrupting it")

    last_key = list(doc.keys())[-1]
    if last_key != "models":
        raise ModelsYamlShapeError(
            f"models.yaml's last top-level key is {last_key!r}, not "
            "'models'; an EOF append would land under that key instead and "
            "silently vanish from the tracked list, so refusing to write")

    if not isinstance(doc["models"], list):
        raise ModelsYamlShapeError(
            "models.yaml's 'models' key is not a list; refusing to append")


def append_models(path, rows):
    """Append rows to models.yaml. Returns the count appended.

    APPEND-ONLY, and literally so: the existing file is not parsed and
    re-dumped, it is read as text and added to. Round-tripping through
    yaml.safe_dump would reflow every hand-curated row and drop the comment
    header, which is precisely the human ownership this file exists to hold.
    Before touching anything, _assert_appendable_shape checks (read-only)
    that the file's last top-level key really is `models:` — see
    ModelsYamlShapeError for why that precondition matters. This function
    still never re-dumps the rows it writes; the check only reads.

    Rows are dumped on their own (not wrapped in a {"models": rows} document)
    because PyYAML's block-sequence indent is 0 by default — a top-level dump
    would emit "- name: ..." flush against the margin, but every row already
    in the file sits indented two spaces under the `models:` key. Dumping the
    bare list and indenting each line by hand matches that existing
    convention instead of fighting it. Each row is dumped separately (rather
    than all at once) so a blank line can be inserted before every one of
    them, not just before the first — every row pair already in the file is
    separated by a blank line, including consecutive rows within a single
    append batch, and this matches that existing hand-formatting.

    The write itself is atomic: the new content is written to a temp file in
    the same directory and moved into place with os.replace(), which is
    atomic on POSIX. models.yaml is hand-curated source of truth with no
    automated backup; a plain write_text() truncates in place, so a process
    killed mid-write would leave it truncated with the appended rows (and
    possibly existing ones) gone.
    """
    rows = list(rows)
    if not rows:
        return 0

    existing = Path(path).read_text()
    _assert_appendable_shape(existing)

    blocks = []
    for row in rows:
        dumped = yaml.safe_dump([row], sort_keys=False, allow_unicode=True, width=100)
        blocks.append("".join("  " + line if line.strip() else line
                              for line in dumped.splitlines(keepends=True)))
    body = "\n".join(blocks)  # blank line between consecutive appended rows too
    if not existing.endswith("\n"):
        existing += "\n"
    new_text = existing + "\n" + body

    directory = Path(path).resolve().parent
    fd, tmp_name = tempfile.mkstemp(prefix=".models.yaml.", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(new_text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return len(rows)


def load_staged(path=CANDIDATES):
    """Candidate rows still sitting in the review queue."""
    return _rows_of(path)


def surviving_staged(staged, tracked):
    """Staged rows minus the ones a human has since promoted to models.yaml."""
    return [r for r in staged if r["hf_repo"].lower() not in tracked]


def sweep_orgs(api, orgs, min_params, known, get_json=hf_meta._http_get_json,
               get_text=enrich._http_get_text, max_age_days=None, today=None):
    """One list_models call per org. Returns (candidates, skip_counts).

    A failure on one org is logged and skipped — it never aborts the sweep.

    max_age_days bounds how far back the sweep reaches. `limit=50` is a size
    cap, not a recency filter: an org that publishes rarely has its 50 newest
    repos stretching back years, so without an age window the sweep stages the
    entire back-catalogue of every allowlisted org instead of what is new. A
    repo with no created_at is kept — an unknown date cannot prove staleness.
    Pass 0 or None to disable the window.

    Each candidate is enriched (enrich_row) before it is returned. This must
    happen here, not later in refresh(), because classify.route() judges
    completeness on the row it is handed — an MoE row would look permanently
    incomplete if the card-derived active-params figure were never fetched.
    """
    candidates = []
    skips = {"known": 0, "derivative": 0, "small": 0, "license": 0,
             "no_params": 0, "org_error": 0, "stale": 0, "empty_org": 0}
    seen = set(known)
    cutoff = None
    if max_age_days:
        cutoff = (today or date.today()) - timedelta(days=max_age_days)

    for org in orgs:
        try:
            # list_models() is a generator — it does not make the HTTP
            # request until iterated. Force it here with list(...) so the
            # request (and any rate-limit/5xx exception) happens inside the
            # try, not in the unguarded loop below.
            models = list(api.list_models(
                author=org,
                pipeline_tag="text-generation",
                sort="created_at",
                limit=50,
                expand=hf_meta.EXPAND,
            ))
        except Exception as exc:
            print(f"  ! {org}: {exc}")
            skips["org_error"] += 1
            continue

        # An allowlisted org that returns NO repos at all is a configuration
        # bug, not a quiet week: orgs are on the list precisely because they
        # publish. HF answers an unknown or miscased author with an empty
        # list rather than an error, so without this the sweep reports a
        # normal run while covering nothing — which is exactly what
        # "Tencent" (vs "tencent"), CohereForAI, THUDM and databricks did.
        # Distinct from zero CANDIDATES, which is ordinary: everything the
        # org published may be known, too small, or a quantization.
        #
        # The unfiltered re-query separates two very different findings that
        # look identical from here. thinkingmachines publishes six repos and
        # none carry a text-generation tag (Inkling is multimodal), so the
        # org is alive and the SWEEP is what cannot see it; calling that
        # "dead or misspelled" sends a maintainer hunting a typo that does
        # not exist. Costs one extra request per empty org, and only ever
        # for an org that already returned nothing.
        if not models:
            skips["empty_org"] += 1
            try:
                any_repo = list(api.list_models(author=org, limit=1))
            except Exception:
                any_repo = []
            if any_repo:
                print(f"  ! {org}: no text-generation repos, though the org "
                      f"publishes others — this sweep cannot see its models")
            else:
                print(f"  ! {org}: no repos on HF — dead or misspelled org?")
            continue

        for info in models:
            if info.id.lower() in seen:
                skips["known"] += 1
                continue
            if cutoff is not None:
                created = hf_meta.to_date(getattr(info, "created_at", None))
                if created is not None and created < cutoff:
                    skips["stale"] += 1
                    continue
            keep, reason = hf_meta.should_track(info, min_params)
            if not keep:
                skips[reason] += 1
                continue
            notes = {}
            ctx, arch = hf_meta.resolve_facts(info, get_json, notes=notes)
            row = hf_meta.candidate_from_repo(
                info, discovered_via=["org-sweep"], context_window=ctx,
                architecture=arch)
            if not ctx and notes.get("gated"):
                row["gated_no_access"] = True
            candidates.append(enrich_row(row, info, get_text, get_json))
            seen.add(info.id.lower())

    return candidates, skips


def load_arena(path=ARENA):
    """Read arena_agent_rankings.yaml. Returns (resolved_rows, new_orgs).

    Arena is never load-bearing: a missing, unreadable, unparsable, or
    wrong-shaped file yields ([], []) so the org sweep still produces
    candidates. "Wrong-shaped" covers syntactically valid YAML that isn't
    the mapping/list structure we expect (a top-level list or scalar,
    arena_agent/new_orgs not being lists, rows that aren't mappings) — those
    parse fine but would otherwise crash the .get() calls below.
    """
    path = Path(path)
    if not path.exists():
        return [], []
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        print(f"  ! arena file unreadable ({exc}); continuing without it")
        return [], []
    except OSError as exc:
        print(f"  ! arena file unreadable ({exc}); continuing without it")
        return [], []

    if not isinstance(doc, dict):
        print(f"  ! arena file malformed (expected a mapping, got "
              f"{type(doc).__name__}); continuing without it")
        return [], []

    raw_rows = doc.get("arena_agent")
    if raw_rows is not None and not isinstance(raw_rows, list):
        print(f"  ! arena file malformed (arena_agent is not a list, got "
              f"{type(raw_rows).__name__}); continuing without it")
        raw_rows = []
    rows = [r for r in (raw_rows or [])
            if isinstance(r, dict) and r.get("resolved_repo")]

    raw_new_orgs = doc.get("new_orgs")
    if raw_new_orgs is not None and not isinstance(raw_new_orgs, list):
        print(f"  ! arena file malformed (new_orgs is not a list, got "
              f"{type(raw_new_orgs).__name__}); continuing without it")
        raw_new_orgs = []

    return rows, list(raw_new_orgs or [])


def arena_candidates(api, rows, min_params, known, get_json=hf_meta._http_get_json,
                     get_text=enrich._http_get_text):
    """Build candidates from arena-resolved repos via the shared hf_meta path.

    Enriched the same way as sweep_orgs, and for the same reason: classify.py
    must see the card-derived active-params figure, not just what the HF API
    expand exposes.
    """
    out = []
    for row in rows:
        repo = row["resolved_repo"]
        if repo.lower() in known:
            continue
        try:
            info = api.model_info(repo, expand=hf_meta.EXPAND)
        except Exception as exc:
            print(f"  ! {repo}: {exc}")
            continue
        keep, reason = hf_meta.should_track(info, min_params)
        if not keep:
            # Name the rank being dropped. The filter is right to reject a
            # quantization, but the rank is real and the model may deserve a
            # row under its primary repo — this is how Nemotron 3 Ultra's
            # rank 42 vanished from the queue without a trace.
            print(f"  - {repo} skipped ({reason}; arena rank "
                  f"{row.get('rank')} dropped)")
            continue
        notes = {}
        ctx, arch = hf_meta.resolve_facts(info, get_json, notes=notes)
        candidate = hf_meta.candidate_from_repo(
            info, discovered_via=["arena"], arena_rank=row.get("rank"),
            needs_hf_repo=row.get("needs_hf_repo"),
            resolution_confidence=row.get("resolution_confidence"),
            context_window=ctx, architecture=arch)
        if not ctx and notes.get("gated"):
            candidate["gated_no_access"] = True
        out.append(enrich_row(candidate, info, get_text, get_json))
    return out


def enrich_row(row, info, get_text, get_json):
    """Fill fields the HF API does not expose, in place. Never fabricates.

    Enrichment must run before classify.py judges a row complete: the HF API
    has no field for MoE active params, so without this step every MoE
    candidate looks incomplete regardless of what its model card actually
    states.

    Only MoE rows get an activation lookup: a dense row's active params equal
    its total by definition, and validate.py enforces that, so writing a
    card-derived figure onto one could only break it. The guard checks the
    row's OWN stated architecture — it does not re-derive one, so a model
    upstream mislabelled "dense" is a pre-existing gap this function must not
    try to paper over.

    Every lookup is None-safe: enrich.* returns None rather than a guess when
    it cannot find an answer, and None here means "leave the field alone",
    never "write nothing over something". Overwriting a real value with
    nothing would destroy information a human would otherwise have reviewed.

    params_active_source records the exact sentence the number came from, so
    a reviewer can check the claim without re-reading the card.
    """
    if row.get("architecture") == "moe" and \
            row.get("params_active_b") == row.get("params_total_b"):
        found = enrich.active_params_from_card(
            enrich.fetch_card(row["hf_repo"], get_text))
        if found is not None:
            row["params_active_b"], row["params_active_source"] = found

    # gated_no_access is RE-DERIVED here on every run, never carried forward.
    # candidates.yaml rows persist between runs, so a stale True would outlive
    # the access grant that fixed it and the row would claim to be locked
    # forever — exactly the freezing bug arena_rank had before it was
    # re-stamped each run. The flag is therefore always popped first and only
    # re-set if this run actually hit an access failure.
    row.pop("gated_no_access", None)
    if not row.get("context_window"):
        notes = {}
        # config.json is the authoritative source and is retried only for a
        # CARRIED row (info is None — see refresh's call site). A row built
        # this run already went through resolve_facts, which fetched
        # config.json; re-reading it here would double the requests on every
        # incomplete row of an org sweep for no new information.
        ctx = None
        if info is None:
            ctx = hf_meta.context_from_config(row["hf_repo"], get_json,
                                              notes=notes)
        if not ctx:
            ctx = enrich.context_from_tokenizer(row["hf_repo"], get_json,
                                                notes=notes)
        if ctx:
            row["context_window"] = ctx
        elif notes.get("gated"):
            row["gated_no_access"] = True

    lic = enrich.license_string(info)
    if lic:
        row["license"] = lic

    return row


def merge_candidates(org_rows, arena_rows):
    """Dedup on lowercased hf_repo, then sort by arena rank, then recency.

    A model found by both sources keeps both tags and the arena rank, so the
    review queue leads with models people actually use.
    """
    by_repo = {}
    for c in list(org_rows) + list(arena_rows):
        key = c["hf_repo"].lower()
        if key not in by_repo:
            by_repo[key] = dict(c)
            continue
        merged = by_repo[key]
        merged["discovered_via"] = sorted(
            set(merged["discovered_via"]) | set(c["discovered_via"]))
        # Carry the arena-only fields over onto whichever row we kept as the
        # base (org-sweep rows come first and have none of them).
        for field in ("arena_rank", "needs_hf_repo", "resolution_confidence"):
            if c.get(field) is not None:
                merged[field] = c[field]

    # ranked first by rank ascending; unranked after, newest first
    return sorted(by_repo.values(),
                  key=lambda c: (c.get("arena_rank") is None,
                                 c.get("arena_rank") or 0,
                                 -_release_ordinal(c)))


def load_aa_index(path=AA):
    """{lower_repo: intelligence_index}. {} on missing/unreadable/malformed.

    Never load-bearing, exactly like the arena file: a discovery run with no
    aa_scores.yaml simply stages rows without an index.
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
            idx = entry.get("intelligence_index")
            if isinstance(idx, int) and not isinstance(idx, bool):
                out[str(repo).lower()] = idx
    return out


def annotate_aa(rows, aa_index):
    """Stamp each row with its Artificial Analysis index, in place.

    Mirrors arena_rank: a discovery-only field, absent rather than null when
    unknown, and stripped on promotion. The point is that a reviewer sees the
    score while deciding whether to take a model, instead of only after it
    lands in models.yaml.

    A row whose repo has dropped out of the sidecar loses the field rather
    than keeping the old number — AA delists older models, and a stale score
    on a carried-forward row would silently misrepresent it.

    NOTE ON FRESHNESS: the workflow runs pull_aa.py before discover.py, and
    pull_aa scores what is already in candidates.yaml. A brand-new candidate
    therefore gets its index on the NEXT weekly run, not the one that found
    it. That lag is accepted: closing it would need a second discovery pass.
    """
    for row in rows:
        idx = aa_index.get((row.get("hf_repo") or "").lower())
        if idx is None:
            row.pop("aa_index", None)
        else:
            row["aa_index"] = idx
    return rows


def load_arena_index(path=ARENA):
    """{lower_repo: best_rank}. {} on missing/unreadable/malformed.

    Mirrors load_aa_index: arena_rank is re-annotated onto every row on every
    run rather than merged once when a row is first staged, so a rank that
    changes -- or vanishes -- underneath a carried-forward row is reflected
    the next time refresh() runs. load_arena (above) builds candidate rows
    for NEW repos and has no need to type-check `rank`; this function feeds
    a comparison/sort (`rank < out[key]`), so a non-int value like "n/a"
    would blow up there, and is dropped instead -- matching
    render_readme.load_arena_ranks_from_rows's isinstance(rank, int) check,
    the stricter of the two contracts already in this codebase for the same
    file.

    A repo can appear more than once (the same model at several reasoning
    efforts); the BEST -- numerically lowest -- rank wins, since that is the
    rank that actually matters for notability and the promotion floor.
    """
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    rows = doc.get("arena_agent") if isinstance(doc, dict) else None
    out = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            repo = row.get("resolved_repo")
            rank = row.get("rank")
            if not repo or not isinstance(rank, int) or isinstance(rank, bool):
                continue
            key = str(repo).lower()
            if key not in out or rank < out[key]:
                out[key] = rank
    return out


def annotate_arena_rank(rows, arena_index):
    """Stamp each row with its arena_agent leaderboard rank, in place.

    Mirrors annotate_aa exactly, and the reason is identical: arena_rank is a
    discovery-only field that both classify.is_notable and the promotion
    floor treat as sufficient on its own (`arena_rank is not None`), so a
    stale value does not just misinform a reviewer -- it keeps a row that
    fell off the leaderboard being treated as ranked, and therefore
    promotable, forever. sweep_orgs/arena_candidates only ever build rows for
    repos NOT already `known`, so a carried-forward row's rank is never
    touched by either of them again; this is the only code path that
    refreshes it.

    A row whose repo the current file does not rank loses the field rather
    than keeping the old number. Concretely: moonshotai/Kimi-K2.7-Code was
    staged at rank 23; the model was later dropped from the leaderboard
    entirely, yet without this the row kept asserting rank 23 -- and kept
    clearing the notability and promotion-floor checks -- indefinitely.
    """
    for row in rows:
        rank = arena_index.get((row.get("hf_repo") or "").lower())
        if rank is None:
            row.pop("arena_rank", None)
        else:
            row["arena_rank"] = rank
    return rows


def _release_ordinal(candidate):
    """Sort key for release_date, tolerant of hand-edited candidates.yaml.

    Rows we generate always carry a real date, but humans edit this file, so
    a missing or non-date value sorts last instead of aborting the run.
    """
    value = candidate.get("release_date")
    return value.toordinal() if isinstance(value, date) else 0


HEADER = (
    "# AUTO-GENERATED candidate models from scripts/discover.py\n"
    "# A row that clears the notability bar with no missing vitals and no\n"
    "# schema problems is promoted straight into models.yaml by discover.py\n"
    "# itself -- these rows are the ones still waiting. Each carries a\n"
    "# needs_review list saying exactly what is missing (a vitals gap, e.g.\n"
    "# gated-repo-no-access/moe-active-params-unknown, or a schema-invalid:\n"
    "# ... entry quoting validate.py's complaint about something a vitals\n"
    "# reason did not already cover). Fix the gap in place; the next\n"
    "# run promotes the row automatically once nothing is left to flag. To\n"
    "# promote a row yourself instead of waiting, move it into models.yaml and\n"
    "# strip the discovery-only fields listed in SCHEMA.md.\n"
    "#\n"
    "# family-already-tracked is the one reason you cannot fix by filling a\n"
    "# gap -- it means this release collides with a family already in\n"
    "# models.yaml and someone has to decide supersede-or-coexist. To answer\n"
    "# COEXIST, add `family_collision_reviewed: true` to the row and the next\n"
    "# run promotes it. It must be a real YAML bool -- `true`, not \"true\" or\n"
    "# yes -- and it clears ONLY the collision, never any other reason. To\n"
    "# answer SUPERSEDE, edit models.yaml by hand; discover.py only ever\n"
    "# appends and will never retire a row for you.\n"
    "#\n"
    "# gated-repo-no-access means the repo's config.json answered 401/403:\n"
    "# the context window exists, we are just not allowed to read it. Accept\n"
    "# the licence terms on the model page with the account whose token CI\n"
    "# uses, and the next run fills the field and clears the flag by itself.\n"
    "#\n"
    "# signal-too-weak means nothing about the row reaches the promotion bar.\n"
    "# signal-too-stale means it does, but the release is old enough that a\n"
    "# human should confirm it is still worth carrying.\n"
)


def write_candidates(path, candidates):
    Path(path).write_text(HEADER + yaml.safe_dump({"models": candidates},
                                                  sort_keys=False,
                                                  allow_unicode=True, width=100))


def refresh(api, min_params, *, orgs=None, data_path=DATA,
            candidates_path=CANDIDATES, arena_path=ARENA, aa_path=AA,
            use_arena=True, get_json=hf_meta._http_get_json,
            get_text=enrich._http_get_text,
            max_age_days=DEFAULT_MAX_AGE_DAYS, today=None):
    """Build the next promotion batch and review queue.

    Returns (promoted, queue, skips, new_orgs).

    The queue is cumulative: rows already staged are carried forward (minus
    any promoted into models.yaml) and newly discovered rows are added to
    them, because write_candidates() replaces the file wholesale and anything
    not returned here is lost. Carried-forward rows that still have unmet
    vitals also get a fresh enrich_row() attempt here (see the comment at
    that call site) — otherwise a staged row only ever gets enriched once,
    at the run that first discovered it.
    """
    orgs = ORG_ALLOWLIST if orgs is None else orgs
    tracked = tracked_repos(data_path)
    staged = surviving_staged(load_staged(candidates_path), tracked)
    known = tracked | {r["hf_repo"].lower() for r in staged}

    # Re-attempt enrichment on carried-forward rows. sweep_orgs/arena_candidates
    # each call enrich_row exactly once, at construction time, because a row
    # freshly built there is the only row they ever see. A row already sitting
    # in candidates.yaml skips both of those paths entirely (it is "known"),
    # so without this loop enrich.py never touches it again after the run
    # that first discovered it — every one of the 358 real staged rows had
    # gotten exactly one enrichment attempt, permanently. A vendor who
    # publishes the missing activation figure a week later should unblock the
    # row automatically instead of leaving it stuck in review forever.
    #
    # Gated on missing_vitals so a fully complete row is never re-fetched —
    # tracked_stems(data_path) does not change within this call, so computing
    # it once up front is enough to gate this loop.
    #
    # info=None: a carried-forward row has no ModelInfo (it was not fetched
    # this run), and enrich_row's licence half needs one for
    # enrich.license_string. Rather than fabricate an info object, pass None
    # and let license_string's existing None-tolerant path make that half a
    # documented no-op for carried rows — only the card-derived active-params
    # and tokenizer-derived context-window halves can actually help them.
    stems = tracked_stems(data_path)
    for row in staged:
        if classify.missing_vitals(row, stems, today=today):
            enrich_row(row, None, get_text, get_json)

    print(f"Sweeping {len(orgs)} orgs (min {min_params}B params, "
          f"max age {max_age_days or 'unlimited'} days)")
    org_rows, skips = sweep_orgs(api, orgs, min_params, known,
                                 get_json=get_json, get_text=get_text,
                                 max_age_days=max_age_days, today=today)
    print(f"  carried {len(staged)} staged candidate(s) forward")
    print(f"  org sweep found {len(org_rows)} new candidate(s); skipped: {skips}")

    arena_rows, new_orgs = [], []
    if use_arena:
        resolved, new_orgs = load_arena(arena_path)
        if resolved:
            print(f"  arena contributed {len(resolved)} resolved repo(s)")
            arena_rows = arena_candidates(api, resolved, min_params, known,
                                          get_json=get_json, get_text=get_text)
        else:
            print("  no arena data (run scripts/pull_arena.py first)")

    candidates = merge_candidates(staged + org_rows, arena_rows)

    aa_index = load_aa_index(aa_path)
    annotate_aa(candidates, aa_index)

    arena_index = load_arena_index(arena_path)
    annotate_arena_rank(candidates, arena_index)

    # Reuses the `stems` computed above the re-enrichment loop rather than
    # recomputing tracked_stems(data_path) — nothing in between mutates
    # data_path, so the value is still current.
    promoted, queue = [], []
    for row in candidates:
        verdict = classify.route(row, stems, today=today)
        if verdict == "drop":
            continue
        if verdict == "promote":
            promoted.append(row)
            stems.add(names.family_stem(row["hf_repo"]))
            continue
        # route() can land here for two independent reasons: missing_vitals
        # (worth a look but not complete) and/or schema_errors (would fail
        # validate.py outright, e.g. a hand-edited candidates.yaml row with a
        # release_date that is not a date). review_reasons surfaces both so a
        # reviewer sees everything wrong in one pass, without reporting the
        # same gap twice — see it for how the overlap is suppressed.
        row["needs_review"] = classify.review_reasons(row, stems, today=today)
        queue.append(row)

    print(f"  {len(promoted)} promotable, {len(queue)} need review, "
          f"{len(candidates) - len(promoted) - len(queue)} below the bar")

    return promoted, queue, skips, new_orgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-params", type=float, default=3.0,
                    help="minimum total params in billions")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help="ignore org-sweep repos created longer ago than this "
                         "(0 = no age limit)")
    ap.add_argument("--no-arena", action="store_true",
                    help="skip merging arena_agent_rankings.yaml")
    args = ap.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    promoted, queue, _, new_orgs = refresh(api, args.min_params,
                                           use_arena=not args.no_arena,
                                           max_age_days=args.max_age_days)

    if new_orgs:
        print(f"\nNEW ORGS seen on the leaderboard but not in ORG_ALLOWLIST: "
              f"{', '.join(new_orgs)}")
        print("Add them to ORG_ALLOWLIST in this file to widen coverage.")

    n = append_models(DATA, [promotion_row(r) for r in promoted])
    write_candidates(CANDIDATES, queue)
    print(f"\nPromoted {n} model(s) into {DATA.name}; "
          f"{len(queue)} awaiting review in {CANDIDATES.name}")
    # No explicit exit code: a normal run falls off the end and exits 0, and
    # the Action decides whether the diff is non-empty. This is NOT a
    # guarantee append_models() always succeeds — a models.yaml that fails
    # _assert_appendable_shape() raises ModelsYamlShapeError here, which
    # propagates uncaught and exits non-zero on purpose, so a shape problem
    # fails the Action loudly instead of silently corrupting the file.


if __name__ == "__main__":
    main()
