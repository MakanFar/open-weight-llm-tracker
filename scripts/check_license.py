#!/usr/bin/env python3
"""
Gather the EVIDENCE needed to verify a row's commercial_use claim.

This script decides nothing. It fetches what the vendor actually published
and reports it; a human (or an agent following .claude/skills/
verify-commercial-use) reads the report and makes the call.

WHY IT EXISTS:
    commercial_use on an auto-promoted row is inferred from the Hugging Face
    licence TAG, which is uploader-supplied metadata, not a licence. The
    README marks those with a trailing `?`. Clearing that marker means
    confirming the repo really carries the licence the tag names -- and the
    interesting failure is not a tag that is simply wrong, it is a repo that
    ships a standard licence with extra use restrictions bolted on. That
    still tags as `apache-2.0` and it is no longer plain Apache-2.0.

WHAT IT REPORTS, per repo:
    licence_files   what the repo actually serves (LICENSE, NOTICE, ...)
    identified      which canonical licence the text matches, if any
    added_terms     restriction language found in a file that otherwise
                    matches a permissive licence -- the case worth a human
    card_licence    the licence section of the model card, which is the only
                    published statement when there is no licence file at all

  python scripts/check_license.py --unverified        # every row still `?`
  python scripts/check_license.py Qwen/Qwen3.5-27B    # one repo
  python scripts/check_license.py --unverified --json # machine-readable
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_meta  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "models.yaml"

API = "https://huggingface.co/api/models/{repo}?expand[]=cardData&expand[]=siblings"
RAW = "https://huggingface.co/{repo}/raw/main/{path}"

# Phrases that appear in every copy of a licence and nowhere else. Matched
# against whitespace-normalised lowercase text, so reflowed copies still hit.
# ALL of a licence's markers must be present -- a card that merely mentions
# "the Apache License" does not thereby become one.
LICENSE_MARKERS = {
    "apache-2.0": (
        "apache license version 2.0",
        "licensed under the apache license",
        "unless required by applicable law or agreed to in writing",
    ),
    "mit": (
        "permission is hereby granted, free of charge",
        "the software is provided \"as is\", without warranty",
    ),
}

# Commercial-use consequence of each canonical licence, once the text really
# is that licence AND carries no added terms. These are settled: both are
# OSI-approved and neither restricts field of use.
COMMERCIAL_USE = {"apache-2.0": True, "mit": True}

# Language that restricts HOW the weights may be used. A permissive licence
# contains none of it, so a hit means the file is that licence PLUS something
# -- which is the whole point of reading the file instead of the tag.
RESTRICTION_PATTERNS = (
    r"acceptable use", r"you may not use", r"shall not use", r"must not use",
    r"prohibited", r"restrictions? on use", r"field[- ]of[- ]use",
    r"non-?commercial", r"research (?:purposes? )?only", r"military",
    r"monthly active users", r"supplemental terms", r"additional terms",
    r"use policy", r"you are not permitted",
)

# Lines that ARE the canonical licence and must never read as an added
# restriction. Apache-2.0's own boilerplate says "you may not use this file
# except in compliance with the License" -- without this, every clean Apache
# repo flags itself and the signal is worthless.
CANONICAL_BOILERPLATE = (
    "you may not use this file except in compliance with the license",
    "this license does not grant permission to use the trade names",
    "you may add your own copyright statement",
    # Apache-2.0 section 5, Submission of Contributions: "...shall be under
    # the terms and conditions of this License, without any additional terms
    # or conditions." Flagged 7 of the 8 readable licences on the first live
    # run -- a scanner that fires on every clean Apache file reports nothing.
    "without any additional terms or conditions",
)

CARD_LICENCE_HEADING = re.compile(
    r"^#{1,4}\s*.*licen[cs]e.*$", re.I | re.M)


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or hf_meta.auth_headers(
        "owlt-license/1.0"))
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def normalise(text):
    """Lowercase, whitespace-collapsed text, for marker matching.

    Licence files are reflowed, re-indented and re-wrapped constantly; none of
    that changes the licence, and all of it defeats a naive substring search.
    """
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def identify(text):
    """Which canonical licence this text IS, or None.

    None is the important answer: it means nobody has confirmed anything and
    the row must keep its `?`. A near-miss is not a match -- every marker for
    a licence has to be present.
    """
    flat = normalise(text)
    for name, markers in LICENSE_MARKERS.items():
        if all(m in flat for m in markers):
            return name
    return None


def added_terms(text):
    """Restriction language found in the file, as (pattern, quoted line) pairs.

    Reported even when identify() matched, because "Apache-2.0 plus an
    acceptable-use policy" tags as apache-2.0 and is not Apache-2.0. Returning
    the surrounding line rather than the bare match is deliberate: the whole
    value of this step is that a reviewer can read the clause.
    """
    hits = []
    for line in (text or "").splitlines():
        stripped = " ".join(line.split())
        if not stripped:
            continue
        flat = stripped.lower()
        if any(b in flat for b in CANONICAL_BOILERPLATE):
            continue
        for pat in RESTRICTION_PATTERNS:
            if re.search(pat, stripped, re.I):
                hits.append((pat, stripped[:300]))
                break
    return hits


def card_licence_section(card):
    """The model card's licence section, or None.

    The only published statement of terms when a repo ships no licence file,
    which is the majority case. It is evidence, not proof: a card is prose a
    vendor wrote, and it is not the instrument that grants the licence.
    """
    if not card:
        return None
    matches = list(CARD_LICENCE_HEADING.finditer(card))
    if not matches:
        return None
    start = matches[0].start()
    nxt = re.search(r"^#{1,4}\s", card[matches[0].end():], re.M)
    end = matches[0].end() + nxt.start() if nxt else len(card)
    section = " ".join(card[start:end].split())
    return section[:600] or None


def gather(repo, get=_get):
    """Everything published about this repo's licence. Never raises."""
    report = {"repo": repo, "tag": None, "license_name": None,
              "licence_files": [], "identified": None, "added_terms": [],
              "card_licence": None, "errors": []}
    try:
        meta = json.loads(get(API.format(repo=repo)))
    except Exception as exc:
        report["errors"].append(f"metadata: {type(exc).__name__}")
        return report

    card_data = meta.get("cardData") or {}
    report["tag"] = card_data.get("license")
    report["license_name"] = card_data.get("license_name")
    report["license_link"] = card_data.get("license_link")

    names = [s.get("rfilename") for s in (meta.get("siblings") or [])
             if isinstance(s, dict) and s.get("rfilename")]
    report["licence_files"] = [n for n in names
                               if n.lower().startswith(("licen", "notice"))]

    for path in report["licence_files"]:
        try:
            body = get(RAW.format(repo=repo, path=path))
        except Exception as exc:
            report["errors"].append(f"{path}: {type(exc).__name__}")
            continue
        found = identify(body)
        if found and not report["identified"]:
            report["identified"] = found
        report["added_terms"] += [(path, pat, line)
                                  for pat, line in added_terms(body)]

    # The card is supplementary evidence, so its absence is reported as
    # card_licence=None rather than as an error. Putting it in errors would
    # make a repo with no README look like a fetch failure, which verdict()
    # treats as "we could not look" instead of "there is nothing to find".
    try:
        report["card_licence"] = card_licence_section(
            get(RAW.format(repo=repo, path="README.md")))
    except Exception:
        pass
    return report


def verdict(report, row_license):
    """A routing label, NOT a decision. See the skill for what each means.

    Deliberately conservative: only `confirmed` may clear a `?`, and it
    requires the repo's own licence text to match the licence the row claims
    with no restriction language anywhere in it.
    """
    if report["errors"] and not report["licence_files"]:
        return "fetch-failed"
    if not report["licence_files"]:
        return "tag-only"
    if report["identified"] is None:
        return "unrecognised-text"
    if report["identified"] != row_license:
        return "tag-mismatch"
    if report["added_terms"]:
        return "added-terms"
    return "confirmed"


AUTO_NOTE = "AUTO-DISCOVERED"

BLOB = "https://huggingface.co/{repo}/blob/main/{path}"


def plan_edits(reports, rows):
    """{repo: {field: value}} — the ONLY writes --apply is allowed to make.

    Exactly two verdicts are safe to act on without a person:

    `confirmed` — the repo's own licence file is verbatim a licence whose
    commercial consequence is settled (COMMERCIAL_USE), with no restriction
    language anywhere in it. Nothing is being decided here: Apache-2.0 and MIT
    are OSI-approved and neither restricts field of use, so the only per-repo
    question was whether the text really is that licence, and reading it
    answered that. A licence identify() recognises but COMMERCIAL_USE has no
    entry for is NOT confirmed — inventing the consequence is the one thing
    this must never do.

    `tag-only` — the vendor publishes no licence file, so the row is marked
    license_text_published: false and keeps its unverified status. Saying "this
    cannot be checked" is a fact about the release; it is not a claim that
    anyone checked.

    Everything else (added-terms, tag-mismatch, unrecognised-text,
    fetch-failed) is a human's call and produces no edit at all.

    A row a human already verified is never touched, and license_notes is
    replaced only when it is still the AUTO-DISCOVERED placeholder — a
    reviewer's own words are not ours to overwrite.
    """
    edits = {}
    for report in reports:
        repo = report["repo"]
        row = rows.get(repo)
        if row is None or row.get("commercial_use_verified"):
            continue
        verdict_ = report.get("verdict")
        change = {}
        if verdict_ == "confirmed":
            allows = COMMERCIAL_USE.get(report["identified"])
            if allows is None:
                continue
            path = report["licence_files"][0]
            change["commercial_use"] = allows
            change["commercial_use_verified"] = True
            change["commercial_use_source"] = (
                f"{path} @ {BLOB.format(repo=repo, path=path)} — "
                f"{report['identified']} verbatim, no added use restrictions")
            if str(row.get("license_notes", "")).startswith(AUTO_NOTE):
                change["license_notes"] = (
                    f"{report['identified']}, unmodified.")
        elif verdict_ == "tag-only":
            change["license_text_published"] = False
            if str(row.get("license_notes", "")).startswith(AUTO_NOTE):
                change["license_notes"] = (
                    f"Vendor publishes no licence file; the Hugging Face tag "
                    f"'{report['tag']}' is the only claim.")
        if change:
            edits[repo] = change
    return edits


def apply_edits(path, edits):
    """Rewrite models.yaml in place. Returns the number of rows changed.

    Line surgery rather than a YAML round-trip: safe_dump would reorder every
    key and drop the file's header comments.

    Works a row block at a time. An earlier version replaced fields inline and
    appended the not-yet-present ones when it saw the NEXT row's hf_repo — by
    which point the next row's `- name:` line was already emitted, so a new
    field silently attached itself to the following model. Buffering the block
    is what makes "insert into this row" mean this row.
    """
    lines = Path(path).read_text().split("\n")
    out, block, repo = [], [], None
    changed = 0

    def flush():
        nonlocal changed
        if not block:
            return
        pending = dict(edits.get(repo, {}))
        body = []
        for line in block:
            fm = re.match(r"^\s{4}([a-z_]+):", line)
            if fm and fm.group(1) in pending:
                body.append(f"    {fm.group(1)}: "
                            f"{_yaml_scalar(pending.pop(fm.group(1)))}")
            else:
                body.append(line)
        if pending:
            # insert after the row's last field, never after its trailing
            # blank line, which belongs to the gap between rows
            last = max((i for i, l in enumerate(body) if l.strip()),
                       default=len(body) - 1)
            body[last + 1:last + 1] = [f"    {k}: {_yaml_scalar(v)}"
                                       for k, v in pending.items()]
        if repo in edits:
            changed += 1
        out.extend(body)
        block.clear()

    in_rows = False
    for line in lines:
        if re.match(r"^\s{2}-\s+name:", line):
            flush()
            in_rows, repo = True, None
        m = re.match(r"^\s{4}hf_repo:\s*(\S+)\s*$", line)
        if m:
            repo = m.group(1)
        (block if in_rows else out).append(line)
    flush()
    Path(path).write_text("\n".join(out))
    return changed


def _yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace('"', "'")
    return f'"{text}"'


def unverified_rows(path=DATA):
    doc = yaml.safe_load(Path(path).read_text()) or {}
    return [r for r in (doc.get("models") or [])
            if isinstance(r, dict) and r.get("hf_repo")
            and not r.get("commercial_use_verified")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", help="hf repo ids to check")
    ap.add_argument("--unverified", action="store_true",
                    help="check every models.yaml row still marked `?`")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--apply", action="store_true",
                    help="write the two safe outcomes into models.yaml: a "
                         "confirmed licence clears the `?`, a repo with no "
                         "licence file is marked license_text_published: "
                         "false. Every other verdict needs a human and is "
                         "left untouched.")
    args = ap.parse_args()

    rows = {r["hf_repo"]: r for r in unverified_rows()}
    targets = args.repos or (list(rows) if args.unverified else [])
    if not targets:
        sys.exit("nothing to do: pass repo ids or --unverified")

    results = []
    for repo in targets:
        report = gather(repo)
        report["row_license"] = rows.get(repo, {}).get("license", report["tag"])
        report["row_commercial_use"] = rows.get(repo, {}).get("commercial_use")
        report["verdict"] = verdict(report, report["row_license"])
        report["implies_commercial_use"] = (
            COMMERCIAL_USE.get(report["identified"])
            if report["verdict"] == "confirmed" else None)
        results.append(report)
        if not args.json:
            print(f"\n{repo}")
            print(f"  verdict        {report['verdict']}")
            print(f"  row claims     license={report['row_license']!r} "
                  f"commercial_use={report['row_commercial_use']!r}")
            print(f"  hf tag         {report['tag']!r}"
                  + (f"  name={report['license_name']!r}"
                     if report["license_name"] else ""))
            print(f"  licence files  {report['licence_files'] or '(none)'}")
            print(f"  text matches   {report['identified'] or '(no match)'}")
            if report["implies_commercial_use"] is not None:
                print(f"  => licence permits commercial use: "
                      f"{report['implies_commercial_use']}")
            for path, pat, line in report["added_terms"][:6]:
                print(f"  ! added terms  [{path}] {line[:120]}")
            if report["card_licence"]:
                print(f"  card says      {report['card_licence'][:200]}")
            for e in report["errors"]:
                print(f"  ? error        {e}")

    if args.apply:
        edits = plan_edits(results, rows)
        n = apply_edits(DATA, edits)
        print(f"\n{n} row(s) updated in {DATA.name}")
        for repo, change in sorted(edits.items()):
            print(f"  {repo}: {', '.join(sorted(change))}")
        skipped = [r["repo"] for r in results if r["repo"] not in edits]
        if skipped:
            print(f"\n{len(skipped)} row(s) left for a human:")
            for r in results:
                if r["repo"] in skipped:
                    print(f"  {r['repo']:44s} {r['verdict']}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print("\n--- " + "  ".join(f"{v} {k}" for k, v in
                                   sorted(counts.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
