---
name: verify-commercial-use
description: Use when clearing the trailing `?` or `†` on the README's Commercial column — establishing a models.yaml row's licence and commercial_use from what the vendor actually published, including licence pages, vendor repos and announcements found by web search, rather than the Hugging Face licence tag it was inferred from.
---

# Verify commercial_use

**You are the reviewer.** Not a triage step that hands work to someone else —
you do the research, read the licence, and make the determination. The trailing
`?` and `†` in the README exist because nobody had done that yet. Your job is to
do it.

## What the markers mean

- **`?`** — `commercial_use` was inferred from the Hugging Face licence tag.
  That tag is uploader-supplied metadata typed into a form. Nobody read a
  licence.
- **`†`** — someone looked and the repo publishes no licence file. The tag is
  the only claim *in the repo*. It does not mean the licence is unpublished —
  it usually is, elsewhere, and finding it is your job.

## The one rule that matters

**Your knowledge directs the search. It is never the evidence.**

You may know Apache-2.0 permits commercial use — that part is settled and you
should not re-litigate it. You may *believe* Qwen ships Apache-2.0. But
"Qwen ships Apache-2.0" is a claim about a specific release, and the only thing
that settles it is a document you fetched. Every determination you write must
cite a URL you actually retrieved and quote the operative text.

A confident answer with no fetched source is the single worst output here: it is
indistinguishable in the data from a checked one, and it poisons the field the
whole tracker exists to be trusted on. **If research fails, leave the marker.**
That is a finished, correct outcome.

## Workflow

### 1. Gather what the repo itself publishes

```bash
python scripts/check_license.py --unverified          # every unsettled row
python scripts/check_license.py Qwen/Qwen3.5-27B      # one repo
python scripts/check_license.py --unverified --json   # machine-readable
```

Then let the script settle everything mechanical:

```bash
python scripts/check_license.py --unverified --apply
```

`--apply` writes only the two outcomes needing no judgement — a `confirmed`
licence file clears the marker, a `tag-only` row is marked
`license_text_published: false`. It never touches a row a human already
verified. Everything else it prints and leaves for you.

### 2. Research what is left, per vendor — not per row

Nine Qwen rows share one licensing decision. Do it once, apply it to all of
them, and cite the same source. Group the queue by org and family first; the
18 rows in the first real run were 7 decisions.

Look in this order, and stop at the first tier that actually answers:

| tier | source | strength |
|---|---|---|
| 1 | The repo's own `LICENSE` file | Definitive. `check_license.py` already checked this. |
| 2 | `license_link` in the card's frontmatter — the script prints it | The vendor naming its own instrument. Follow it with WebFetch. |
| 3 | The vendor's official licence page or GitHub `LICENSE` for the same release | Strong. Confirm it names *this* release. |
| 4 | The model card's licence section | Evidence, not the instrument. Can raise a doubt; cannot settle one alone. |
| 5 | Release announcement, tech report, press coverage (WebSearch) | Corroboration only. Never the sole basis. |

Then read for the two things that actually decide it:

1. **Is the text a standard licence, unmodified?** If so its commercial terms
   are settled and you are done.
2. **Is anything incorporated by reference?** A prohibited-use or acceptable-use
   policy *inside* the licence makes it `conditional`. The same policy merely
   *linked from the same website* does not. This distinction is the whole job —
   check whether the licence text itself conditions the grant on it.

### 3. Decide

| finding | `commercial_use` |
|---|---|
| Standard permissive licence (Apache-2.0, MIT, BSD), unmodified | `true` |
| Permissive licence + use restrictions incorporated into the grant | `conditional` |
| Custom vendor licence permitting commercial use with conditions (MAU caps, naming, use policy) | `conditional` |
| Licence forbids commercial use | `false` |
| You could not find the instrument | **leave it — change nothing** |

If the licence turns out not to be the one the row claims, fix `license` too,
and add the new string to `validate.LICENSES` in the same change.

### 4. Write it

```yaml
    license: apache-2.0
    commercial_use: true
    commercial_use_verified: true
    commercial_use_source: "https://ai.google.dev/gemma/docs/gemma_4_license — unmodified Apache-2.0; the Gemma Prohibited Use Policy is a separately linked document, not incorporated into the grant"
```

`commercial_use_source` must contain **the URL you fetched** and **what it
said** — enough that the next person re-checks it without repeating your search.
Not "verified via web search". Drop `license_text_published` if you found the
licence elsewhere; keep it if the repo still ships no file but you verified from
the vendor's site — the two facts are independent.

Replace any `AUTO-DISCOVERED` placeholder in `license_notes` with what you found.

### 5. Confirm

```bash
python scripts/validate.py
python scripts/render_readme.py && python scripts/render_json.py
python -m pytest tests/ -q
```

## Rules

1. **Never write `commercial_use_verified: true` without a URL you fetched and
   a clause you read.** Not from memory, not from a tag, not from "everyone
   knows".
2. **Report what you could not settle, explicitly and by name.** A run that
   verifies 12 of 18 and names the other 6 is a success. A run that verifies 18
   is a claim you should be suspicious of.
3. **A separate policy page is not a licence term** unless the licence text
   incorporates it. Getting this backwards turns every permissively-licensed
   model into `conditional`, and getting it the other way publishes a
   restricted model as free.
4. **Distrust a permissive tag from a vendor with a history of custom
   licences** — and equally, do not assume the history still holds. Gemma 1–3
   used the custom Gemma Terms of Use; Gemma 4 genuinely moved to Apache-2.0.
   Both the tag *and* the prior assumption needed checking.
5. **Accepting a tag as the licence is a maintainer decision, never yours.**
   Where the only evidence is the HF tag — or where two found documents
   disagree about which governs the weights — you may not resolve it by
   picking one. Present the conflict and let the maintainer choose. When they
   do, say so in `commercial_use_source` in those words ("maintainer
   decision"), so the row is never later mistaken for one where somebody read
   the instrument. Four rows in `models.yaml` carry exactly that wording.
6. **Do not touch rows already `commercial_use_verified: true`.**
7. **One licence, one determination, many rows.** Never re-derive what
   Apache-2.0 means per model. The per-row question is only ever *which*
   licence this release is under.

## Watch out

- `check_license.py`'s restriction scanner skips `CANONICAL_BOILERPLATE`,
  because Apache-2.0's own text says "you may not use this file except in
  compliance with the License" and (§5) "without any additional terms or
  conditions". Those flagged every clean Apache repo on the first run. Add new
  false positives there **with a test**; never loosen `RESTRICTION_PATTERNS`.
- `link-contradicts-tag` fires when a permissive tag links to a vendor-hosted
  licence page. It is a *lead*, not a verdict: the Gemma 4 link looked like a
  custom licence and turned out to serve unmodified Apache-2.0. Follow it.
- Gated repos (`meta-llama/*`, older `google/gemma-*`) may 401 without a token.
  That is `fetch-failed` — "could not look", not "nothing to find".
- A card saying "intended for research and educational use" alongside an
  Apache-2.0 grant is a *guideline*, not a licence term. Check whether the
  licence conditions anything on it before downgrading to `conditional`.
