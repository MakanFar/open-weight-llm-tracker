---
name: verify-commercial-use
description: Use when clearing the trailing `?` on the README's Commercial column — verifying a models.yaml row's commercial_use against the licence the vendor actually published, rather than the Hugging Face licence tag it was inferred from.
---

# Verify commercial_use

## What the `?` means

`commercial_use` on an auto-promoted row was **inferred from the Hugging Face
licence tag**. That tag is uploader-supplied metadata typed into a form — it is
not a licence, nobody checked it, and `render_readme.commercial_badge` marks
every such row with a trailing `?`.

Your job is to replace an inference with a reading. **The `?` is honest. A
wrongly cleared `?` is a false legal claim published as settled fact**, so
leaving a row unverified is always an acceptable outcome and is often the
correct one.

## The failure this is actually looking for

Not a tag that is simply wrong — those are rare. The one that matters is a repo
shipping a **standard licence with extra use restrictions appended**. That still
tags as `apache-2.0`, still looks permissive, and is not Apache-2.0. Reading the
file is the only way to see it.

## Workflow

### 1. Gather evidence

```bash
python scripts/check_license.py --unverified          # every row still `?`
python scripts/check_license.py Qwen/Qwen3.5-27B      # one repo
python scripts/check_license.py --unverified --json   # machine-readable
```

The script **decides nothing**. It fetches what the vendor published — the
repo's own licence files, whether the text matches a canonical licence, any
restriction language in it, and the model card's licence section — and gives
each row a routing verdict. You make the call.

### 2. Route on the verdict

| verdict | what it means | what to do |
|---|---|---|
| `confirmed` | The repo's own licence file **is** the licence the row claims, with no restriction language anywhere in it. | Set `commercial_use` from the licence, `commercial_use_verified: true`, and cite the file. |
| `tag-only` | The repo publishes **no licence file at all**. The tag is the only claim there has ever been. | **Do not verify.** Mark `license_text_published: false` — the table renders a trailing `†`, meaning "cannot be checked" rather than "not checked yet". |
| `added-terms` | Text matches, but carries restriction language. | Read the quoted clause. Usually `conditional`. Never `true` without reading it. Ask a human if it is not obvious. |
| `tag-mismatch` | The licence text is a **different licence** from the tag. | Fix `license` first, then re-run. Needs a human. |
| `unrecognised-text` | A licence file exists but matches nothing canonical — a bespoke vendor licence. | Read it in full. Add the licence to `validate.LICENSES` if adopting it. Human decision. |
| `fetch-failed` | Could not look. | Not evidence of anything. Retry later. |

### 3. Write the result

```bash
python scripts/check_license.py --unverified --apply
```

`--apply` writes **only the two outcomes that need no judgement** and prints
everything it refused. A `confirmed` row gets verified; a `tag-only` row gets
marked unpublishable. `added-terms`, `tag-mismatch`, `unrecognised-text` and
`fetch-failed` are always left untouched for you.

It never touches a row already `commercial_use_verified: true`, and it replaces
`license_notes` only while that note is still the `AUTO-DISCOVERED` placeholder
— a reviewer's own words are not the script's to overwrite.

For a verdict `--apply` refuses, write it by hand. For `confirmed` that shape is:

```yaml
    commercial_use: true
    commercial_use_verified: true
    commercial_use_source: "LICENSE @ https://huggingface.co/<repo>/blob/main/LICENSE — Apache-2.0 verbatim, no added use restrictions"
```

and for a repo that publishes no licence file:

```yaml
    license_text_published: false
    license_notes: "Vendor publishes no licence file; the Hugging Face tag 'apache-2.0' is the only claim."
```

Replace any `license_notes: "AUTO-DISCOVERED — verify license terms."` with
what you found, or drop it.

`commercial_use_source` is the citation: **what you read and where**, precise
enough that the next person can re-check it without repeating the search. Not
"checked the licence".

### 4. Confirm nothing broke

```bash
python scripts/validate.py
python scripts/render_readme.py && python scripts/render_json.py
python -m pytest tests/ -q
```

The `?` should be gone from exactly the rows you verified.

## Rules

1. **Never set `commercial_use_verified: true` without licence text you read.**
   A tag, a card sentence, a blog post and "everyone knows Apache is
   permissive" are all insufficient. The point of the field is that someone
   read the instrument.
2. **`tag-only` cannot be verified.** 18 of the 26 rows were this. It is not a
   gap to be filled by trying harder — the vendor published no licence file, so
   there is nothing to confirm and no amount of searching changes that. Mark it
   `license_text_published: false` and move on. That is a finished outcome, not
   a deferral: `†` in the table says the vendor never published the instrument,
   which is a fact about the release and useful to a reader.
3. **A model card is evidence, not proof.** It is prose the vendor wrote
   alongside the weights; it is not the instrument granting the licence. It can
   raise a doubt (worth acting on) but cannot settle one.
4. **Abstain loudly.** Say which rows you could not verify and why. A run that
   verifies 8 of 26 and marks the other 18 unpublishable is a success. A run
   that verifies 26 is a bug.
5. **One licence, one determination.** `apache-2.0` and `mit` are OSI-approved
   and neither restricts field of use, so both imply `commercial_use: true` —
   *once the text really is that licence and carries nothing extra*. The
   per-repo work is confirming that, never re-litigating what Apache-2.0 means.
6. **Do not touch rows already `commercial_use_verified: true`.** A human
   settled those.

## Watch out

- `check_license.py`'s restriction scanner skips `CANONICAL_BOILERPLATE`,
  because Apache-2.0's own text contains "you may not use this file except in
  compliance with the License" and "without any additional terms or
  conditions". Those two lines flagged every clean Apache repo on the first
  run. If you see a new false positive that is canonical licence text, add it
  there **with a test** — do not loosen `RESTRICTION_PATTERNS`, which would
  cost real detections.
- Gated repos (`meta-llama/*`, `google/gemma-*`) may 401 on file fetches
  without a token. That is `fetch-failed`, not `tag-only`.
- Google's Gemma 4 rows tag `apache-2.0` while every previous Gemma release
  used the custom, conditional Gemma licence, and they ship no licence file.
  `models.yaml` carries a standing `VERIFY:` note about it. Treat a permissive
  tag from a vendor with a history of custom licences as a reason for more
  suspicion, not less.
