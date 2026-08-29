import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_license as cl

APACHE = """
                              Apache License
                        Version 2.0, January 2004

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS.
"""

MIT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software, to deal in the Software without restriction.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


def test_identifies_apache_and_mit():
    assert cl.identify(APACHE) == "apache-2.0"
    assert cl.identify(MIT) == "mit"


def test_identify_survives_reflowed_text():
    """Vendors re-wrap and re-indent licence files constantly. None of that
    changes the licence, and all of it defeats a naive substring search."""
    squashed = " ".join(APACHE.split())
    assert cl.identify(squashed) == "apache-2.0"


def test_identify_returns_none_for_a_mere_mention():
    """A card saying 'released under the Apache License' is not a licence.
    Every marker has to be present, not just the name."""
    assert cl.identify("This model is released under the Apache License.") is None
    assert cl.identify("") is None
    assert cl.identify(None) is None


def test_added_terms_flags_a_bolted_on_restriction():
    """The interesting failure is not a wrong tag -- it is a standard licence
    with use restrictions appended. That still tags as apache-2.0."""
    text = APACHE + "\n7. You may not use the Model for military purposes.\n"
    hits = cl.added_terms(text)
    assert hits and "military" in hits[0][1].lower()


def test_added_terms_is_silent_on_a_clean_licence():
    assert cl.added_terms(APACHE) == []
    assert cl.added_terms(MIT) == []


def test_card_licence_section_is_extracted():
    card = "# Model\n\nblah\n\n## License\n\nApache 2.0, no strings.\n\n## Usage\n\ncode"
    got = cl.card_licence_section(card)
    assert "Apache 2.0, no strings." in got
    assert "Usage" not in got


def test_card_licence_section_absent():
    assert cl.card_licence_section("# Model\n\njust prose") is None
    assert cl.card_licence_section(None) is None


# --- verdict routing -------------------------------------------------------

def _report(**kw):
    base = dict(repo="org/m", tag="apache-2.0", licence_files=["LICENSE"],
                identified="apache-2.0", added_terms=[], card_licence=None,
                errors=[])
    base.update(kw)
    return base


def test_confirmed_only_when_the_text_matches_the_row():
    assert cl.verdict(_report(), "apache-2.0") == "confirmed"


def test_tag_only_when_the_repo_publishes_no_licence_file():
    """18 of the 26 unverified rows are this. There is no text to read, so
    nothing can be confirmed -- the row keeps its `?`."""
    assert cl.verdict(_report(licence_files=[], identified=None),
                      "apache-2.0") == "tag-only"


def test_added_terms_never_confirms():
    """Apache-2.0 plus an acceptable-use policy is not Apache-2.0, and
    silently confirming it would publish a false permissive claim."""
    r = _report(added_terms=[("LICENSE", "prohibited", "you are prohibited")])
    assert cl.verdict(r, "apache-2.0") == "added-terms"


def test_tag_mismatch_when_the_text_is_a_different_licence():
    assert cl.verdict(_report(identified="mit"), "apache-2.0") == "tag-mismatch"


def test_unrecognised_text_is_not_confirmed():
    assert cl.verdict(_report(identified=None), "apache-2.0") == "unrecognised-text"


def test_fetch_failure_is_distinct_from_tag_only():
    """A network failure must not read as 'the vendor published nothing'."""
    r = _report(licence_files=[], identified=None, errors=["metadata: HTTPError"])
    assert cl.verdict(r, "apache-2.0") == "fetch-failed"


# --- gather ----------------------------------------------------------------

def _fake_get(meta, files):
    import json as _json

    def get(url):
        if url.startswith(cl.API.split("{")[0]) and "api/models" in url:
            return _json.dumps(meta)
        for name, body in files.items():
            if url.endswith("/" + name):
                return body
        raise RuntimeError("404")
    return get


def test_gather_reads_the_licence_file():
    meta = {"cardData": {"license": "apache-2.0"},
            "siblings": [{"rfilename": "LICENSE"}, {"rfilename": "config.json"}]}
    r = cl.gather("org/m", get=_fake_get(meta, {"LICENSE": APACHE}))
    assert r["licence_files"] == ["LICENSE"]
    assert r["identified"] == "apache-2.0"
    assert r["errors"] == []


def test_gather_never_raises_on_a_dead_repo():
    def boom(url):
        raise RuntimeError("410")
    r = cl.gather("org/gone", get=boom)
    assert r["errors"] and r["identified"] is None


def test_apache_section_5_is_not_an_added_term():
    """"...under the terms and conditions of this License, without any
    additional terms or conditions" is Apache-2.0's own Submission of
    Contributions clause. It flagged 7 of 8 real licences before this."""
    text = APACHE + ("\n5. Submission of Contributions. Any Contribution shall be"
                     " under the terms and conditions of this License, without"
                     " any additional terms or conditions.\n")
    assert cl.added_terms(text) == []


# --- what --apply is allowed to write --------------------------------------

def _row(**kw):
    base = dict(hf_repo="org/m", license="apache-2.0", commercial_use=True)
    base.update(kw)
    return base


def test_plan_writes_verification_for_a_confirmed_row():
    r = _report(repo="org/m")
    r["verdict"] = "confirmed"
    edits = cl.plan_edits([r], {"org/m": _row()})
    assert edits["org/m"]["commercial_use_verified"] is True
    assert edits["org/m"]["commercial_use"] is True
    assert "LICENSE" in edits["org/m"]["commercial_use_source"]


def test_plan_marks_a_tag_only_row_unpublished_without_verifying_it():
    """The whole point: this row can never be verified, and saying so is
    different from claiming it was checked."""
    r = _report(repo="org/m", licence_files=[], identified=None)
    r["verdict"] = "tag-only"
    edits = cl.plan_edits([r], {"org/m": _row()})
    assert edits["org/m"]["license_text_published"] is False
    assert "commercial_use_verified" not in edits["org/m"]


def test_plan_refuses_every_verdict_a_human_owns():
    """added-terms, tag-mismatch, unrecognised-text and fetch-failed all need
    a person. Writing anything for them is the failure mode this guards."""
    for verdict in ("added-terms", "tag-mismatch", "unrecognised-text",
                    "fetch-failed"):
        r = _report(repo="org/m")
        r["verdict"] = verdict
        assert cl.plan_edits([r], {"org/m": _row()}) == {}


def test_plan_never_touches_a_row_a_human_already_verified():
    r = _report(repo="org/m")
    r["verdict"] = "confirmed"
    rows = {"org/m": _row(commercial_use_verified=True)}
    assert cl.plan_edits([r], rows) == {}


def test_plan_only_replaces_the_auto_discovered_placeholder_note():
    """A human's licence_notes is the reviewer's own words and must survive."""
    r = _report(repo="org/m"); r["verdict"] = "confirmed"
    auto = cl.plan_edits([r], {"org/m": _row(
        license_notes="AUTO-DISCOVERED — verify license terms.")})
    assert "license_notes" in auto["org/m"]
    human = cl.plan_edits([r], {"org/m": _row(
        license_notes="Free below 700M MAU; naming terms apply.")})
    assert "license_notes" not in human["org/m"]


def test_plan_will_not_confirm_a_licence_with_no_known_consequence():
    """identify() could match a licence nobody has decided the commercial
    consequence of. Better to route it to a human than invent one."""
    r = _report(repo="org/m", identified="bsd-3-clause")
    r["verdict"] = "confirmed"
    assert cl.plan_edits([r], {"org/m": _row(license="bsd-3-clause")}) == {}


# --- apply_edits: line surgery on models.yaml ------------------------------

YAML_FIXTURE = """# header comment must survive
models:
  - name: A
    hf_repo: org/a
    license: apache-2.0
    commercial_use: true
    license_notes: AUTO-DISCOVERED — verify license terms.
    commercial_use_verified: false

  - name: B
    hf_repo: org/b
    license: mit
    commercial_use: true
"""


def _load(path):
    import yaml
    return {r["hf_repo"]: r for r in yaml.safe_load(path.read_text())["models"]}


def test_apply_replaces_an_existing_field(tmp_path):
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/a": {"commercial_use_verified": True}})
    assert _load(f)["org/a"]["commercial_use_verified"] is True


def test_apply_inserts_a_missing_field_into_the_right_row(tmp_path):
    """The bug this guards: a field the row does not already have has to land
    inside THAT row. Appending it when the next row's hf_repo appears puts it
    after the next `- name:` line, silently attaching it to the wrong model."""
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/a": {"license_text_published": False}})
    rows = _load(f)
    assert rows["org/a"]["license_text_published"] is False
    assert "license_text_published" not in rows["org/b"]


def test_apply_leaves_untargeted_rows_alone(tmp_path):
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/a": {"commercial_use_verified": True}})
    assert _load(f)["org/b"] == {"name": "B", "hf_repo": "org/b",
                                 "license": "mit", "commercial_use": True}


def test_apply_handles_the_last_row_in_the_file(tmp_path):
    """No following row to trigger a flush — the classic off-by-one here."""
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/b": {"license_text_published": False}})
    assert _load(f)["org/b"]["license_text_published"] is False


def test_apply_preserves_the_header_comment(tmp_path):
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/a": {"commercial_use_verified": True}})
    assert f.read_text().startswith("# header comment must survive")


def test_apply_quotes_a_string_with_a_colon(tmp_path):
    """A source citation contains a URL, so an unquoted scalar breaks YAML."""
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    cl.apply_edits(f, {"org/a": {"commercial_use_source":
                                 "LICENSE @ https://hf.co/x — verbatim"}})
    assert "https://hf.co/x" in _load(f)["org/a"]["commercial_use_source"]


def test_apply_reports_how_many_rows_changed(tmp_path):
    f = tmp_path / "models.yaml"; f.write_text(YAML_FIXTURE)
    n = cl.apply_edits(f, {"org/a": {"commercial_use_verified": True},
                           "org/b": {"license_text_published": False}})
    assert n == 2


# --- a licence link that contradicts the tag -------------------------------

def test_canonical_link_is_not_a_contradiction():
    """thinkingmachines points license_link at apache.org's own text. That
    corroborates the tag; it does not contradict it."""
    assert not cl.link_contradicts_tag(
        "https://www.apache.org/licenses/LICENSE-2.0", "apache-2.0", "org/m")
    assert not cl.link_contradicts_tag(
        "https://opensource.org/licenses/MIT", "mit", "org/m")


def test_a_link_into_the_repo_itself_is_not_a_contradiction():
    assert not cl.link_contradicts_tag(
        "https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/LICENSE",
        "apache-2.0", "Qwen/Qwen3.5-27B")


def test_a_vendor_licence_page_contradicts_a_permissive_tag():
    """Gemma 4 tags apache-2.0 and links to ai.google.dev/gemma/docs/
    gemma_4_license. A vendor hosting its own licence page is not publishing
    Apache-2.0, whatever the tag says."""
    assert cl.link_contradicts_tag(
        "https://ai.google.dev/gemma/docs/gemma_4_license",
        "apache-2.0", "google/gemma-4-12B-it")


def test_no_link_is_not_a_contradiction():
    assert not cl.link_contradicts_tag(None, "apache-2.0", "org/m")
    assert not cl.link_contradicts_tag("", "apache-2.0", "org/m")


def test_a_bespoke_tag_with_a_vendor_link_is_consistent():
    """Only a PERMISSIVE tag is contradicted by a bespoke licence page. A row
    already claiming a vendor licence and linking to it agrees with itself."""
    assert not cl.link_contradicts_tag(
        "https://ai.google.dev/gemma/terms", "gemma", "google/gemma-3-27b-it")


def test_verdict_reports_the_contradiction_over_tag_only():
    """It must outrank tag-only: 'no licence file' understates a row whose
    own metadata points at a different licence."""
    r = _report(licence_files=[], identified=None,
                license_link="https://ai.google.dev/gemma/docs/gemma_4_license")
    assert cl.verdict(r, "apache-2.0") == "link-contradicts-tag"
