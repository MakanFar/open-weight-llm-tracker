import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import enrich

CARDS = Path(__file__).resolve().parent / "fixtures" / "cards"


def _card(name):
    return (CARDS / name).read_text()


def test_reads_parenthesised_activated_figure():
    got = enrich.active_params_from_card(_card("deepseek.md"))
    assert got is not None
    value, quote = got
    assert value == 49.0
    assert "49B activated" in quote


def test_reads_approximate_activated_parameters():
    value, _ = enrich.active_params_from_card(_card("minimax.md"))
    assert value == 23.0


def test_reads_active_parameters_phrasing():
    value, _ = enrich.active_params_from_card(_card("hy3.md"))
    assert value == 21.0


def test_reads_a_table_cell():
    value, _ = enrich.active_params_from_card(_card("kimi.md"))
    assert value == 104.0


def test_returns_none_when_the_card_states_no_figure():
    """None means 'ask a human', never 0 — a 0 would look like a real value."""
    assert enrich.active_params_from_card(_card("nofigure.md")) is None


def test_returns_none_when_a_shared_card_states_two_distinct_figures():
    """A card describing multiple variants (e.g. Pro/Flash) is ambiguous — the
    function has no way to know which variant the caller means, so guessing
    wrong and silently publishing it is far worse than asking a human."""
    assert enrich.active_params_from_card(_card("two_variants.md")) is None


def test_repeated_mentions_of_the_same_figure_are_not_ambiguous():
    """The same number said twice is not a conflict — only distinct values are."""
    text = (
        "Model-X has 23B activated parameters in the abstract. "
        "Later, the table repeats: ~23B activated parameters."
    )
    value, _ = enrich.active_params_from_card(text)
    assert value == 23.0


def test_unit_t_converts_to_billions():
    """T (trillion) must scale UP by 1000x — untested until now, so an inverted
    _UNIT_TO_B multiplier would only misfire on the very largest models."""
    value, _ = enrich.active_params_from_card("Model-Y (1.6T activated)")
    assert value == 1600.0


def test_unit_m_converts_to_billions():
    """M (million) must scale DOWN by 1000x, the inverse of the T case above."""
    value, _ = enrich.active_params_from_card("Model-Z (800M activated)")
    assert value == 0.8


def test_returns_none_for_empty_input():
    assert enrich.active_params_from_card("") is None
    assert enrich.active_params_from_card(None) is None


def test_fetch_card_returns_none_on_failure():
    def boom(url):
        raise RuntimeError("404")
    assert enrich.fetch_card("org/m", get_text=boom) is None


def test_fetch_card_returns_the_body():
    assert enrich.fetch_card("org/m", get_text=lambda url: "# card") == "# card"


class FakeInfo:
    def __init__(self, card_data):
        self.card_data = card_data


def test_license_passes_through_a_usable_tag():
    assert enrich.license_string(FakeInfo({"license": "mit"})) == "mit"
    assert enrich.license_string(FakeInfo({"license": "apache-2.0"})) == "apache-2.0"


def test_license_maps_hf_llama_tags_to_allowlist_spellings():
    """HF tags say llama3.1; validate.LICENSES says llama-3.1-community."""
    assert enrich.license_string(FakeInfo({"license": "llama3.1"})) == "llama-3.1-community"
    assert enrich.license_string(FakeInfo({"license": "llama3.3"})) == "llama-3.3-community"
    assert enrich.license_string(FakeInfo({"license": "llama4"})) == "llama-4-community"


def test_license_recovers_the_real_name_when_the_tag_is_other():
    """'other' is not a licence. cardData.license_name carries the real one."""
    info = FakeInfo({"license": "other", "license_name": "kimi-k3"})
    assert enrich.license_string(info) == "kimi-k3"


def test_license_normalises_a_recovered_name():
    info = FakeInfo({"license": "other", "license_name": "MiniMax Community"})
    assert enrich.license_string(info) == "minimax-community"


def test_license_returns_none_when_other_has_no_name():
    assert enrich.license_string(FakeInfo({"license": "other"})) is None


def test_license_returns_none_when_absent():
    assert enrich.license_string(FakeInfo({})) is None
    assert enrich.license_string(FakeInfo(None)) is None


def test_context_from_tokenizer_config():
    cfg = {"model_max_length": 262144}
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: cfg) == 262144


def test_context_ignores_the_sentinel_length():
    """Transformers writes a huge int32 sentinel meaning 'unset'."""
    cfg = {"model_max_length": 1000000000000000019884624838656}
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: cfg) is None


def test_context_returns_none_when_absent_or_unfetchable():
    assert enrich.context_from_tokenizer("org/m", get_json=lambda u: {}) is None

    def boom(url):
        raise RuntimeError("404")
    assert enrich.context_from_tokenizer("org/m", get_json=boom) is None


def test_context_from_tokenizer_records_a_gated_failure():
    """The tokenizer fetch 403s on a gated repo just as config.json does, so
    it can report the access problem without costing an extra request."""
    import urllib.error
    notes = {}
    def boom(url):
        raise urllib.error.HTTPError(url, 403, "no", {}, None)
    assert enrich.context_from_tokenizer("google/gemma-3-12b-it", boom,
                                         notes=notes) is None
    assert notes.get("gated") is True


def test_context_from_tokenizer_ignores_other_failures():
    notes = {}
    def boom(url): raise TimeoutError("slow")
    assert enrich.context_from_tokenizer("org/m", boom, notes=notes) is None
    assert notes == {}


# --- phrasings the parser did not recognise, each from a real card ---------

def test_reads_a_bare_activated_figure_with_no_following_noun():
    """Qwen states 'N in total and M activated' with no 'parameters' after it.

    The pattern required activ* to be followed by 'param', so every Qwen MoE
    card fell through and six flagship rows parked on
    moe-active-params-unknown.
    """
    text = "- Number of Parameters: 397B in total and 17B activated"
    value, quote = enrich.active_params_from_card(text)
    assert value == 17.0
    assert "17B activated" in quote


def test_reads_the_total_comma_active_phrasing():
    """thinkingmachines/Inkling: '975B total, 41B active'."""
    value, _ = enrich.active_params_from_card("975B total, 41B active")
    assert value == 41.0


def test_reads_a_table_cell_wrapped_in_markdown_emphasis():
    """Gemma 4 writes '| **Active Parameters** | 3.8B |'. The old table
    pattern demanded the pipe immediately after 'Parameters', so the bold
    markers defeated it and the 10M-download row stayed unfilled."""
    value, _ = enrich.active_params_from_card("| **Active Parameters** | 3.8B |")
    assert value == 3.8


def test_reads_the_a_notation_written_in_prose():
    """GLM-4.7-Flash states its size only as '30B-A3B'."""
    value, quote = enrich.active_params_from_card(
        "GLM-4.7-Flash is a 30B-A3B MoE model.")
    assert value == 3.0
    assert "30B-A3B" in quote


# --- disambiguating a card that documents a whole family -------------------

def test_a_family_card_resolves_when_the_row_total_names_one_variant():
    """two_variants.md documents Pro (1.6T/49B) and Flash (284B/13B).

    Given the row's own total, only one variant is a plausible match, so the
    figure is no longer a guess — it is the one the card pairs with this
    model's size.
    """
    value, quote = enrich.active_params_from_card(
        _card("two_variants.md"), total_b=290.9)
    assert value == 13.0
    assert "284B" in quote


def test_a_family_card_still_abstains_when_no_variant_matches_the_total():
    """A total matching neither variant means we still do not know which row
    the card is describing — the tie-break must not fall back to guessing."""
    assert enrich.active_params_from_card(
        _card("two_variants.md"), total_b=50.0) is None


def test_a_family_card_still_abstains_when_two_variants_are_both_plausible():
    text = ("Model-A with 100B parameters (10B activated) and "
            "Model-B with 104B parameters (12B activated).")
    assert enrich.active_params_from_card(text, total_b=102.0) is None


def test_the_total_tiebreak_never_overrides_an_unambiguous_card():
    """A card stating one figure resolves to it regardless of the total —
    the tie-break exists only to break ties."""
    value, _ = enrich.active_params_from_card("Model-X (23B activated)",
                                              total_b=999.0)
    assert value == 23.0


# --- the vendor's own A-notation in the repo id ----------------------------

def test_repo_name_yields_the_a_notation_figure():
    """Qwen3-VL-235B-A22B states its activation nowhere in the card; the
    vendor put it in the repo name, which is as authoritative a statement."""
    value, quote = enrich.active_params_from_repo_name(
        "Qwen/Qwen3-VL-235B-A22B-Instruct")
    assert value == 22.0
    assert "235B-A22B" in quote


def test_repo_name_requires_the_paired_total_and_active_shape():
    """A bare 'A<n>' is not the notation and must not be read as one, or an
    accelerator name or revision tag becomes an activation figure."""
    assert enrich.active_params_from_repo_name("openai/gpt-oss-120b") is None
    assert enrich.active_params_from_repo_name(
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct") is None
    assert enrich.active_params_from_repo_name("org/model-A100-tuned") is None
    assert enrich.active_params_from_repo_name("") is None
    assert enrich.active_params_from_repo_name(None) is None


def test_repo_name_abstains_when_its_total_contradicts_the_row():
    """The name encodes a total too. If that disagrees with the measured
    total the name is describing a different model (a distill, a mirror), so
    its activation figure cannot be trusted onto this row."""
    assert enrich.active_params_from_repo_name(
        "google/gemma-4-26B-A4B-it", total_b=250.0) is None
    value, _ = enrich.active_params_from_repo_name(
        "google/gemma-4-26B-A4B-it", total_b=26.5)
    assert value == 4.0


def test_a_precise_figure_beats_the_rounded_a_notation_on_the_same_page():
    """Gemma 4's card prints its own repo id ('26B-A4B') beside a table cell
    reading 3.8B. Those are the same fact at two precisions, not two rival
    claims -- so the rounded one must not win the tie-break and publish 4.0
    for a 3.8B model."""
    text = ("- google/gemma-4-26B-A4B\n"
            "| Property | 26B A4B MoE |\n"
            "| **Active Parameters** | 3.8B |")
    value, _ = enrich.active_params_from_card(text, total_b=26.5)
    assert value == 3.8


# --- the total the quote asserts, reported alongside it --------------------

def test_card_reports_the_total_it_quoted():
    """A quote like '284B parameters (13B activated)' asserts a TOTAL as well.
    Storing the sentence without that number is what made a row contradict
    itself: params_total_b said 290.9 beside a quote plainly saying 284."""
    notes = {}
    value, _ = enrich.active_params_from_card(
        "**Flash** with 284B parameters (13B activated)",
        total_b=290.9, notes=notes)
    assert value == 13.0
    assert notes["stated_total"] == 284.0


def test_no_stated_total_when_the_quote_asserts_none():
    """'~23B activated parameters' names no total. Recording one would be
    inventing a vendor claim that was never made."""
    notes = {}
    enrich.active_params_from_card("~23B activated parameters", notes=notes)
    assert "stated_total" not in notes


def test_nothing_recorded_when_the_card_abstains():
    notes = {}
    assert enrich.active_params_from_card(_card("two_variants.md"),
                                          notes=notes) is None
    assert notes == {}


def test_the_rounded_tier_reports_its_total_too():
    notes = {}
    enrich.active_params_from_card("GLM-4.7-Flash is a 30B-A3B MoE model.",
                                   notes=notes)
    assert notes["stated_total"] == 30.0


def test_repo_name_reports_its_stated_total():
    notes = {}
    enrich.active_params_from_repo_name("Qwen/Qwen3-VL-235B-A22B-Instruct",
                                        notes=notes)
    assert notes["stated_total"] == 235.0


def test_repo_name_records_nothing_when_it_abstains():
    notes = {}
    assert enrich.active_params_from_repo_name(
        "google/gemma-4-26B-A4B-it", total_b=250.0, notes=notes) is None
    assert notes == {}
