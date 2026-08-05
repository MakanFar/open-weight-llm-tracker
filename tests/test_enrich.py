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
