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


def test_returns_none_for_empty_input():
    assert enrich.active_params_from_card("") is None
    assert enrich.active_params_from_card(None) is None


def test_fetch_card_returns_none_on_failure():
    def boom(url):
        raise RuntimeError("404")
    assert enrich.fetch_card("org/m", get_text=boom) is None


def test_fetch_card_returns_the_body():
    assert enrich.fetch_card("org/m", get_text=lambda url: "# card") == "# card"
