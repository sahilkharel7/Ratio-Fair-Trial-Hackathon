"""Hard rule 3: no verdicts about people. All on-screen wording comes from reviewed templates, and
model-generated notes are filtered for characterisations of people."""

import re

import pytest

from ratio.config import load_config
from ratio.messages import REMOVED, contains_blocked_term, filter_model_note, note, render


@pytest.fixture(scope="module")
def messages():
    return load_config().messages


def test_no_template_or_note_contains_a_blocked_term(messages):
    for text in list(messages.templates.values()) + list(messages.notes.values()):
        assert not contains_blocked_term(text, messages.block_list), text


def test_render_fills_every_placeholder(messages):
    text = render(messages, "judge_hidden", n=3, min=5)
    assert text == "Not shown: 3 cases (fewer than 5)."
    assert not re.search(r"{\w+}", text)


def test_render_rejects_unknown_template_and_missing_values(messages):
    with pytest.raises(KeyError):
        render(messages, "no_such_template")
    with pytest.raises(KeyError):
        render(messages, "judge_hidden", n=3)


def test_filter_removes_characterisations_of_people(messages):
    filtered = filter_model_note("The judge was clearly biased and corrupt.", messages.block_list)
    assert filtered is not None
    assert "biased" not in filtered and "corrupt" not in filtered
    assert REMOVED in filtered


def test_filter_keeps_neutral_text_and_word_parts(messages):
    assert filter_model_note("The record is unbiased on this point.", messages.block_list) == (
        "The record is unbiased on this point."
    )


def test_filter_returns_none_for_blank_notes(messages):
    assert filter_model_note("   ", messages.block_list) is None
    assert filter_model_note(None, messages.block_list) is None


def test_selection_bias_note_is_explicit(messages):
    text = note(messages, "selection_bias")
    assert "suspected of unfairness" in text
    assert "not representative" in text
    assert "reviewing lawyer" in text


@pytest.mark.parametrize(
    "text",
    ["shows bias", "BIASED ruling", "bi\u00adased", "corruption", "he lies", "a rubber stamp", "acted in bad faith", "prejudice"],
)
def test_word_forms_and_hidden_characters_are_caught(messages, text):
    assert contains_blocked_term(text, messages.block_list)


@pytest.mark.parametrize("text", ["unbiased", "unfairness of the trial", "partial closure of the hearing", "the court applies"])
def test_ordinary_wording_is_not_caught(messages, text):
    assert not contains_blocked_term(text, messages.block_list)
