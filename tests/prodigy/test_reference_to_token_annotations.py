#!/usr/bin/env python3
# coding: utf-8

import pytest

from deep_reference_parser.io import read_jsonl
from deep_reference_parser.prodigy.reference_to_token_annotations import TokenTagger

from .common import TEST_REF_EXPECTED_SPANS, TEST_REF_SPANS, TEST_REF_TOKENS


@pytest.fixture(scope="function")
def splitter():
    return TokenTagger(task="splitting", text=False)


@pytest.fixture(scope="function")
def parser():
    return TokenTagger(task="parsing", text=True)


@pytest.fixture(scope="module")
def doc():
    doc = {}
    doc["tokens"] = read_jsonl(TEST_REF_TOKENS)[0]
    doc["spans"] = read_jsonl(TEST_REF_SPANS)[0]

    return doc


@pytest.fixture(scope="module")
def expected():
    spans = read_jsonl(TEST_REF_EXPECTED_SPANS)

    return spans


def test_TokenTagger(splitter):

    doc = dict()

    doc["spans"] = [
        {"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "BE"},
    ]

    doc["tokens"] = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    out = [
        {"start": 0, "end": 0, "token_start": 0, "token_end": 0, "label": "o"},
        {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "o"},
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e-r"},
        {"start": 5, "end": 5, "token_start": 5, "token_end": 5, "label": "o"},
        {"start": 6, "end": 6, "token_start": 6, "token_end": 6, "label": "o"},
    ]

    tagged = splitter.run([doc])

    assert out == tagged[0]["spans"]


def test_create_span(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
    ]

    after = {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "foo"}

    out = splitter.create_span(tokens=tokens, index=1, label="foo")

    assert out == after


def test_split_long_span_three_token_span(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    span = {"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "BE"}

    expected = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e-r"},
    ]

    actual = splitter.split_long_span(
        tokens, span, start_label="b-r", end_label="e-r", inside_label="i-r"
    )

    assert expected == actual


def test_split_long_span_two_token_span(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    span = {"start": 2, "end": 3, "token_start": 2, "token_end": 3, "label": "BE"}

    expected = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "e-r"},
    ]

    actual = splitter.split_long_span(
        tokens, span, start_label="b-r", end_label="e-r", inside_label="i-r"
    )

    assert expected == actual


def test_split_long_span_one_token_span(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    span = {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "BE"}

    expected = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
    ]

    actual = splitter.split_long_span(
        tokens, span, start_label="b-r", end_label="e-r", inside_label="i-r"
    )

    assert expected == actual


def test_reference_spans_be(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [{"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "BE"}]

    after = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e-r"},
    ]

    out = splitter.reference_spans(spans, tokens, task="splitting")

    assert out == after


def test_reference_spans_bi(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [{"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "BI"}]

    after = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "i-r"},
    ]

    out = splitter.reference_spans(spans, tokens, task="splitting")

    assert out == after


def test_reference_spans_ie(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [{"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "IE"}]

    after = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "i-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e-r"},
    ]

    out = splitter.reference_spans(spans, tokens, task="splitting")

    assert out == after


def test_reference_spans_ii(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [{"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "II"}]

    after = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "i-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "i-r"},
    ]

    out = splitter.reference_spans(spans, tokens, task="splitting")

    assert out == after


def test_reference_spans_parsing(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [
        {"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "author"}
    ]

    after = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "author"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "author"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "author"},
    ]

    out = splitter.reference_spans(spans, tokens, task="parsing")

    assert out == after


def test_reference_spans_parsing_single_token(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "author"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "year"},
    ]

    expected = [
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "author"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "year"},
    ]

    actual = splitter.reference_spans(spans, tokens, task="parsing")

    print(actual)

    assert actual == expected


def test_outside_spans(splitter):

    tokens = [
        {"start": 0, "end": 0, "id": 0},
        {"start": 1, "end": 1, "id": 1},
        {"start": 2, "end": 2, "id": 2},
        {"start": 3, "end": 3, "id": 3},
        {"start": 4, "end": 4, "id": 4},
        {"start": 5, "end": 5, "id": 5},
        {"start": 6, "end": 6, "id": 6},
    ]

    spans = [
        {"start": 2, "end": 4, "token_start": 2, "token_end": 4, "label": "b-r"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "i-r"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e-r"},
    ]

    after = [
        {"start": 0, "end": 0, "token_start": 0, "token_end": 0, "label": "o"},
        {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "o"},
        {"start": 5, "end": 5, "token_start": 5, "token_end": 5, "label": "o"},
        {"start": 6, "end": 6, "token_start": 6, "token_end": 6, "label": "o"},
    ]

    out = splitter.outside_spans(spans, tokens)

    assert out == after


def test_reference_spans_real_example(doc, parser, expected):

    actual = parser.run([doc])[0]["spans"]
    assert actual == expected
