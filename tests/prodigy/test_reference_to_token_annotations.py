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


    doc = dict()

    doc["spans"] = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'BE'},
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
        {'start': 0, 'end': 0, 'token_start': 0, 'token_end': 0, 'label': 'o'},
        {'start': 1, 'end': 1, 'token_start': 1, 'token_end': 1, 'label': 'o'},
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'b-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'e-r'},
        {'start': 5, 'end': 5, 'token_start': 5, 'token_end': 5, 'label': 'o'},
        {'start': 6, 'end': 6, 'token_start': 6, 'token_end': 6, 'label': 'o'},
    ]

    tagged = tagger.run([doc])

    assert out == tagged[0]["spans"]


def test_create_span(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
    ]

    after = {'start': 1, 'end': 1, 'token_start': 1, 'token_end': 1, 'label': 'foo'}

    out = tagger.create_span(tokens=tokens, index=1, label="foo")

    assert out == after


def test_split_long_span(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    span = {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'BE'}

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'b-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'e-r'},
    ]

    out = tagger.split_long_span(tokens, span, start_label="b-r", end_label="e-r", inside_label="i-r")

    assert out == after


def test_reference_spans_be(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'BE'}
    ]

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'b-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'e-r'},
    ]

    out = tagger.reference_spans(spans, tokens, task="splitting")

    assert out == after

def test_reference_spans_bi(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'BI'}
    ]

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'b-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'i-r'},
    ]

    out = tagger.reference_spans(spans, tokens, task="splitting")

    assert out == after

def test_reference_spans_ie(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'IE'}
    ]

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'i-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'e-r'},
    ]

    out = tagger.reference_spans(spans, tokens, task="splitting")

    assert out == after

def test_reference_spans_ii(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'II'}
    ]

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'i-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'i-r'},
    ]

    out = tagger.reference_spans(spans, tokens, task="splitting")

    assert out == after

def test_reference_spans_author(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'author'}
    ]

    after = [
        {'start': 2, 'end': 2, 'token_start': 2, 'token_end': 2, 'label': 'author'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'author'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'author'},
    ]

    out = tagger.reference_spans(spans, tokens, task="parsing")

    assert out == after


def test_outside_spans(tagger):

    tokens = [
        {'start': 0, 'end': 0, 'id': 0},
        {'start': 1, 'end': 1, 'id': 1},
        {'start': 2, 'end': 2, 'id': 2},
        {'start': 3, 'end': 3, 'id': 3},
        {'start': 4, 'end': 4, 'id': 4},
        {'start': 5, 'end': 5, 'id': 5},
        {'start': 6, 'end': 6, 'id': 6},
    ]

    spans = [
        {'start': 2, 'end': 4, 'token_start': 2, 'token_end': 4, 'label': 'b-r'},
        {'start': 3, 'end': 3, 'token_start': 3, 'token_end': 3, 'label': 'i-r'},
        {'start': 4, 'end': 4, 'token_start': 4, 'token_end': 4, 'label': 'e-r'},
    ]

    after = [
        {'start': 0, 'end': 0, 'token_start': 0, 'token_end': 0, 'label': 'o'},
        {'start': 1, 'end': 1, 'token_start': 1, 'token_end': 1, 'label': 'o'},
        {'start': 5, 'end': 5, 'token_start': 5, 'token_end': 5, 'label': 'o'},
        {'start': 6, 'end': 6, 'token_start': 6, 'token_end': 6, 'label': 'o'},
    ]

    out = tagger.outside_spans(spans, tokens)

    assert out == after
def test_reference_spans_real_example(doc, parser, expected):

    actual = parser.run([doc])[0]["spans"]
    assert actual == expected
