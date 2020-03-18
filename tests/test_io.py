#!/usr/bin/env python3
# coding: utf-8

import os

import pytest

from deep_reference_parser.io.io import read_jsonl, write_jsonl, load_tsv, write_tsv, _split_list_by_linebreaks
from deep_reference_parser.reference_utils import yield_token_label_pairs

from .common import TEST_JSONL, TEST_TSV_TRAIN, TEST_TSV_PREDICT, TEST_LOAD_TSV


@pytest.fixture(scope="module")
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


def test_write_tsv(tmpdir):

    expected = (
        [
            [],
            ["the", "focus", "in", "Daloa", ",", "Côte", "d’Ivoire]."],
            ["Bulletin", "de", "la", "Société", "de", "Pathologie"],
            ["Exotique", "et"],
        ],
        [
            [],
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r"],
        ],
    )

    token_label_tuples = list(yield_token_label_pairs(expected[0], expected[1]))

    PATH = os.path.join(tmpdir, "test_tsv.tsv")
    write_tsv(token_label_tuples, PATH)
    actual = load_tsv(os.path.join(PATH))

    assert expected == actual

def test_load_tsv_train():
    """
    Text of TEST_TSV_TRAIN:

    ```
        the	i-r
        focus	i-r
        in	i-r
        Daloa	i-r
        ,	i-r
        Côte	i-r
        d’Ivoire].	i-r

        Bulletin	i-r
        de	i-r
        la	i-r
        Société	i-r
        de	i-r
        Pathologie	i-r

        Exotique	i-r
        et	i-r
    ```
    """

    expected = (
        [
            ["the", "focus", "in", "Daloa", ",", "Côte", "d’Ivoire]."],
            ["Bulletin", "de", "la", "Société", "de", "Pathologie"],
            ["Exotique", "et"],
        ],
        [
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r"],
        ],
    )

    actual = load_tsv(TEST_TSV_TRAIN)

    assert len(actual[0][0]) == len(expected[0][0])
    assert len(actual[0][1]) == len(expected[0][1])
    assert len(actual[0][2]) == len(expected[0][2])

    assert len(actual[1][0]) == len(expected[1][0])
    assert len(actual[1][1]) == len(expected[1][1])
    assert len(actual[1][2]) == len(expected[1][2])

    assert actual == expected


def test_load_tsv_predict():
    """
    Text of TEST_TSV_PREDICT:

    ```
        the
        focus
        in
        Daloa
        ,
        Côte
        d’Ivoire].

        Bulletin
        de
        la
        Société
        de
        Pathologie

        Exotique
        et
    ```
    """

    expected = (
        [
            ["the", "focus", "in", "Daloa", ",", "Côte", "d’Ivoire]."],
            ["Bulletin", "de", "la", "Société", "de", "Pathologie"],
            ["Exotique", "et"],
        ],
    )

    actual = load_tsv(TEST_TSV_PREDICT)

    assert actual == expected

def test_load_tsv_train_multiple_labels():
    """
    Text of TEST_TSV_TRAIN:

    ```
        the	i-r
        focus	i-r
        in	i-r
        Daloa	i-r
        ,	i-r
        Côte	i-r
        d’Ivoire].	i-r

        Bulletin	i-r
        de	i-r
        la	i-r
        Société	i-r
        de-r
        Pathologie	i-r

        Exotique	i-r
        et	i-r
    ```
    """

    expected = (
        [
            ["the", "focus", "in", "Daloa", ",", "Côte", "d’Ivoire]."],
            ["Bulletin", "de", "la", "Société", "de", "Pathologie"],
            ["Exotique", "et"],
        ],
        [
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
            ["i-r", "i-r"],
        ],
        [
            ["a", "a", "a", "a", "a", "a", "a"],
            ["a", "a", "a", "a", "a", "a"],
            ["a", "a"],
        ],
    )

    actual = load_tsv(TEST_LOAD_TSV)

    assert actual == expected


def test_yield_toke_label_pairs():

    tokens = [
        [],
        ["the", "focus", "in", "Daloa", ",", "Côte", "d’Ivoire]."],
        ["Bulletin", "de", "la", "Société", "de", "Pathologie"],
        ["Exotique", "et"],
    ]

    labels = [
        [],
        ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
        ["i-r", "i-r", "i-r", "i-r", "i-r", "i-r"],
        ["i-r", "i-r"],
    ]

    expected = [
        (None, None),
        ("the", "i-r"),
        ("focus", "i-r"),
        ("in", "i-r"),
        ("Daloa", "i-r"),
        (",", "i-r"),
        ("Côte", "i-r"),
        ("d’Ivoire].", "i-r"),
        (None, None),
        ("Bulletin", "i-r"),
        ("de", "i-r"),
        ("la", "i-r"),
        ("Société", "i-r"),
        ("de", "i-r"),
        ("Pathologie", "i-r"),
        (None, None),
        ("Exotique", "i-r"),
        ("et", "i-r"),
        (None, None),
    ]

    actual = list(yield_token_label_pairs(tokens, labels))

    assert expected == actual

def test_read_jsonl():

    expected = [
        {
            "text": "a b c\n a b c",
            "tokens": [
                {"text": "a", "start": 0, "end": 1, "id": 0},
                {"text": "b", "start": 2, "end": 3, "id": 1},
                {"text": "c", "start": 4, "end": 5, "id": 2},
                {"text": "\n ", "start": 5, "end": 7, "id": 3},
                {"text": "a", "start": 7, "end": 8, "id": 4},
                {"text": "b", "start": 9, "end": 10, "id": 5},
                {"text": "c", "start": 11, "end": 12, "id": 6},
            ],
            "spans": [
                {"start": 2, "end": 3, "token_start": 1, "token_end": 2, "label": "b"},
                {"start": 4, "end": 5, "token_start": 2, "token_end": 3, "label": "i"},
                {"start": 7, "end": 8, "token_start": 4, "token_end": 5, "label": "i"},
                {"start": 9, "end": 10, "token_start": 5, "token_end": 6, "label": "e"},
            ],
        }
    ]

    expected = expected * 3

    actual = read_jsonl(TEST_JSONL)
    assert expected == actual


def test_write_jsonl(tmpdir):

    expected = [
        {
            "text": "a b c\n a b c",
            "tokens": [
                {"text": "a", "start": 0, "end": 1, "id": 0},
                {"text": "b", "start": 2, "end": 3, "id": 1},
                {"text": "c", "start": 4, "end": 5, "id": 2},
                {"text": "\n ", "start": 5, "end": 7, "id": 3},
                {"text": "a", "start": 7, "end": 8, "id": 4},
                {"text": "b", "start": 9, "end": 10, "id": 5},
                {"text": "c", "start": 11, "end": 12, "id": 6},
            ],
            "spans": [
                {"start": 2, "end": 3, "token_start": 1, "token_end": 2, "label": "b"},
                {"start": 4, "end": 5, "token_start": 2, "token_end": 3, "label": "i"},
                {"start": 7, "end": 8, "token_start": 4, "token_end": 5, "label": "i"},
                {"start": 9, "end": 10, "token_start": 5, "token_end": 6, "label": "e"},
            ],
        }
    ]

    expected = expected * 3

    temp_file = os.path.join(tmpdir, "file.jsonl")

    write_jsonl(expected, temp_file)
    actual = read_jsonl(temp_file)

    assert expected == actual

def test_split_list_by_linebreaks():

    lst = ["a", "b", "c", None, "d"]
    expected = [["a", "b", "c"], ["d"]]

    actual = _split_list_by_linebreaks(lst)

def test_list_by_linebreaks_ending_in_None():

    lst = ["a", "b", "c", float("nan"), "d", None]
    expected = [["a", "b", "c"], ["d"]]

    actual = _split_list_by_linebreaks(lst)

def test_list_by_linebreaks_starting_in_None():

    lst = [None, "a", "b", "c", None, "d"]
    expected = [["a", "b", "c"], ["d"]]

    actual = _split_list_by_linebreaks(lst)
