#!/usr/bin/env python3
# coding: utf-8

import os
import tempfile

import pytest

from deep_reference_parser.reference_utils import (
    break_into_chunks,
    load_tsv,
    prodigy_to_conll,
    write_tsv,
    yield_token_label_pairs,
    split_list_by_linebreaks,
)

from .common import TEST_TSV_PREDICT, TEST_TSV_TRAIN, TEST_LOAD_TSV


def test_prodigy_to_conll():

    before = [
        {"text": "References",},
        {"text": "37. No single case of malaria reported in"},
        {
            "text": "an essential requirement for the correct labelling of potency for therapeutic"
        },
        {"text": "EQAS, quality control for STI"},
    ]

    after = "DOCSTART\n\nReferences\n\n37\n.\nNo\nsingle\ncase\nof\nmalaria\nreported\nin\n\nan\nessential\nrequirement\nfor\nthe\ncorrect\nlabelling\nof\npotency\nfor\ntherapeutic\n\nEQAS\n,\nquality\ncontrol\nfor\nSTI"

    out = prodigy_to_conll(before)

    assert after == out


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


def test_write_tsv():

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

    _, path = tempfile.mkstemp()

    token_label_tuples = list(yield_token_label_pairs(expected[0], expected[1]))

    write_tsv(token_label_tuples, path)
    actual = load_tsv(path)

    assert expected == actual

    os.remove(path)


def test_break_into_chunks():

    before = ["a", "b", "c", "d", "e"]
    expected = [["a", "b"], ["c", "d"], ["e"]]

    actual = break_into_chunks(before, max_words=2)

    assert expected == actual

def test_split_list_by_linebreaks():

    lst = ["a", "b", "c", None, "d"]
    expected = [["a", "b", "c"], ["d"]]

    actual = split_list_by_linebreaks(lst)

def test_list_by_linebreaks_ending_in_None():

    lst = ["a", "b", "c", float("nan"), "d", None]
    expected = [["a", "b", "c"], ["d"]]

    actual = split_list_by_linebreaks(lst)

def test_list_by_linebreaks_starting_in_None():

    lst = [None, "a", "b", "c", None, "d"]
    expected = [["a", "b", "c"], ["d"]]

    actual = split_list_by_linebreaks(lst)
