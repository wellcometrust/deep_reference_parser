#!/usr/bin/env python3
# coding: utf-8

import os
import sys
from copy import deepcopy

import pytest
from deep_reference_parser.io import load_tsv, read_jsonl
from deep_reference_parser.prodigy.prodigy_to_tsv import (TokenLabelPairs,
                                                          prodigy_to_tsv)

from .common import (TEST_SPANS, TEST_TOKEN_LABELLED, TEST_TOKEN_LABELLED_TSV,
                     TEST_TOKENS)


@pytest.fixture(scope="session")
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")

@pytest.fixture(scope="module")
def doc():
    doc = {}
    doc["_input_hash"] = 1337
    doc["tokens"] = read_jsonl(TEST_TOKENS)[0]
    doc["spans"] = read_jsonl(TEST_SPANS)[0]

    return doc

@pytest.fixture(scope="module")
def spans():
    spans = [
        {"start": 0, "end": 0, "token_start": 0, "token_end": 0, "label": "a"},
        {"start": 1, "end": 1, "token_start": 1, "token_end": 1, "label": "b"},
        {"start": 2, "end": 2, "token_start": 2, "token_end": 2, "label": "c"},
        {"start": 3, "end": 3, "token_start": 3, "token_end": 3, "label": "d"},
        {"start": 4, "end": 4, "token_start": 4, "token_end": 4, "label": "e"},
        {"start": 5, "end": 5, "token_start": 5, "token_end": 5, "label": "f"},
        {"start": 6, "end": 6, "token_start": 6, "token_end": 6, "label": "g"},
    ]

    return spans

@pytest.fixture(scope="module")
def tokens():
    tokens = [
        {"text": "A", "start": 0, "end": 0, "id": 0},
        {"text": "B", "start": 1, "end": 1, "id": 1},
        {"text": "C", "start": 2, "end": 2, "id": 2},
        {"text": "D", "start": 3, "end": 3, "id": 3},
        {"text": "E", "start": 4, "end": 4, "id": 4},
        {"text": "F", "start": 5, "end": 5, "id": 5},
        {"text": "G", "start": 6, "end": 6, "id": 6},
    ]

    return tokens

@pytest.fixture(scope="module")
def dataset(tokens, spans):

    doc0 = dict()

    doc0["spans"] = spans
    doc0["tokens"] = tokens

    doc1 = deepcopy(doc0)
    doc2 = deepcopy(doc0)

    doc0["_input_hash"] = 76786
    doc1["_input_hash"] = 65876
    doc2["_input_hash"] = 68457
    dataset = [doc0, doc1, doc2]

    return dataset

def test_yield_token_label_pair(tokens, spans):

    expected = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        (None, None),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)

    tokens = [token["text"] for token in tokens]
    spans = [span["label"] for span in spans]

    tokens_and_labels = list(zip(*[tokens, spans]))
    actual = list(tlp.yield_token_label_pair(tokens_and_labels))

    assert expected == actual

def test_TokenLabelPairs_single_dataset(dataset):

    datasets = [dataset]

    expected = [
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
        ("A", "a"),
        ("B", "b"),
        ("C", "c"),
        ("D", "d"),
        ("E", "e"),
        ("F", "f"),
        ("G", "g"),
    ]

    tlp = TokenLabelPairs(
        line_limit=73, respect_line_endings=True, respect_doc_endings=False
    )
    actual = tlp.run(datasets)

    assert actual == expected


def test_TokenLabelPairs_two_datasets(dataset):

    datasets = [dataset, dataset]

    expected = [
        ("A", "a", "a"),
        ("B", "b", "b"),
        ("C", "c", "c"),
        ("D", "d", "d"),
        ("E", "e", "e"),
        ("F", "f", "f"),
        ("G", "g", "g"),
        ("A", "a", "a"),
        ("B", "b", "b"),
        ("C", "c", "c"),
        ("D", "d", "d"),
        ("E", "e", "e"),
        ("F", "f", "f"),
        ("G", "g", "g"),
        ("A", "a", "a"),
        ("B", "b", "b"),
        ("C", "c", "c"),
        ("D", "d", "d"),
        ("E", "e", "e"),
        ("F", "f", "f"),
        ("G", "g", "g"),
    ]

    tlp = TokenLabelPairs(
        line_limit=73, respect_line_endings=True, respect_doc_endings=False
    )
    actual = tlp.run(datasets)

    assert actual == expected


def test_TokenLabelPairs_works_on_unlabelled(dataset):

    [doc.pop("spans") for doc in dataset]

    datasets = [dataset]

    expected = [
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        (None,),
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        (None,),
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        (None,),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)
    actual = tlp.run(datasets)

    assert actual == expected


def test_TokenLabelPairs_cleans_whitespace():

    doc = dict()

    doc["_input_hash"] = 1337
    doc["tokens"] = [
        {"text": "A ", "start": 0, "end": 0, "id": 0},
        {"text": "B  ", "start": 1, "end": 1, "id": 1},
        {"text": "C   ", "start": 2, "end": 2, "id": 2},
        {"text": "D\t", "start": 3, "end": 3, "id": 3},
        {"text": "E\t\t", "start": 4, "end": 4, "id": 4},
        {"text": "F\t\t \t", "start": 5, "end": 5, "id": 5},
        {"text": "G \t \t \t \t", "start": 6, "end": 6, "id": 6},
        {"text": "\n", "start": 7, "end": 7, "id": 7},
        {"text": "\n ", "start": 8, "end": 8, "id": 8},
        {"text": "\n\t \t \t \t", "start": 9, "end": 6, "id": 9},
    ]

    dataset = [doc]
    datasets = [dataset]

    expected = [
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        (None,),
        (None,),
        (None,),
    ]

    tlp = TokenLabelPairs(line_limit=73, respect_line_endings=True)
    actual = tlp.run(datasets)

    assert expected == actual

def test_TokenLabelPairs_retains_line_endings():
    """
    Rodrigues et al. retain the line endings as they appear in the text, meaning
    that on average a line is very short.
    """

    doc = dict()

    doc["_input_hash"] = 1337
    doc["tokens"] = [
        {"text": "\n", "start": 0, "end": 0, "id": 0},
        {"text": "\n", "start": 1, "end": 1, "id": 1},
        {"text": "\n", "start": 2, "end": 2, "id": 2},
        {"text": "\n", "start": 3, "end": 3, "id": 3},
    ]

    dataset = [doc]
    datasets = [dataset, dataset]

    expected = [
        (None,),
        (None,),
        (None,),
        (None,),
    ]

    tlp = TokenLabelPairs(respect_line_endings=True)
    actual = tlp.run(datasets)

    assert actual == expected


def test_TokenLabelPairs_ignores_line_endings():

    doc = dict()

    doc["_input_hash"] = 1337
    doc["tokens"] = [
        {"text": "a", "start": 0, "end": 0, "id": 0},
        {"text": "b", "start": 1, "end": 1, "id": 1},
        {"text": "c", "start": 2, "end": 2, "id": 2},
        {"text": "d", "start": 3, "end": 3, "id": 3},
    ]

    dataset = [doc]
    datasets = [dataset]

    expected = [
        ("a",),
        ("b",),
        (None,),
        ("c",),
        ("d",),
        (None,),
    ]

    tlp = TokenLabelPairs(line_limit=2, respect_line_endings=False)
    actual = tlp.run(datasets)

    assert actual == expected


def test_TokenLabelPairs_respects_ignores_doc_endings(dataset):

    datasets = [dataset, dataset, dataset]

    expected = [
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
        ("A",),
        ("B",),
        ("C",),
        ("D",),
        ("E",),
        ("F",),
        ("G",),
    ]

    tlp = TokenLabelPairs(
        line_limit=73, respect_line_endings=False, respect_doc_endings=False
    )
    actual = tlp.run(datasets)

    assert actual == expected


def test_reference_spans_real_example(doc):

    expected = [
        ("References", "o"),
        ("1", "o"),
        (".", "o"),
        ("United", "author"),
        ("Nations", "author"),
        ("Development", "author"),
        ("Programme", "author"),
        ("(", "author"),
        ("UNDP", "author"),
        (")", "author"),
        (".", "o"),
        ("A", "o"),
        ("Guide", "title"),
        ("to", "title"),
        ("Civil", "title"),
        ("Society", "title"),
        ("Organizations", "title"),
        ("working", "title"),
        ("on", "title"),
        ("Democratic", "title"),
        ("Governance", "title"),
        ("[", "title"),
        ("online", "title"),
        ("publication].", "title"),
        ("New", "o"),
        ("York", "o"),
        (",", "o"),
        ("NY", "o"),
        (";", "o"),
        ("UNDP", "o"),
        (";", "o"),
        ("2005", "year"),
        (".", "year"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        ("http://www.undp.org", "o"),
        ("/", "o"),
        ("content", "o"),
        ("/", "o"),
        ("dam", "o"),
        ("/", "o"),
        ("aplaws", "o"),
        ("/", "o"),
        ("publication", "o"),
        ("/", "o"),
        ("en", "o"),
        ("/", "o"),
        ("publications", "o"),
        ("/", "o"),
        ("democratic-", "o"),
        ("governance", "o"),
        ("/", "o"),
        ("oslo", "o"),
        ("-", "o"),
        ("governance", "o"),
        ("-", "o"),
        ("center", "o"),
        ("/", "o"),
        ("civic", "o"),
        ("-", "o"),
        ("engagement", "o"),
        ("/", "o"),
        ("a", "o"),
        ("-", "o"),
        ("guide", "o"),
        ("-", "o"),
        ("to", "o"),
        ("-", "o"),
        ("civil", "o"),
        ("-", "o"),
        ("society-", "o"),
        ("organizations", "o"),
        ("-", "o"),
        ("working", "o"),
        ("-", "o"),
        ("on", "o"),
        ("-", "o"),
        ("democratic", "o"),
        ("-", "o"),
        ("governance-/3665%20Booklet_heleWEB_.pdf", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("2", "o"),
        (".", "o"),
        ("Mental", "o"),
        ("Health", "author"),
        ("Peer", "author"),
        ("Connection", "author"),
        ("(", "author"),
        ("MHPC", "author"),
        (")", "author"),
        (".", "author"),
        ("Mental", "o"),
        ("Health", "title"),
        ("Peer", "title"),
        ("Connection", "title"),
        ("[", "title"),
        ("website].", "o"),
        ("Buffalo", "o"),
        (",", "o"),
        ("NY", "o"),
        (";", "o"),
        ("MHPC", "o"),
        (";", "o"),
        ("n.d", "o"),
        (".", "o"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        ("http://wnyil.org/mhpc.html", "o"),
        (",", "o"),
        ("a", "o"),
        ("ccessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("3", "o"),
        (".", "o"),
        ("Avery", "o"),
        ("S", "author"),
        (",", "author"),
        ("Mental", "author"),
        ("Health", "author"),
        ("Peer", "author"),
        ("Connection", "author"),
        ("(", "author"),
        ("MHPC", "author"),
        (")", "author"),
        (".", "author"),
        ("Channels", "o"),
        ("2013", "o"),
        (",", "o"),
        ("“", "o"),
        ("Not", "title"),
        ("Without", "title"),
        ("Us", "title"),
        ("”", "title"),
        ("[", "title"),
        ("video].", "o"),
        ("Western", "o"),
        ("New", "o"),
        ("York", "o"),
        ("(", "o"),
        ("WNY", "o"),
        (")", "o"),
        (";", "o"),
        ("Squeeky", "o"),
        ("Wheel", "o"),
        (";", "o"),
        ("2013", "o"),
        (".", "year"),
        ("(", "year"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        ("https://vimeo.com/62705552", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("4", "o"),
        (".", "o"),
        ("Alzheimer", "o"),
        ("'s", "o"),
        ("Disease", "author"),
        ("International", "author"),
        ("(", "author"),
        ("ADI", "author"),
        (")", "author"),
        (".", "author"),
        ("How", "author"),
        ("to", "o"),
        ("develop", "title"),
        ("an", "title"),
        ("Alzheimer", "title"),
        ("'s", "title"),
        ("association", "title"),
        ("and", "title"),
        ("get", "title"),
        ("results", "title"),
        ("[", "title"),
        ("website].", "title"),
        ("United", "title"),
        ("Kingdom", "title"),
        (";", "o"),
        ("ADI", "o"),
        (";", "o"),
        ("2006", "o"),
        (".", "o"),
        ("(", "year"),
        ("Available", "year"),
        ("from", "o"),
        (":", "o"),
        ("https:/", "o"),
        ("/", "o"),
        ("www.alz.co.uk", "o"),
        ("/", "o"),
        ("how-", "o"),
        ("to", "o"),
        ("-", "o"),
        ("develop", "o"),
        ("-", "o"),
        ("an", "o"),
        ("-", "o"),
        ("association", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("5", "o"),
        (".", "o"),
        ("Normal", "o"),
        ("Difference", "o"),
        ("Mental", "o"),
        ("Health", "author"),
        ("Kenya", "author"),
        ("(", "author"),
        ("NDMHK", "author"),
        (")", "author"),
        (".", "author"),
        ("About", "author"),
        ("Us", "author"),
        ("[", "o"),
        ("website].", "title"),
        ("Kenya", "title"),
        (";", "o"),
        ("NDMHK", "o"),
        (";", "o"),
        ("n.d", "o"),
        (".", "o"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        (None, None),
        ("http://www.normal-difference.org/?page_id=15", "o"),
        (",", "o"),
        ("ac", "o"),
        ("cessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("6", "o"),
        (".", "o"),
        ("TOPSIDE", "o"),
        (".", "o"),
        ("Training", "o"),
        ("Opportunities", "author"),
        ("for", "author"),
        ("Peer", "o"),
        ("Supporters", "title"),
        ("with", "title"),
        ("Intellectual", "title"),
        ("Disabilities", "title"),
        ("in", "title"),
        ("Europe", "title"),
        ("[", "title"),
        ("website", "title"),
        ("]", "title"),
        (";", "o"),
        ("TOPSIDE", "o"),
        (";", "o"),
        ("n.d", "o"),
        (".", "o"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        ("http://www.peer-support.eu/about-the-project/", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("7", "o"),
        (".", "o"),
        ("KOSHISH", "o"),
        ("National", "o"),
        ("Mental", "o"),
        ("Health", "o"),
        ("Self", "author"),
        ("-", "author"),
        ("help", "author"),
        ("Organisation", "author"),
        (".", "author"),
        ("Advocacy", "author"),
        ("and", "author"),
        ("Awareness", "author"),
        ("[", "o"),
        ("website].", "title"),
        ("Nepal", "title"),
        (";", "o"),
        ("KOSHISH", "o"),
        (";", "o"),
        ("2015", "o"),
        (".", "o"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "year"),
        (":", "year"),
        ("http://koshishnepal.org/advocacy", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("8", "o"),
        (".", "o"),
        ("Dementia", "o"),
        ("Alliance", "o"),
        ("International", "o"),
        ("(", "o"),
        ("DAI", "o"),
        (")", "author"),
        (".", "author"),
        ("Dementia", "author"),
        ("Alliance", "author"),
        ("International", "author"),
        ("[", "author"),
        ("website].", "o"),
        ("Ankeny", "title"),
        (",", "title"),
        ("IA", "title"),
        (";", "o"),
        ("DAI", "o"),
        (";", "o"),
        ("2014/2015", "o"),
        (".", "o"),
        ("(", "o"),
        ("Available", "o"),
        ("from", "o"),
        (":", "o"),
        ("http://www.dementiaallianceinternational.org/", "o"),
        (",", "o"),
        ("accessed", "o"),
        ("15", "o"),
        ("February", "o"),
        ("2017", "o"),
        (")", "o"),
        (".", "o"),
        ("9", "o"),
        (".", "o"),
        (None, None),
    ]

    dataset = [doc]
    datasets = [dataset]

    tlp = TokenLabelPairs(respect_line_endings=False)
    actual = tlp.run(datasets)

    assert actual == expected


def test_prodigy_to_tsv(tmpdir):

    output_file = os.path.join(tmpdir, "prodigy_to_tsv.tsv")

    prodigy_to_tsv(TEST_TOKEN_LABELLED, output_file, respect_lines=False, respect_docs=True)

    expected = load_tsv(TEST_TOKEN_LABELLED_TSV)
    actual = load_tsv(output_file)

    assert expected == actual
