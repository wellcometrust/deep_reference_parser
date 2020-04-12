#!/usr/bin/env python3
# coding: utf-8

import pytest

from deep_reference_parser.parse import Parser
from deep_reference_parser.split import Splitter
from deep_reference_parser.split_parse import SplitParser

from .common import TEST_CFG, TEST_REFERENCES


@pytest.fixture
def splitter():
    return Splitter(TEST_CFG)


@pytest.fixture
def parser():
    return Parser(TEST_CFG)


@pytest.fixture
def split_parser():
    return SplitParser(TEST_CFG)


@pytest.fixture
def text():
    with open(TEST_REFERENCES, "r") as fb:
        text = fb.read()

    return text


@pytest.mark.slow
def test_splitter_list_output(text, splitter):
    """
    Test that the splitter entrypoint works as expected.

    If the model artefacts and embeddings are not present this test will
    downloaded them, which can be slow.
    """
    out = splitter.split(text, return_tokens=False, verbose=False)

    assert isinstance(out, list)


@pytest.mark.slow
def test_parser_list_output(text, parser):
    """
    Test that the parser entrypoint works as expected.

    If the model artefacts and embeddings are not present this test will
    downloaded them, which can be slow.
    """
    out = parser.parse(text, verbose=False)

    assert isinstance(out, list)


@pytest.mark.slow
def test_split_parser_list_output(text, split_parser):
    """
    Test that the parser entrypoint works as expected.

    If the model artefacts and embeddings are not present this test will
    downloaded them, which can be slow.
    """
    out = split_parser.split_parse(text, verbose=False)
    print(out)

    assert isinstance(out, list)


def test_splitter_tokens_output(text, splitter):
    """
    """
    out = splitter.split(text, return_tokens=True, verbose=False)

    assert isinstance(out, list)
    assert isinstance(out[0], tuple)
    assert len(out[0]) == 2
    assert isinstance(out[0][0], str)
    assert isinstance(out[0][1], str)


def test_parser_tokens_output(text, parser):
    """
    """
    out = parser.parse(text, verbose=False)

    assert isinstance(out, list)
    assert isinstance(out[0], tuple)
    assert len(out[0]) == 2
    assert isinstance(out[0][0], str)
    assert isinstance(out[0][1], str)


def test_split_parser_tokens_output(text, split_parser):
    """
    """
    out = split_parser.split_parse(text, verbose=False)

    assert isinstance(out, list)

    # NOTE: full functionality of split_parse is not yet implemented.

    # assert isinstance(out[0], tuple)
    # assert len(out[0]) == 2
    # assert isinstance(out[0][0], str)
    # assert isinstance(out[0][1], str)
