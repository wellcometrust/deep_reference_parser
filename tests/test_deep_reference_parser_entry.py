
#!/usr/bin/env python3
# coding: utf-8

import pytest

from deep_reference_parser.predict import Predictor

from .common import TEST_CFG, TEST_REFERENCES


@pytest.fixture
def predictor():
    return Predictor(TEST_CFG)

@pytest.fixture
def text():
    with open(TEST_REFERENCES, "r") as fb:
        text = fb.read()

    return text


@pytest.mark.slow
def test_predictor_list_output(text, predictor):
    """
    Test that the predict entrypoint works as expected.

    If the model artefacts and embeddings are not present this test will
    downloaded them, which can be slow.
    """
    out = predictor.split(text, return_tokens=False, verbose=False)

    assert isinstance(out, list)


# Allow to xfail as this depends on the model
@pytest.mark.xfail
def test_predictor_output_length(text, predictor):
    """
    For now use a minimal set of weights which may fail to predict anything
    useful. Hence this test is xfailed.
    """
    out = predictor.split(text, return_tokens=False, verbose=False)

    assert isinstance(out[0], str)
    assert len(out) == 3


def test_predictor_tokens_output(text, predictor):
    """
    """
    out = predictor.split(text, return_tokens=True, verbose=False)

    assert isinstance(out, list)
    assert isinstance(out[0], tuple)
    assert len(out[0]) == 2
    assert isinstance(out[0][0], str)
    assert isinstance(out[0][1], str)
