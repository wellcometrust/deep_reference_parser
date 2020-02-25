#!/usr/bin/env python3
# coding: utf-8

import pytest

from deep_reference_parser.model_utils import remove_padding_from_predictions


def test_remove_pre_padding():

    predictions = [
        ["pad", "pad", "pad", "pad", "token", "token", "token"],
        ["pad", "pad", "pad", "pad", "pad", "token", "token"],
        ["pad", "pad", "pad", "pad", "pad", "pad", "token"],
    ]

    X = [
        ["token", "token", "token"],
        ["token", "token"],
        ["token"],
    ]

    out = remove_padding_from_predictions(X, predictions, "pre")

    assert out == X


def test_remove_post_padding():

    predictions = [
        ["token", "token", "token", "pad", "pad", "pad", "pad"],
        ["token", "token", "pad", "pad", "pad", "pad", "pad"],
        ["token", "pad", "pad", "pad", "pad", "pad", "pad"],
    ]
    X = [
        ["token", "token", "token"],
        ["token", "token"],
        ["token"],
    ]

    out = remove_padding_from_predictions(X, predictions, "post")

    assert out == X
