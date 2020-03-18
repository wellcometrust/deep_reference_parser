#!/usr/bin/env python3
# coding: utf-8

import pytest

from deep_reference_parser.reference_utils import break_into_chunks


def test_break_into_chunks():

    before = ["a", "b", "c", "d", "e"]
    expected = [["a", "b"], ["c", "d"], ["e"]]

    actual = break_into_chunks(before, max_words=2)

    assert expected == actual
