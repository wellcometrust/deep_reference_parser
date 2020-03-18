#!/usr/bin/env python3
# coding: utf-8

"""
"""

from .logger import logger

def yield_token_label_pairs(tokens, labels):
    """
    Convert matching lists of tokens and labels to tuples of (token, label) but
    preserving the nexted list boundaries as (None, None).

    Args:
        tokens(list): list of tokens.
        labels(list): list of labels corresponding to tokens.
    """

    for tokens, labels in zip(tokens, labels):
        if tokens and labels:
            for token, label in zip(tokens, labels):
                yield (token, label)
            yield (None, None)
        else:
            yield (None, None)

def break_into_chunks(doc, max_words=250):
    """
    Breaks a list into lists of lists of length max_words
    Also works on lists:

    >>> doc = ["a", "b", "c", "d", "e"]
    >>> break_into_chunks(doc, max_words=2)
        [['a', 'b'], ['c', 'd'], ['e']]
    """
    out = []
    chunk = []
    for i, token in enumerate(doc, 1):
        chunk.append(token)
        if (i > 0 and i % max_words == 0) or i == len(doc):
            out.append(chunk)
            chunk = []
    return out
