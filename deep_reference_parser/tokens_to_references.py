#!/usr/bin/env python3
# coding: utf-8

"""
Converts a list of tokens and labels into a list of human readable references
"""

import itertools

from .deep_reference_parser import logger


def tokens_to_references(tokens, labels):
    """
    Given a corresponding list of tokens and a list of labels: split the tokens
    and return a list of references.

    Args:
        tokens(list): A list of tokens.
        labels(list): A corresponding list of labels.

    """

    # Flatten the lists of tokens and predictions into a single list.

    flat_tokens = list(itertools.chain.from_iterable(tokens))
    flat_predictions = list(itertools.chain.from_iterable(labels))

    # Find all b-r and e-r tokens.

    ref_starts = [
        index for index, label in enumerate(flat_predictions) if label == "b-r"
    ]

    ref_ends = [index for index, label in enumerate(flat_predictions) if label == "e-r"]

    logger.debug("Found %s b-r tokens", len(ref_starts))
    logger.debug("Found %s e-r tokens", len(ref_ends))

    references = []

    n_refs = len(ref_starts)

    # Split on each b-r.
    # TODO: It may be worth including some simple post processing steps here
    # to pick up false positives, for instance cutting short a reference
    # after n tokens.

    for i in range(0, n_refs):
        token_start = ref_starts[i]
        if i + 1 < n_refs:

            token_end = ref_starts[i + 1] - 1
        else:
            token_end = len(flat_tokens)

        ref = flat_tokens[token_start : token_end + 1]
        flat_ref = " ".join(ref)
        references.append(flat_ref)

    return references


def tokens_to_reference_lists(tokens, spans, components):
    """
    Given a corresponding list of tokens, a list of
    reference spans (e.g. 'b-r') and components (e.g. 'author;):
    split the tokens according to the spans and return a
    list of reference components for each reference.

    Args:
        tokens(list): A list of tokens.
        spans(list): A corresponding list of reference spans.
        components(list): A corresponding list of reference components.

    """

    # Flatten the lists of tokens and predictions into a single list.

    flat_tokens = list(itertools.chain.from_iterable(tokens))
    flat_spans = list(itertools.chain.from_iterable(spans))
    flat_components = list(itertools.chain.from_iterable(components))

    # Find all b-r and e-r tokens.

    ref_starts = [
        index for index, label in enumerate(flat_spans) if label == "b-r"
    ]

    ref_ends = [index for index, label in enumerate(flat_spans) if label == "e-r"]

    logger.debug("Found %s b-r tokens", len(ref_starts))
    logger.debug("Found %s e-r tokens", len(ref_ends))

    references_components = []

    n_refs = len(ref_starts)

    # Split on each b-r.

    for i in range(0, n_refs):
        token_start = ref_starts[i]
        if i + 1 < n_refs:

            token_end = ref_starts[i + 1] - 1
        else:
            token_end = len(flat_tokens)
        
        ref_tokens = flat_tokens[token_start : token_end + 1]
        ref_components = flat_components[token_start : token_end + 1]
        flat_ref = " ".join(ref_tokens)

        references_components.append({'Reference': flat_ref, 'Attributes': list(zip(ref_tokens, ref_components))})

    return references_components
