#!/usr/bin/env python3
# coding: utf-8

import os


def get_path(p):
    return os.path.join(os.path.dirname(__file__), p)


TEST_TOKENS = get_path("test_data/test_tokens_to_tsv_tokens.jsonl")
TEST_SPANS = get_path("test_data/test_tokens_to_tsv_spans.jsonl")
TEST_REF_TOKENS = get_path("test_data/test_reference_to_token_tokens.jsonl")
TEST_REF_SPANS = get_path("test_data/test_reference_to_token_spans.jsonl")
TEST_REF_EXPECTED_SPANS = get_path("test_data/test_reference_to_token_expected.jsonl")

# Prodigy format document containing numbered reference section

TEST_NUMBERED_REFERENCES = get_path("test_data/test_numbered_references.jsonl")

# Prodigy format document with spans annotating every token in the document

TEST_TOKEN_LABELLED = get_path("test_data/test_token_labelled_references.jsonl")

# Reference section in Reach format

TEST_REACH = get_path("test_data/test_reach.jsonl")
