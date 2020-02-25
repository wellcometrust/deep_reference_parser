#!/usr/bin/env python3
# coding: utf-8

import os


def get_path(p):
    return os.path.join(
        os.path.dirname(__file__),
        p
    )

TEST_TOKENS = get_path('test_data/test_tokens_to_tsv_tokens.jsonl')
TEST_SPANS = get_path('test_data/test_tokens_to_tsv_spans.jsonl')
TEST_REF_TOKENS = get_path('test_data/test_reference_to_token_tokens.jsonl')
TEST_REF_SPANS = get_path('test_data/test_reference_to_token_spans.jsonl')
TEST_REF_EXPECTED_SPANS = get_path('test_data/test_reference_to_token_expected.jsonl')
