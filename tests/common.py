#!/usr/bin/env python3
# coding: utf-8

import os


def get_path(p):
    return os.path.join(
        os.path.dirname(__file__),
        p
    )

TEST_CFG = get_path('test_data/test_config.ini')
TEST_JSONL = get_path('test_data/test_jsonl.jsonl')
TEST_REFERENCES = get_path('test_data/test_references.txt')
TEST_TSV_PREDICT = get_path('test_data/test_tsv_predict.tsv')
TEST_TSV_TRAIN = get_path('test_data/test_tsv_train.tsv')
