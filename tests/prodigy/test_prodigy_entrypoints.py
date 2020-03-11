"""Simple tests that entrypoints run. Functionality is tested in other more 
specific tests
"""

import os

import pytest

from deep_reference_parser.prodigy import (
    annotate_numbered_references,
    prodigy_to_tsv,
    reach_to_prodigy,
    reference_to_token_annotations,
)

from .common import TEST_NUMBERED_REFERENCES, TEST_TOKEN_LABELLED, TEST_REACH


@pytest.fixture(scope="session")
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


def test_annotate_numbered_references_entrypoint(tmpdir):
    annotate_numbered_references(
        TEST_NUMBERED_REFERENCES, os.path.join(tmpdir, "references.jsonl")
    )


def test_prodigy_to_tsv(tmpdir):
    prodigy_to_tsv(
        TEST_TOKEN_LABELLED,
        os.path.join(tmpdir, "tokens.tsv"),
        respect_lines=False,
        respect_docs=True,
    )


def test_reach_to_prodigy(tmpdir):
    reach_to_prodigy(TEST_REACH, os.path.join(tmpdir, "prodigy.jsonl"))


def test_reference_to_token_annotations(tmpdir):
    reference_to_token_annotations(
        TEST_NUMBERED_REFERENCES, os.path.join(tmpdir, "tokens.jsonl")
    )
