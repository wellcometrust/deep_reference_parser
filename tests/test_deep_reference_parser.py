#!/usr/bin/env python3
# coding: utf-8

import shutil
import tempfile

import pytest
from deep_reference_parser import DeepReferenceParser, get_config, load_tsv
from deep_reference_parser.__version__ import __model_version__
from deep_reference_parser.common import LATEST_CFG

from .common import TEST_CFG, TEST_TSV_PREDICT, TEST_TSV_TRAIN
from wasabi import msg


@pytest.fixture(scope="session")
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("data")


@pytest.fixture(scope="module")
def cfg():
    return get_config(TEST_CFG)


@pytest.mark.slow
def test_DeepReferenceParser_train(tmpdir, cfg):
    """
    This test creates the artefacts that will be used in the next test
    """

    X_test, y_test = load_tsv(TEST_TSV_TRAIN)

    X_test = X_test[0:100]
    y_test = y_test[0:100]

    drp = DeepReferenceParser(
        X_train=X_test,
        X_test=X_test,
        X_valid=X_test,
        y_train=y_test,
        y_test=y_test,
        y_valid=y_test,
        output_path=tmpdir,
    )

    # Prepare the data

    drp.prepare_data(save=True)

    # Build the model architecture

    drp.build_model(
        output=cfg["build"]["output"],
        word_embeddings=cfg["build"]["word_embeddings"],
        pretrained_embedding=cfg["build"]["pretrained_embedding"],
        dropout=float(cfg["build"]["dropout"]),
        lstm_hidden=int(cfg["build"]["lstm_hidden"]),
        word_embedding_size=int(cfg["build"]["word_embedding_size"]),
        char_embedding_size=int(cfg["build"]["char_embedding_size"]),
    )

    # Train the model (quickly)

    drp.train_model(
        epochs=int(cfg["train"]["epochs"]), batch_size=int(cfg["train"]["batch_size"])
    )

    # Evaluate the model. This will write some evalutaion data to the
    # tempoary directory.

    drp.evaluate(load_weights=False, test_set=True, validation_set=True)

    examples = [
        "This is an example".split(" "),
        "This is also an example".split(" "),
        "And so is this".split(" "),
    ]


@pytest.mark.slow
def test_DeepReferenceParser_predict(tmpdir, cfg):
    """
    You must run this test after the previous one, or it will fail
    """

    drp = DeepReferenceParser(
        # Nothign will be written here
        # output_path=cfg["build"]["output_path"]
        output_path=tmpdir
    )

    # Load mapping dicts from the baseline model

    drp.load_data(tmpdir)

    # Build the model architecture

    drp.build_model(
        output=cfg["build"]["output"],
        word_embeddings=cfg["build"]["word_embeddings"],
        pretrained_embedding=False,
        dropout=float(cfg["build"]["dropout"]),
        lstm_hidden=int(cfg["build"]["lstm_hidden"]),
        word_embedding_size=int(cfg["build"]["word_embedding_size"]),
        char_embedding_size=int(cfg["build"]["char_embedding_size"]),
    )

    examples = [
        "This is an example".split(" "),
        "This is also an example".split(" "),
        "And so is this".split(" "),
    ]

    preds = drp.predict(examples, load_weights=True)

    assert len(preds) == len(examples)

    assert len(preds[0]) == len(examples[0])
    assert len(preds[1]) == len(examples[1])
    assert len(preds[2]) == len(examples[2])

    assert isinstance(preds[0][0], str)
    assert isinstance(preds[1][0], str)
    assert isinstance(preds[2][0], str)
