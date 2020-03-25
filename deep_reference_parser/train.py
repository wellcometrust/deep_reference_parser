#!/usr/bin/env python3
# coding: utf-8
"""
Runs the model using configuration defined in a config file. This is suitable for
running model versions < 2019.10.8
"""

import plac
import wasabi

from deep_reference_parser import load_tsv
from deep_reference_parser.common import download_model_artefact
from deep_reference_parser.deep_reference_parser import DeepReferenceParser
from deep_reference_parser.logger import logger
from deep_reference_parser.model_utils import get_config

msg = wasabi.Printer()


@plac.annotations(config_file=("Path to config file", "positional", None, str),)
def train(config_file):

    # Load variables from config files. Config files are used instead of ENV
    # vars due to the relatively large number of hyper parameters, and the need
    # to load these configs in both the train and predict moduldes.

    cfg = get_config(config_file)

    # Data config

    POLICY_TRAIN = cfg["data"]["policy_train"]
    POLICY_TEST = cfg["data"]["policy_test"]
    POLICY_VALID = cfg["data"]["policy_valid"]

    # Build config

    OUTPUT_PATH = cfg["build"]["output_path"]
    S3_SLUG = cfg["data"]["s3_slug"]

    # Check on word embedding and download if not exists

    WORD_EMBEDDINGS = cfg["build"]["word_embeddings"]

    with msg.loading(f"Could not find {WORD_EMBEDDINGS} locally, downloading..."):
        try:
            download_model_artefact(WORD_EMBEDDINGS, S3_SLUG)
            msg.good(f"Found {WORD_EMBEDDINGS}")
        except:
            msg.fail(f"Could not download {WORD_EMBEDDINGS}")
            logger.exception()

    OUTPUT = cfg["build"]["output"]
    WORD_EMBEDDINGS = cfg["build"]["word_embeddings"]
    PRETRAINED_EMBEDDING = cfg["build"]["pretrained_embedding"]
    DROPOUT = float(cfg["build"]["dropout"])
    LSTM_HIDDEN = int(cfg["build"]["lstm_hidden"])
    WORD_EMBEDDING_SIZE = int(cfg["build"]["word_embedding_size"])
    CHAR_EMBEDDING_SIZE = int(cfg["build"]["char_embedding_size"])
    MAX_LEN = int(cfg["data"]["line_limit"])

    # Train config

    EPOCHS = int(cfg["train"]["epochs"])
    BATCH_SIZE = int(cfg["train"]["batch_size"])
    EARLY_STOPPING_PATIENCE = int(cfg["train"]["early_stopping_patience"])
    METRIC = cfg["train"]["metric"]

    # Load policy data

    train_data = load_tsv(POLICY_TRAIN)
    test_data = load_tsv(POLICY_TEST)
    valid_data = load_tsv(POLICY_VALID)

    X_train, y_train = train_data[0], train_data[1:]
    X_test, y_test = test_data[0], test_data[1:]
    X_valid, y_valid = valid_data[0], valid_data[1:]

    import statistics

    logger.info("Max token length %s", max([len(i) for i in X_train]))
    logger.info("Min token length %s", min([len(i) for i in X_train]))
    logger.info("Mean token length %s", statistics.median([len(i) for i in X_train]))

    logger.info("Max token length %s", max([len(i) for i in X_test]))
    logger.info("Min token length %s", min([len(i) for i in X_test]))
    logger.info("Mean token length %s", statistics.median([len(i) for i in X_test]))

    logger.info("Max token length %s", max([len(i) for i in X_valid]))
    logger.info("Min token length %s", min([len(i) for i in X_valid]))
    logger.info("Mean token length %s", statistics.median([len(i) for i in X_valid]))

    logger.info("X_train, y_train examples: %s, %s", len(X_train), list(map(len, y_train)))
    logger.info("X_test, y_test examples: %s, %s", len(X_test), list(map(len, y_test)))
    logger.info("X_valid, y_valid examples: %s, %s", len(X_valid), list(map(len, y_valid)))

    drp = DeepReferenceParser(
        X_train=X_train,
        X_test=X_test,
        X_valid=X_valid,
        y_train=y_train,
        y_test=y_test,
        y_valid=y_valid,
        max_len=MAX_LEN,
        output_path=OUTPUT_PATH,
    )

    ## Encode data and create required mapping dicts

    drp.prepare_data(save=True)

    ## Build the model architecture

    drp.build_model(
        output=OUTPUT,
        word_embeddings=WORD_EMBEDDINGS,
        pretrained_embedding=PRETRAINED_EMBEDDING,
        dropout=DROPOUT,
        lstm_hidden=LSTM_HIDDEN,
        word_embedding_size=WORD_EMBEDDING_SIZE,
        char_embedding_size=CHAR_EMBEDDING_SIZE,
    )

    ## Train the model. Not required if downloading weights from s3

    drp.train_model(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        metric=METRIC,
    )

    # Evaluate the model. Confusion matrices etc will be stored in
    # data/model_output

    drp.evaluate(
        load_weights=True,
        test_set=True,
        validation_set=True,
        print_padding=False,
    )
