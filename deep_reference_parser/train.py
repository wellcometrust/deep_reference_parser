#!/usr/bin/env python3
# coding: utf-8
"""
Runs the model using configuration defined in a config file. This is suitable for
running model versions < 2019.10.8
"""

import os
import sys
import plac
import wasabi

from deep_reference_parser import (
    DeepReferenceParser,
    get_config,
    load_data,
    load_tsv,
    logger,
)

msg = wasabi.Printer()


@plac.annotations(
    config_file=("Path to config file", "positional", None, str),
)

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
    OUTPUT = cfg["build"]["output"]
    WORD_EMBEDDINGS = cfg["build"]["word_embeddings"]
    PRETRAINED_EMBEDDING = cfg["build"]["pretrained_embedding"]
    DROPOUT = float(cfg["build"]["dropout"])
    LSTM_HIDDEN = int(cfg["build"]["lstm_hidden"])
    WORD_EMBEDDING_SIZE = int(cfg["build"]["word_embedding_size"])
    CHAR_EMBEDDING_SIZE = int(cfg["build"]["char_embedding_size"])

    # Train config

    EPOCHS = int(cfg["train"]["epochs"])
    BATCH_SIZE = int(cfg["train"]["batch_size"])
    EARLY_STOPPING_PATIENCE = int(cfg["train"]["early_stopping_patience"])
    METRIC = cfg["train"]["metric"]

    # Evaluate config

    OUT_FILE = cfg["evaluate"]["out_file"]

    # Load policy data

    X_train, y_train = load_tsv(POLICY_TRAIN)
    X_test, y_test = load_tsv(POLICY_TEST)
    X_valid, y_valid = load_tsv(POLICY_VALID)

    logger.info("X_train, y_train examples: %s, %s", len(X_train), len(y_train))
    logger.info("X_test, y_test  examples: %s, %s", len(X_test), len(y_test))
    logger.info("X_valid, y_valid  examples: %s, %s", len(X_valid), len(y_valid))

    drp = DeepReferenceParser(
        X_train=X_train,
        X_test=X_test,
        X_valid=X_valid,
        y_train=y_train,
        y_test=y_test,
        y_valid=y_valid,
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
        out_file=cfg["evaluate"]["out_file"],
    )


