#!/usr/bin/env python3
# coding: utf-8
"""
Run predictions from a pre-trained model
"""

import itertools
import os

import en_core_web_sm
import plac
import spacy
import wasabi

from deep_reference_parser import __file__
from deep_reference_parser.common import LATEST_CFG, download_model_artefact
from deep_reference_parser.deep_reference_parser import DeepReferenceParser
from deep_reference_parser.logger import logger
from deep_reference_parser.model_utils import get_config
from deep_reference_parser.reference_utils import break_into_chunks
from deep_reference_parser.tokens_to_references import tokens_to_references

msg = wasabi.Printer(icons={"check":"\u2023"})

class Predictor:

    def __init__(self, config_file):

        msg.info(f"Using config file: {config_file}")

        cfg = get_config(config_file)

        # Build config

        OUTPUT_PATH = cfg["build"]["output_path"]
        S3_SLUG = cfg["data"]["s3_slug"]

        # Check whether the necessary artefacts exists and download them if
        # not.

        artefacts = [
            "char2ind.pickle", "ind2label.pickle", "ind2word.pickle",
            "label2ind.pickle", "maxes.pickle", "weights.h5",
            "word2ind.pickle"
        ]

        for artefact in artefacts:
            with msg.loading(f"Could not find {artefact} locally, downloading..."):
                try:
                    artefact = os.path.join(OUTPUT_PATH, artefact)
                    download_model_artefact(artefact, S3_SLUG)
                    msg.good(f"Found {artefact}")
                except:
                    msg.fail(f"Could not download {S3_SLUG}{artefact}")
                    logger.exception()

        # Check on word embedding and download if not exists

        WORD_EMBEDDINGS = cfg["build"]["word_embeddings"]

        with msg.loading(f"Could not find {WORD_EMBEDDINGS} locally, downloading..."):
            try:
                download_model_artefact(WORD_EMBEDDINGS, S3_SLUG)
                msg.good(f"Found {WORD_EMBEDDINGS}")
            except:
                msg.fail(f"Could not download {S3_SLUG}{WORD_EMBEDDINGS}")
                logger.exception()


        OUTPUT = cfg["build"]["output"]
        PRETRAINED_EMBEDDING = cfg["build"]["pretrained_embedding"]
        DROPOUT = float(cfg["build"]["dropout"])
        LSTM_HIDDEN = int(cfg["build"]["lstm_hidden"])
        WORD_EMBEDDING_SIZE = int(cfg["build"]["word_embedding_size"])
        CHAR_EMBEDDING_SIZE = int(cfg["build"]["char_embedding_size"])

        self.MAX_WORDS = int(cfg["data"]["line_limit"])

        # Evaluate config

        self.drp = DeepReferenceParser(output_path=OUTPUT_PATH)

        # Encode data and load required mapping dicts. Note that the max word and
        # max char lengths will be loaded in this step.

        self.drp.load_data(OUTPUT_PATH)

        # Build the model architecture

        self.drp.build_model(
            output=OUTPUT,
            word_embeddings=WORD_EMBEDDINGS,
            pretrained_embedding=PRETRAINED_EMBEDDING,
            dropout=DROPOUT,
            lstm_hidden=LSTM_HIDDEN,
            word_embedding_size=WORD_EMBEDDING_SIZE,
            char_embedding_size=CHAR_EMBEDDING_SIZE,
        )

    def split(self, text, return_tokens=False, verbose=False):

        nlp = en_core_web_sm.load()
        doc = nlp(text)
        chunks = break_into_chunks(doc, max_words=self.MAX_WORDS)
        tokens = [[token.text for token in chunk] for chunk in chunks]

        preds = self.drp.predict(tokens, load_weights=True)

        # If tokens argument passed, return the labelled tokens

        if return_tokens:

            flat_predictions = list(itertools.chain.from_iterable(preds))
            flat_X = list(itertools.chain.from_iterable(tokens))
            rows = [i for i in zip(flat_X, flat_predictions)]

            if verbose:

                msg.divider("Token Results")

                header = ("token", "label")
                aligns = ("r", "l")
                formatted = wasabi.table(rows, header=header, divider=True,
                    aligns=aligns)
                print(formatted)

            out = rows

        else:

            # Otherwise convert the tokens into references and return

            refs = tokens_to_references(tokens, preds)

            if verbose:

                msg.divider("Results")

                if refs:

                    msg.good(f"Found {len(refs)} references.")
                    msg.info("Printing found references:")

                    for ref in refs:
                        msg.text(ref, icon="check", spaced=True)

                else:

                    msg.fail("Failed to find any references.")

            out = refs

        return out

@plac.annotations(
    text=("Plaintext from which to extract references", "positional", None, str),
    config_file=("Path to config file", "option", "c", str),
    tokens=("Output tokens instead of complete references", "flag", "t", str),
    verbose=("Output more verbose results", "flag", "v", str),
)
def predict(text, config_file=LATEST_CFG, tokens=False, verbose=False):
    predictor = Predictor(config_file)
    out = predictor.split(text, return_tokens=tokens, verbose=verbose)

    if not verbose:
        print(out)
