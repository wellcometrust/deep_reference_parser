#!/usr/bin/env python3
# coding: utf-8
"""
Run predictions from a pre-trained model
"""

import itertools
import json
import os

import en_core_web_sm
import plac
import spacy
import wasabi

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from deep_reference_parser import __file__
    from deep_reference_parser.__version__ import __splitter_model_version__
    from deep_reference_parser.common import MULTITASK_CFG, download_model_artefact
    from deep_reference_parser.deep_reference_parser import DeepReferenceParser
    from deep_reference_parser.logger import logger
    from deep_reference_parser.model_utils import get_config
    from deep_reference_parser.reference_utils import break_into_chunks
    from deep_reference_parser.tokens_to_references import tokens_to_reference_lists

msg = wasabi.Printer(icons={"check": "\u2023"})


class SplitParser:
    def __init__(self, config_file):

        msg.info(f"Using config file: {config_file}")

        cfg = get_config(config_file)

        try:
            OUTPUT_PATH = cfg["build"]["output_path"]
            S3_SLUG = cfg["data"]["s3_slug"]
        except KeyError:
            config_dir, missing_config = os.path.split(config_file)
            files = os.listdir(config_dir)
            other_configs = [f for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f))]
            msg.fail(f"Could not find config {missing_config}, perhaps you meant one of {other_configs}")

        # Check whether the necessary artefacts exists and download them if
        # not.

        artefacts = [
            "indices.pickle",
            "weights.h5",
        ]

        for artefact in artefacts:
            with msg.loading(f"Could not find {artefact} locally, downloading..."):
                try:
                    artefact = os.path.join(OUTPUT_PATH, artefact)
                    download_model_artefact(artefact, S3_SLUG)
                    msg.good(f"Found {artefact}")
                except:
                    msg.fail(f"Could not download {S3_SLUG}{artefact}")
                    logger.exception("Could not download %s%s", S3_SLUG, artefact)

        # Check on word embedding and download if not exists

        WORD_EMBEDDINGS = cfg["build"]["word_embeddings"]

        with msg.loading(f"Could not find {WORD_EMBEDDINGS} locally, downloading..."):
            try:
                download_model_artefact(WORD_EMBEDDINGS, S3_SLUG)
                msg.good(f"Found {WORD_EMBEDDINGS}")
            except:
                msg.fail(f"Could not download {S3_SLUG}{WORD_EMBEDDINGS}")
                logger.exception("Could not download %s", WORD_EMBEDDINGS)

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

    def split_parse(self, text, return_tokens=False, verbose=False):

        nlp = en_core_web_sm.load()
        doc = nlp(text)
        chunks = break_into_chunks(doc, max_words=self.MAX_WORDS)
        tokens = [[token.text for token in chunk] for chunk in chunks]

        preds = self.drp.predict(tokens, load_weights=True)

        # If tokens argument passed, return the labelled tokens

        if return_tokens:

            flat_preds_list = list(map(itertools.chain.from_iterable,preds))
            flat_X = list(itertools.chain.from_iterable(tokens))
            rows = [i for i in zip(*[flat_X] + flat_preds_list)]

            if verbose:

                msg.divider("Token Results")

                header = tuple(["token"] + ["label"] * len(flat_preds_list))
                aligns = tuple(["r"] +  ["l"] * len(flat_preds_list))
                formatted = wasabi.table(
                    rows, header=header, divider=True, aligns=aligns
                )
                print(formatted)

            out = rows

        else:

            # Return references with attributes (author, title, year)
            # in json format.
            # List of lists for each reference - each reference list contains all token attributes predictions
            # [[(token, attribute), ... , (token, attribute)], ..., [(token, attribute), ...]]

            references_components = tokens_to_reference_lists(tokens, spans=preds[1], components=preds[0])
            if verbose:

                msg.divider("Results")

                if references_components:

                    msg.good(f"Found {len(references_components)} references.")
                    msg.info("Printing found references:")

                    for ref in references_components:
                        msg.text(ref['Reference'], icon="check", spaced=True)

                else:

                    msg.fail("Failed to find any references.")

            out = references_components

        return out


@plac.annotations(
    text=("Plaintext from which to extract references", "positional", None, str),
    config_file=("Path to config file", "option", "c", str),
    tokens=("Output tokens instead of complete references", "flag", "t", str),
    outfile=("Path to json file to which results will be written", "option", "o", str),
)
def split_parse(text, config_file=MULTITASK_CFG, tokens=False, outfile=None):
    """
    Runs the default splitting model and pretty prints results to console unless
    --outfile is parsed with a path. Files output to the path specified in
    --outfile will be a valid json. Can output either tokens (with -t|--tokens)
    or split naively into references based on the b-r tag (default).

    NOTE: that this function is provided for examples only and should not be used
    in production as the model is instantiated each time the command is run. To
    use in a production setting, a more sensible approach would be to replicate
    the split or parse functions within your own logic.
    """
    mt = SplitParser(config_file)
    if outfile:
        out = mt.split_parse(text, return_tokens=tokens, verbose=True)

        try:
            with open(outfile, "w") as fb:
                json.dump(out, fb)
            msg.good(f"Wrote model output to {outfile}")
        except:
            msg.fail(f"Failed to write output to {outfile}")

    else:
        out = mt.split_parse(text, return_tokens=tokens, verbose=True)
