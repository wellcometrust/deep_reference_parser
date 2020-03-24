#!/usr/bin/env python3
# coding: utf-8

"""
Class used in scripts/prodigy_to_tsv.py which converts token annotated jsonl
files to tab-separated-values files for use in the deep reference parser
"""

import csv
import re
import sys
from functools import reduce

import numpy as np
import plac
from wasabi import Printer, table

from ..io import read_jsonl, write_tsv
from ..logger import logger

msg = Printer()

ROWS_TO_PRINT = 15


class TokenLabelPairs:
    """
    Convert prodigy format docs or list of lists into tuples of (token, label).
    """

    def __init__(
        self, line_limit=250, respect_line_endings=False, respect_doc_endings=True
    ):
        """
        Args:
            line_limit(int): Maximum number of tokens allowed per training
                example. If you are planning to use this data for making
                predictions, then this should correspond to the max_words
                attribute for the DeepReferenceParser class used to train the
                model.
            respect_line_endings(bool): If true, line endings appearing in the
                text will be respected, leading to much shorter line lengths
                usually <10. Typically this results in a much worser performing
                model, but follows the convention set by Rodrigues et al.
            respect_doc_endings(bool): If true, a line ending is added at the
                end of each document. If false, then the end of a document flows
                into the beginning of the next document.
        """

        self.line_count = 0
        self.line_lengths = []
        self.line_limit = line_limit
        self.respect_doc_endings = respect_doc_endings
        self.respect_line_endings = respect_line_endings

    def run(self, datasets):
        """

        Args:
            datasets (list): An arbitrary number of lists containing an
                arbitrary number of prodigy docs (dicts) which will be combined
                in to a single list of tokens based on the arguments provided
                when instantiating the TokenLabelPairs class, e.g.:

                [
                    (token0, label0_0, label0_1, label0_2),
                    (token1, label1_0, label1_1, label1_2),
                    (token2, label2_0, label2_1, label2_2),
                    (None,) # blank line
                    (token3, label3_0, label3_1, label3_2),
                    (token4, label4_0, label4_1, label4_2),
                    (token5, label5_0, label5_1, label5_2),

                ]

        """

        out = []

        input_hashes = list(map(get_input_hashes, datasets))

        # Check that datasets are compatible by comparing the _input_hash of
        # each document across the list of datasets.

        if not check_all_equal(input_hashes):
            msg.fail("Some documents missing from one of the input datasets")

            # If this is the case, also output some useful information for
            # determining which dataset is at fault.

            for i in range(len(input_hashes)):
                for j in range(i + 1, len(input_hashes)):
                    diff = set(input_hashes[i]) ^ set(input_hashes[j])

                    if diff:
                        msg.fail(
                            f"Docs {diff} unequal between dataset {i} and {j}", exits=1
                        )

        # Now that we know the input_hashes are equal, cycle through the first
        # one, and compare the tokens across the documents in each dataset from
        # datasets.

        for input_hash in input_hashes[0]:

            # Create list of docs whose _input_hash matches _input_hash. 
            # len(matched_docs) == len(datasets)

            matched_docs = list(map(lambda x: get_doc_by_input_hash(x, input_hash), datasets))

            # Create a list of tokens from input_hash_matches

            tokens = list(map(get_sorted_tokens, matched_docs))

            # All the tokens should match because they have the same _input_hash
            # but lets check just be sure...

            if check_all_equal(tokens):
                tokens_and_labels = [tokens[0]]
            else:
                msg.fail(f"Token mismatch for document {input_hash}", exits=1)

            # Create a list of spans from input_hash_matches

            spans = list(map(get_sorted_labels, matched_docs))

            # Create a list of lists like:
            # [[token0, token1, token2],[label0, label1, label2],...]. Sometimes
            # this will just be [None] if there were no spans in the documents,
            # so check for this.

            def all_nones(spans):
                return all(i is None for i in spans)

            if not all_nones(spans):
                tokens_and_labels.extend(spans)

            # Flatten the list of lists to give:
            # [(token0, label0, ...), (token1, label1, ...), (token2, label2, ...)]

            flattened_tokens_and_labels = list(zip(*tokens_and_labels))

            out.extend(list(self.yield_token_label_pair(flattened_tokens_and_labels)))

        # Print some statistics about the data.

        self.stats()

        return out

    def stats(self):

        avg_line_len = np.round(np.mean(self.line_lengths), 2)

        msg.info(f"Returning {self.line_count} examples")
        msg.info(f"Average line length: {avg_line_len}")

    def yield_token_label_pair(self, flattened_tokens_and_labels):
        """
        Args:
            flattened_tokens_and_labels (list): List of tuples relating to the
                tokens and labels of a given document.

        NOTE: Makes the assumption that every token has been labelled in spans. This
        assumption will be true if the data has been labelled with prodigy, then
        spans covering entire references have been converted to token spans. OR that
        there are no spans at all, and this is being used to prepare data for
        prediction.
        """

        # Set a token counter that is used to limit the number of tokens to
        # line_limit.

        token_counter = int(0)

        doc_len = len(flattened_tokens_and_labels)

        for i, token_and_labels in enumerate(flattened_tokens_and_labels, 1):

            token = token_and_labels[0]
            labels = token_and_labels[1:]
            blank = tuple([None] * (len(labels) + 1))

            # If the token is just spaces even if it has been labelled, pass it.

            if re.search(r"^[ ]+$", token):

                pass

            # If the token is a newline and we want to respect line endings in
            # the text, then yield None which will be converted to a blank line
            # when the resulting tsv file is read.

            elif re.search(r"\n", token) and self.respect_line_endings and i != doc_len:

                # Is it blank after whitespace is removed?

                if token.strip() == "":
                    yield blank

                    self.line_lengths.append(token_counter)
                    self.line_count += 1
                    token_counter = 0

                # Was it a \n combined with another token? if so return the
                # stripped token.

                else:
                    yield (token.strip(), *labels)
                    self.line_lengths.append(token_counter)
                    self.line_count += 1
                    token_counter = 1


            # Skip new lines if respect_line_endings not set and not the end
            # of a doc.

            elif re.search(r"\n", token) and i != doc_len:

                pass

            elif token_counter == self.line_limit:

                # Yield blank to signify a line ending, then yield the next
                # token.

                yield blank
                yield (token.strip(), *labels)

                # Set to one to account for the first token being added.

                self.line_lengths.append(token_counter)
                self.line_count += 1

                token_counter = 1

            elif i == doc_len and self.respect_doc_endings:

                # Case when the end of the document has been reached, but it is
                # less than self.lime_limit. This assumes that we want to retain
                # a line ending which denotes the end of a document, and the
                # start of new one.

                if token.strip():
                    yield (token.strip(), *labels)
                yield blank

                self.line_lengths.append(token_counter)
                self.line_count += 1

            else:
                # Returned the stripped label.

                yield (token.strip(), *labels)

                token_counter += 1


def get_input_hashes(dataset):
    """Get the hashes for every doc in a dataset and return as set
    """
    return set([doc["_input_hash"] for doc in dataset])


def check_all_equal(lst):
    """Check that all items in a list are equal and return True or False
    """
    return not lst or lst.count(lst[0]) == len(lst)


def hash_matches(doc, hash):
    """Check whether the hash of the passed doc matches the passed hash
    """
    return doc["_input_hash"] == hash


def get_doc_by_input_hash(dataset, hash):
    """Return a doc from a dataset where hash matches doc["_input_hash"]
    Assumes there will only be one match!
    """
    return [doc for doc in dataset if doc["_input_hash"] == hash][0]


def get_sorted_tokens(doc):
    tokens = sorted(doc["tokens"], key=lambda k: k["id"])
    return [token["text"] for token in doc["tokens"]]

def get_sorted_labels(doc):
    if doc.get("spans"):
        spans = sorted(doc["spans"], key=lambda k: k["token_start"])
        return [span["label"] for span in doc["spans"]]

def sort_docs_list(lst):
    """Sort a list of prodigy docs by input hash
    """
    return sorted(lst, key=lambda k: k["_input_hash"])


def combine_token_label_pairs(pairs):
    """Combines a list of [(token, label), (token, label)] to give
    (token,label,label).
    """
    return pairs[0][0:] + tuple(pair[1] for pair in pairs[1:])


@plac.annotations(
    input_files=(
        "Comma separated list of paths to jsonl files containing prodigy docs.",
        "positional",
        None,
        str,
    ),
    output_file=("Path to output tsv file.", "positional", None, str),
    respect_lines=(
        "Respect line endings? Or parse entire document in a single string?",
        "flag",
        "r",
        bool,
    ),
    respect_docs=(
        "Respect doc endings or parse corpus in single string?",
        "flag",
        "d",
        bool,
    ),
    line_limit=("Number of characters to include on a line", "option", "l", int),
)
def prodigy_to_tsv(
    input_files, output_file, respect_lines, respect_docs, line_limit=250
):
    """
    Convert token annotated jsonl to token annotated tsv ready for use in the
    deep_reference_parser model.

    Will combine annotations from two jsonl files containing the same docs and
    the same tokens by comparing the "_input_hash" and token texts. If they are
    compatible, the output file will contain both labels ready for use in a
    multi-task model, for example:

           token   label   label
    ------------   -----   -----
      References   o       o
               1   o       o
               .   o       o
             WHO   title   b-r
       treatment   title   i-r
      guidelines   title   i-r
             for   title   i-r
            drug   title   i-r
               -   title   i-r
       resistant   title   i-r
    tuberculosis   title   i-r
               ,   title   i-r
            2016   title   i-r

    Multiple files must be passed as a comma separated list e.g.

    python -m deep_reference_parser.prodigy prodigy_to_tsv file1.jsonl,file2.jsonl out.tsv

    """

    input_files = input_files.split(",")

    msg.info(f"Loading annotations from {len(input_files)} datasets")
    msg.info(f"Respect line endings: {respect_lines}")
    msg.info(f"Respect doc endings: {respect_docs}")
    msg.info(f"Target example length (n tokens): {line_limit}")

    # Read the input_files. Note the use of map here, because we don't know
    # how many sets of annotations area being passed in the list. It could be 2
    # but in future it may be more.

    annotated_data = list(map(read_jsonl, input_files))

    # Sort the docs so that they are in the same order before converting to
    # token label pairs.

    tlp = TokenLabelPairs(
        respect_doc_endings=respect_docs,
        respect_line_endings=respect_lines,
        line_limit=line_limit,
    )

    pairs_list = tlp.run(annotated_data)

    write_tsv(pairs_list, output_file)

    # Print out the first ten rows as a sense check

    msg.divider("Example output")
    header = ["token"] + ["label"] * len(annotated_data)
    aligns = ["r"] + ["l"] * len(annotated_data)
    formatted = table(pairs_list[0:ROWS_TO_PRINT], header=header, divider=True, aligns=aligns)
    print(formatted)

    msg.good(f"Wrote token/label pairs to {output_file}")
