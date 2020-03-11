#!/usr/bin/env python3
# coding: utf-8

"""
"""

import csv
import json
import os
import pickle

import spacy

from .logger import logger


def load_data(filepath):
    """
    Load and return the data stored in the given path.

    Adapted from: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    The data is structured as follows:
     * Each line contains four columns separated by a single space.
     * Each word has been put on a separate line and there is an empty line
        after each sentence.
     * The first item on each line is a word, the second, third and fourth are
        tags related to the word.

    Example:

    The sentence "L. Antonielli, Iprefetti dell' Italia napoleonica, Bologna
        1983." is represented in the dataset as:

    ```
    L author b-secondary b-r
    . author i-secondary i-r
    Antonielli author i-secondary i-r
    , author i-secondary i-r
    Iprefetti title i-secondary i-r
    dell title i-secondary i-r
    ’ title i-secondary i-r
    Italia title i-secondary i-r
    napoleonica title i-secondary i-r
    , title i-secondary i-r
    Bologna publicationplace i-secondary i-r
    1983 year e-secondary i-r
    . year e-secondary e-r
    ```

    Args:
        filepath (str): Path to the data.

    Returns:
        four lists: The first contains tokens, the next three contain
            corresponding labels.

    """

    # Arrays to return
    words = []
    tags_1 = []
    tags_2 = []
    tags_3 = []

    word = tags1 = tags2 = tags3 = []
    with open(filepath, "r") as file:
        for line in file:
            # Do not take the first line into consideration

            if "DOCSTART" not in line:
                # Check if empty line

                if line in ["\n", "\r\n"]:
                    # Append line

                    words.append(word)
                    tags_1.append(tags1)
                    tags_2.append(tags2)
                    tags_3.append(tags3)

                    # Reset
                    word = []
                    tags1 = []
                    tags2 = []
                    tags3 = []

                else:
                    # Split the line into words, tag #1
                    w = line[:-1].split(" ")

                    word.append(w[0])
                    tags1.append(w[1])
                    tags2.append(w[2])
                    tags3.append(w[3])

    logger.info("Loaded %s training examples", len(words))

    return words, tags_1, tags_2, tags_3


def load_tsv(filepath, split_char="\t"):
    """
    Load and return the data stored in the given path.

    Adapted from: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    NOTE: In the current implementation in deep_reference_parser, only one set
    of tags is used. The others will be used in a later PR.

    The data is structured as follows:
     * Each line contains four columns separated by a single space.
     * Each word has been put on a separate line and there is an empty line
        after each sentence.
     * The first item on each line is a word, the second, third and fourth are
        tags related to the word.

    Args:
        filepath (str): Path to the data.
        split_char(str): Character to be used to split each line of the
            document.

    Returns:
        two lists: The first contains tokens, the second contains corresponding
        labels.

    """

    # Arrays to return
    words = []
    tags_1 = []

    word = []
    tags1 = []

    with open(filepath, "r") as file:
        for line in file:
            # Check if empty line

            if line in ["\n", "\r\n", "\t\n"]:
                # Append line

                words.append(word)
                tags_1.append(tags1)

                # Reset
                word = []
                tags1 = []

            else:

                # Split the line into words, tag #1

                w = line[:-1].split(split_char)
                word.append(w[0])

                # If tags are passed, (for training) then also add

                if len(w) == 2:

                    tags1.append(w[1])

    logger.info("Loaded %s training examples", len(words))

    return words, tags_1


def prodigy_to_conll(docs):
    """
    Expect list of jsons loaded from a jsonl
    """

    nlp = spacy.load("en_core_web_sm")
    texts = [doc["text"] for doc in docs]
    docs = list(nlp.tokenizer.pipe(texts))

    out = [_join_prodigy_tokens(i) for i in docs]

    out_str = "DOCSTART\n\n" + "\n\n".join(out)

    return out_str


def prodigy_to_lists(docs):
    """
    Expect list of jsons loaded from a jsonl
    """

    nlp = spacy.load("en_core_web_sm")
    texts = [doc["text"] for doc in docs]
    docs = list(nlp.tokenizer.pipe(texts))

    out = [[str(token) for token in doc] for doc in docs]

    return out


def _join_prodigy_tokens(text):
    """Return all prodigy tokens in a single string
    """

    return "\n".join([str(i) for i in text])


def write_json(input_data, output_file, path=None):
    """
    Write a dict to json

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): A filename or path to which the json will be saved.
        path(str): A string which will be prepended onto `output_file` with
            `os.path.join()`. Obviates the need for lengthy `os.path.join`
            statements each time this function is called.
    """

    if path:

        output_file = os.path.join(path, output_file)

    logger.info("Writing data to %s", output_file)

    with open(output_file, "w") as fb:
        fb.write(json.dumps(input_data))


def write_jsonl(input_data, output_file, path=None):
    """
    Write a dict to jsonl (line delimited json)

    Output format will look like:

    ```
    {"a": 0}
    {"b": 1}
    {"c": 2}
    {"d": 3}
    ```

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): A filename or path to which the json will be saved.
        path(str): A string which will be prepended onto `output_file` with
            `os.path.join()`. Obviates the need for lengthy `os.path.join`
            statements each time this function is called.
    """

    if path:

        output_file = os.path.join(path, output_file)

    with open(output_file, "w") as fb:

        # Check if a dict (and convert to list if so)

        if isinstance(input_data, dict):
            input_data = [value for key, value in input_data.items()]

        # Write out to jsonl file

        logger.info("Writing %s lines to %s", len(input_data), output_file)

        for i in input_data:
            json_ = json.dumps(i) + "\n"
            fb.write(json_)


def read_jsonl(input_file, path=None):
    """Create a list from a jsonl file

    Args:
        input_file(str): File to be loaded.
        path(str): A string which will be prepended onto `input_file` with
            `os.path.join()`. Obviates the need for lengthy `os.path.join`
            statements each time this function is called.
    """

    if path:
        input_file = os.path.join(path, input_file)

    out = []
    with open(input_file, "r") as fb:

        logger.info("Reading contents of %s", input_file)

        for i in fb:
            out.append(json.loads(i))

    logger.info("Read %s lines from %s", len(out), input_file)

    return out


def write_txt(input_data, output_file):
    """Write a text string to a file

    Args:
        input_file (str): String to be written
        output_file (str): File to be saved to
    """

    with open(output_file, "w") as fb:
        fb.write(input_data)

    logger.info("Read %s characters to file: %s", len(input_data), output_file)


def labels_to_prodigy(tokens, labels):
    """
    Converts a list of tokens and labels like those used by Rodrigues et al,
    and converts to prodigy format dicts.

    Args:
        tokens (list): A list of tokens.
        labels (list): A list of labels relating to `tokens`.

    Returns:
        A list of prodigy format dicts containing annotated data.
    """

    prodigy_data = []

    all_token_index = 0

    for line_index, line in enumerate(tokens):
        prodigy_example = {}

        tokens = []
        spans = []
        token_start_offset = 0

        for token_index, token in enumerate(line):

            token_end_offset = token_start_offset + len(token)

            tokens.append(
                {
                    "text": token,
                    "id": token_index,
                    "start": token_start_offset,
                    "end": token_end_offset,
                }
            )

            spans.append(
                {
                    "label": labels[line_index][token_index : token_index + 1][0],
                    "start": token_start_offset,
                    "end": token_end_offset,
                    "token_start": token_index,
                    "token_end": token_index,
                }
            )

            prodigy_example["text"] = " ".join(line)
            prodigy_example["tokens"] = tokens
            prodigy_example["spans"] = spans
            prodigy_example["meta"] = {"line": line_index}

            token_start_offset = token_end_offset + 1

        prodigy_data.append(prodigy_example)

    return prodigy_data


def write_to_csv(filename, columns, rows):
    """
    Create a .csv file from data given as columns and rows

    Args:
        filename(str): Path and name of the .csv file, without csv extension
        columns(list): Columns of the csv file (First row of the file)
        rows: Data to write into the csv file, given per row
    """

    with open(filename, "w") as csvfile:
        wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        wr.writerow(columns)

        for i, row in enumerate(rows):
            wr.writerow(row)
    logger.info("Wrote results to %s", filename)


def write_pickle(input_data, output_file, path=None):
    """
    Write an object to pickle

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): A filename or path to which the json will be saved.
        path(str): A string which will be prepended onto `output_file` with
            `os.path.join()`. Obviates the need for lengthy `os.path.join`
            statements each time this function is called.
    """

    if path:

        output_file = os.path.join(path, output_file)

    with open(output_file, "wb") as fb:
        pickle.dump(input_data, fb)


def read_pickle(input_file, path=None):
    """Create a list from a jsonl file

    Args:
        input_file(str): File to be loaded.
        path(str): A string which will be prepended onto `input_file` with
            `os.path.join()`. Obviates the need for lengthy `os.path.join`
            statements each time this function is called.
    """

    if path:
        input_file = os.path.join(path, input_file)

    with open(input_file, "rb") as fb:
        out = pickle.load(fb)

    logger.debug("Read data from %s", input_file)

    return out


def yield_token_label_pairs(tokens, labels):
    """
    Convert matching lists of tokens and labels to tuples of (token, label) but
    preserving the nexted list boundaries as (None, None).

    Args:
        tokens(list): list of tokens.
        labels(list): list of labels corresponding to tokens.
    """

    for tokens, labels in zip(tokens, labels):
        if tokens and labels:
            for token, label in zip(tokens, labels):
                yield (token, label)
            yield (None, None)
        else:
            yield (None, None)


def write_tsv(token_label_pairs, output_path):
    """
    Write tsv files to disk
    """
    with open(output_path, "w") as fb:
        writer = csv.writer(fb, delimiter="\t")
        writer.writerows(token_label_pairs)


def break_into_chunks(doc, max_words=250):
    """
    Breaks a list into lists of lists of length max_words
    Also works on lists:

    >>> doc = ["a", "b", "c", "d", "e"]
    >>> break_into_chunks(doc, max_words=2)
        [['a', 'b'], ['c', 'd'], ['e']]
    """
    out = []
    chunk = []
    for i, token in enumerate(doc, 1):
        chunk.append(token)
        if (i > 0 and i % max_words == 0) or i == len(doc):
            out.append(chunk)
            chunk = []
    return out
