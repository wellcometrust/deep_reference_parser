#!/usr/bin/env python3
# coding: utf-8

"""
Utilities for loading and saving data from various formats
"""

import json
import pickle
import csv
import os
import pandas as pd

from ..logger import logger

def _unpack(tuples):
    """Convert list of tuples into the correct format:

    From:

        [
            (
                (token0, token1, token2, token3),
                (label0, label1, label2, label3),
            ),
            (
                (token0, token1, token2),
                (label0, label1, label2),
            ),
        )

    to:
        ]
            (
                (token0, token1, token2, token3),
                (token0, token1, token2),
            ),
            (
                (label0, label1, label2, label3),
                (label0, label1, label2),
            ),
        ]
    """
    return list(zip(*list(tuples)))

def _split_list_by_linebreaks(rows):
    """Cycle through a list of tokens (or labels) and split them into lists
    based on the presence of Nones or more likely math.nan caused by converting
    pd.DataFrame columns to lists.
    """
    out = []
    rows_gen = iter(rows)
    while True:
        try:
            row = next(rows_gen)
            token = row[0]
            if isinstance(token, str) and token:
                out.append(row)
            else:
                yield out
                out = []
        except StopIteration:
            if out:
                yield out
            break

def load_tsv(filepath, split_char="\t"):
    """
    Load and return the data stored in the given path.

    Expects data in the following format (tab separations).

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

    Args:
        filepath (str): Path to the data.
        split_char(str): Character to be used to split each line of the
            document.

    Returns:
        a series of lists depending on the number of label columns provided in 
        filepath.

    """
    df = pd.read_csv(filepath, delimiter=split_char, header=None, skip_blank_lines=False)
    tuples = _split_list_by_linebreaks(df.to_records(index=False))

    # Remove leading empty lists if found

    tuples = list(filter(None, tuples))

    unpacked_tuples = list(map(_unpack, tuples))

    out = _unpack(unpacked_tuples)

    logger.info("Loaded %s training examples", len(out[0]))

    return tuple(out)

def write_jsonl(input_data, output_file):
    """
    Write a dict to jsonl (line delimited json)

    Output format will look like:

    ```
    {'a': 0}
    {'b': 1}
    {'c': 2}
    {'d': 3}
    ```

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): Filename to which the jsonl will be saved.
    """

    with open(output_file, "w") as fb:

        # Check if a dict (and convert to list if so)

        if isinstance(input_data, dict):
            input_data = [value for key, value in input_data.items()]

        # Write out to jsonl file

        logger.debug("Writing %s lines to %s", len(input_data), output_file)

        for i in input_data:
            json_ = json.dumps(i) + "\n"
            fb.write(json_)


def _yield_jsonl(file_name):
    for row in open(file_name, "r"):
        yield json.loads(row)


def read_jsonl(input_file):
    """Create a list from a jsonl file

    Args:
        input_file(str): File to be loaded.
    """

    out = list(_yield_jsonl(input_file))

    logger.debug("Read %s lines from %s", len(out), input_file)

    return out


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

def write_tsv(token_label_pairs, output_path):
    """
    Write tsv files to disk
    """
    with open(output_path, "w") as fb:
        writer = csv.writer(fb, delimiter="\t")
        writer.writerows(token_label_pairs)
