#!/usr/bin/env python3
# coding: utf-8

import itertools

import plac

from ..io import read_jsonl, write_jsonl
from ..logger import logger


class TokenTagger:
    def __init__(self, task="splitting", lowercase=True, text=True):
        """
        Converts data in prodigy format with full reference spans to per-token
            spans

        Args:
            task (str): One of ["parsing", "splitting"]. See below further
                explanation.
            lowercase (bool): Automatically convert upper case annotations to
                lowercase under the parsing scenario.
            text (bool): Include the token text in the output span (very useful
                for debugging).

        Since the parsing, splitting, and classification tasks have quite
        different labelling requirements, this class behaves differently
        depending on which task is specified in the task argument.

        For splitting:

        Expects one of four labels for the spans:

        * BE: A complete reference
        * BI: A frgament of reference that captures the beginning but not the end
        * IE: A frgament of reference that captures the end but not the beginning
        * II: A fragment of a reference that captures neither the beginning nor the
            end .

        Depending on which label is applied the tokens within the span will be
        labelled differently as one of ["b-r", "i-r", "e-r", "o"].

        For parsing:

        Expects any arbitrary label for spans. All tokens within that span will
        be labelled with the same span.

        """

        self.out = []
        self.task = task
        self.lowercase = lowercase
        self.text = text

    def tag_doc(self, doc):
        """
        Tags a document with appropriate labels for the parsing task

        Args:
            doc(dict): A single document in prodigy dict format to be labelled.
        """

        bie_spans = self.reference_spans(doc["spans"], doc["tokens"], task=self.task)
        o_spans = self.outside_spans(bie_spans, doc["tokens"])

        # Flatten into one list.

        spans = itertools.chain(bie_spans, o_spans)

        # Sort by token id to ensure it is ordered.

        spans = sorted(spans, key=lambda k: k["token_start"])

        doc["spans"] = spans

        return doc

    def run(self, docs):
        """
        Main class method for tagging multiple documents.

        Args:
            docs(dict): A list of docs in prodigy dict format to be labelled.
        """

        for doc in docs:

            self.out.append(self.tag_doc(doc))

        return self.out

    def reference_spans(self, spans, tokens, task):
        """
        Given a whole reference span as labelled in prodigy, break this into
        appropriate single token spans depending on the label that was applied to
        the whole reference span.
        """
        split_spans = []

        if task == "splitting":

            for span in spans:
                if span["label"] in ["BE", "be"]:

                    split_spans.extend(
                        self.split_long_span(tokens, span, "b-r", "e-r", "i-r")
                    )

                elif span["label"] in ["BI", "bi"]:

                    split_spans.extend(
                        self.split_long_span(tokens, span, "b-r", "i-r", "i-r")
                    )

                elif span["label"] in ["IE", "ie"]:

                    split_spans.extend(
                        self.split_long_span(tokens, span, "i-r", "e-r", "i-r")
                    )

                elif span["label"] in ["II", "ii"]:

                    split_spans.extend(
                        self.split_long_span(tokens, span, "i-r", "i-r", "i-r")
                    )

        elif task == "parsing":

            for span in spans:
                if self.lowercase:
                    label = span["label"].lower()
                else:
                    label = span["label"]
                split_spans.extend(
                    self.split_long_span(tokens, span, label, label, label)
                )

        return split_spans

    def outside_spans(self, spans, tokens):
        """
        Label tokens with `o` if they are outside a reference

        Args:
            spans(list): Spans in prodigy format.
            tokens(list): Tokens in prodigy format.

        Returns:
            list: A list of spans in prodigy format that comprises the tokens which
                are outside of a reference.
        """
        # Get the diff between inside and outside tokens

        span_indices = set([span["token_start"] for span in spans])
        token_indices = set([token["id"] for token in tokens])

        outside_indices = token_indices - span_indices

        outside_spans = []

        for index in outside_indices:
            outside_spans.append(self.create_span(tokens, index, "o"))

        return outside_spans

    def create_span(self, tokens, index, label):
        """
        Given a list of tokens, (in prodigy format) and an index relating to one of
        those tokens, and a new label: create a single token span using the new
        label, and the token selected by `index`.
        """

        token = tokens[index]

        span = {
            "start": token["start"],
            "end": token["end"],
            "token_start": token["id"],
            "token_end": token["id"],
            "label": label,
        }

        if self.text:
            span["text"] = token["text"]

        return span

    def split_long_span(self, tokens, span, start_label, end_label, inside_label):
        """
        Split a multi-token span into `n` spans of lengh `1`, where `n=len(tokens)`
        """

        spans = []
        spans.append(self.create_span(tokens, span["token_start"], start_label))
        spans.append(self.create_span(tokens, span["token_end"], end_label))

        for index in range(span["token_start"] + 1, span["token_end"]):
            spans.append(self.create_span(tokens, index, inside_label))

        spans = sorted(spans, key=lambda k: k["token_start"])

        return spans


@plac.annotations(
    input_file=(
        "Path to jsonl file containing chunks of references in prodigy format.",
        "positional",
        None,
        str,
    ),
    output_file=(
        "Path to jsonl file into which fully annotate files will be saved.",
        "positional",
        None,
        str,
    ),
    task=(
        "Which task is being performed. Either splitting or parsing.",
        "positional",
        None,
        str,
    ),
    lowercase=(
        "Convert UPPER case reference labels to lower case token labels?",
        "flag",
        "f",
        bool,
    ),
    text=(
        "Output the token text in the span (useful for debugging).",
        "flag",
        "t",
        bool,
    ),
)
def reference_to_token_annotations(
    input_file, output_file, task="splitting", lowercase=False, text=False
):
    """
    Creates a span for every token from existing multi-token spans

    Converts a jsonl file output by prodigy (using prodigy db-out) with spans
    extending over more than a single token to individual token level spans.

    The rationale for this is that reference level annotations are much easier
    for humans to do, but not useful when training a token level model.

    This command functions in two ways:

    * task=splitting: For the splitting task where we are interested in
        labelling the beginning (b-r) and end (e-r) of references, reference
        spans are labelled with one of BI, BE, IE, II. These are then converted
        to token level spans b-r, i-r, e-r, and o using logic. Symbolically:
            * BE: [BE, BE, BE] becomes [b-r][i-r][e-r]
            * BI: [BI, BI, BI] becomes [b-r][i-r][i-r]
            * IE: [IE, IE, IE] becomes [i-r][i-r][e-r]
            * II: [II, II, II] becomes [i-r][i-r][i-r]
            * All other tokens become [o]

    * task=parsing: For the parsing task, multi-task annotations are much
        simpler and would tend to be just 'author', or 'title'. These simple
        labels can be applied directly to the individual tokens contained within
        these multi-token spans; for each token in the multi-token span, a span
        is created with the same label. Symbolically:
            * [author author author] becomes [author][author][author]
    """

    ref_annotated_docs = read_jsonl(input_file)

    # Only run the tagger on annotated examples.

    not_annotated_docs = [doc for doc in ref_annotated_docs if not doc.get("spans")]
    ref_annotated_docs = [doc for doc in ref_annotated_docs if doc.get("spans")]

    logger.info(
        "Loaded %s documents with reference annotations", len(ref_annotated_docs)
    )
    logger.info(
        "Loaded %s documents with no reference annotations", len(not_annotated_docs)
    )

    annotator = TokenTagger(task=task, lowercase=lowercase, text=text)

    token_annotated_docs = annotator.run(ref_annotated_docs)
    all_docs = token_annotated_docs + not_annotated_docs

    write_jsonl(all_docs, output_file=output_file)

    logger.info(
        "Wrote %s docs with token annotations to %s",
        len(token_annotated_docs),
        output_file,
    )
    logger.info(
        "Wrote %s docs with no annotations to %s", len(not_annotated_docs), output_file
    )
