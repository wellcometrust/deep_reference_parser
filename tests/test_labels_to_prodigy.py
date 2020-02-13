#!/usr/bin/env python3
# coding: utf-8

from deep_reference_parser.reference_utils import labels_to_prodigy

def test_labels_to_prodigy():

    tokens = [
        ['Ackerman', 'J', '.', 'S', '.,', 'Palladio', ',', 'Torino', '1972', '.']
    ]

    labels = [
        ['b-r', 'i-r', 'i-r', 'i-r', 'i-r', 'i-r', 'i-r', 'i-r', 'i-r', 'e-r']
    ]

    expected = [
        {
            "text": "Ackerman J . S ., Palladio , Torino 1972 .",
            "tokens": [
                {'text': 'Ackerman', 'id': 0, 'start': 0, 'end': 8},
                {'text': 'J', 'id': 1, 'start': 9, 'end': 10},
                {'text': '.', 'id': 2, 'start': 11, 'end': 12},
                {'text': 'S', 'id': 3, 'start': 13, 'end': 14},
                {'text': '.,', 'id': 4, 'start': 15, 'end': 17},
                {'text': 'Palladio', 'id': 5, 'start': 18, 'end': 26},
                {'text': ',', 'id': 6, 'start': 27, 'end': 28},
                {'text': 'Torino', 'id': 7, 'start': 29, 'end': 35},
                {'text': '1972', 'id': 8, 'start': 36, 'end': 40},
                {'text': '.', 'id': 9, 'start': 41, 'end': 42}
            ],
            "spans": [
                {'label': 'b-r', 'start': 0, 'end': 8, 'token_start': 0, 'token_end': 0},
                {'label': 'i-r', 'start': 9, 'end': 10, 'token_start': 1, 'token_end': 1},
                {'label': 'i-r', 'start': 11, 'end': 12, 'token_start': 2, 'token_end': 2},
                {'label': 'i-r', 'start': 13, 'end': 14, 'token_start': 3, 'token_end': 3},
                {'label': 'i-r', 'start': 15, 'end': 17, 'token_start': 4, 'token_end': 4},
                {'label': 'i-r', 'start': 18, 'end': 26, 'token_start': 5, 'token_end': 5},
                {'label': 'i-r', 'start': 27, 'end': 28, 'token_start': 6, 'token_end': 6},
                {'label': 'i-r', 'start': 29, 'end': 35, 'token_start': 7, 'token_end': 7},
                {'label': 'i-r', 'start': 36, 'end': 40, 'token_start': 8, 'token_end': 8},
                {'label': 'e-r', 'start': 41, 'end': 42, 'token_start': 9, 'token_end': 9}
            ],
            "meta": {"line": 0}
        }

    ]

    out = labels_to_prodigy(tokens, labels)

    assert out == expected

