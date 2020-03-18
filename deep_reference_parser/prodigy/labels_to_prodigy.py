
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
