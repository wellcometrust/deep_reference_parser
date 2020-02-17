# coding: utf8

"""
Modified from https://github.com/explosion/spaCy/blob/master/spacy/__main__.py

"""

if __name__ == "__main__":
    import plac
    import sys
    from wasabi import msg
    from .numbered_reference_annotator import annotate_numbered_references
    from .prodigy_to_tsv import prodigy_to_tsv
    from .reach_to_prodigy import reach_to_prodigy
    from .reference_to_token_annotations import reference_to_token_annotations

    commands = {
        "annotate_numbered_refs": annotate_numbered_references,
        "prodigy_to_tsv": prodigy_to_tsv,
        "reach_to_prodigy": reach_to_prodigy,
        "refs_to_token_annotations": reference_to_token_annotations,
    }

    if len(sys.argv) == 1:
        msg.info("Available commands", ", ".join(commands), exits=1)
    command = sys.argv.pop(1)
    sys.argv[0] = "deep_reference_parser %s" % command

    if command in commands:
        plac.call(commands[command], sys.argv[1:])
    else:
        available = "Available: {}".format(", ".join(commands))
        msg.fail("Unknown command: {}".format(command), available, exits=1)
