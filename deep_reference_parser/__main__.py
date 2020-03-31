# coding: utf8

"""
Modified from https://github.com/explosion/spaCy/blob/master/spacy/__main__.py

"""

if __name__ == "__main__":
    import plac
    import sys
    from wasabi import msg
    from .train import train
    from .split import split
    from .parse import parse
    from .split_parse import split_parse

    commands = {
        "split": split,
        "parse": parse,
        "train": train,
        "split_parse": split_parse,
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
