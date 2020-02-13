# Prodigy utilities

The `deep_reference_parser.prodigy` module contains a number of utility functions for working with annotations created in [prodi.gy](http://prodi.gy).

The individual functions can be access with the usual `import deep_reference_parser.prodigy` logic, but can also be accessed on the command line with:

```
$ python -m deep_reference_parser.prodigy
Using TensorFlow backend.

ℹ Available commands
annotate_numbered_refs, prodigy_to_tsv, reach_to_prodigy,
refs_to_token_annotations
```

|Name|Description|
|---|---|
|reach_to_prodigy|Converts a jsonl of reference sections output by reach into a jsonl containing prodigy format documents.|
|annotate_numbered_refs|Takes numbered reference sections extract by Reach, and roughly annotates the references by splitting the reference lines apart on the numbers.|
|prodigy_to_tsv|Converts a jsonl file of prodigy documents to a tab separated values (tsv) file where each token and its associated label occupy a line.|
|refs_to_token_annotations|Takes a jsonl of annotated reference sections in prodigy format that have been manually annotated to the reference level, and converts the references into token level annotations based on the IOBE schema, saving a new file or prodigy documents to jsonl.|

Help for each of these commands can be sought with the `--help` flag, e.g.:

```
$ python -m deep_reference_parser.prodigy prodigy_to_tsv --help
Using TensorFlow backend.
usage: deep_reference_parser prodigy_to_tsv [-h] input_file output_file

    Convert token annotated jsonl to token annotated tsv ready for use in the
    Rodrigues model.
    

positional arguments:
  input_file   Path to jsonl file containing prodigy docs.
  output_file  Path to output tsv file.

optional arguments:
  -h, --help   show this help message and exit

```

