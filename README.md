[![Build Status](https://travis-ci.org/wellcometrust/deep_reference_parser.svg?branch=master)](https://travis-ci.org/wellcometrust/deep_reference_parser)[![codecov](https://codecov.io/gh/wellcometrust/deep_reference_parser/branch/master/graph/badge.svg)](https://codecov.io/gh/wellcometrust/deep_reference_parser)

# Deep Reference Parser

Deep Reference Parser is a Deep Learning Model for recognising references in free text. In this context we mean references to other works, for example an academic paper, or a book. Given an arbitrary block of text (nominally a section containing references), the model will extract the limits of the individual references, and identify key information like: authors, year published, and title.

The model itself is a Bi-directional Long Short Term Memory (BiLSTM) Deep Neural Network with a stacked Conditional Random Field (CRF). It is designed to be used in the [Reach](https://github.com/wellcometrust/reach) application to replace a number of existing machine learning models which find references, and extract the constituent parts.

The BiLSTM model is based on [Rodrigues et al. (2018)](https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing) who developed a model to find (split) references, parse them into contituent parts, and classify them according to the type of reference (e.g. primary reference, secondary reference, etc). This implementation of the model implements a the first two tasks and is intened for use in the medical field. Three models are implemented here: individual splitting and parsing models, and a combined multitask model which both splits and parses. We have not yet attempted to include reference type classification, but this may be done in the future.

### Current status:

|Component|Individual|MultiTask|
|---|---|---|
|Spans (splitting)|✔️ Implemented|✔️ Implemented|
|Components (parsing)|✔️ Implemented|✔️ Implemented|
|Type (classification)|❌ Not Implemented|❌ Not Implemented|

### The model

The model itself is based on the work of [Rodrigues et al. (2018)](https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing), although the implemention here differs significantly. The main differences are:

* We use a combination of the training data used by Rodrigues, et al. (2018) in addition to data that we have annotated ourselves. No Rodrigues et al. data are included in the test and validation sets.
* We also use a new word embedding that has been trained on documents relevant to the field of medicine.
* Whereas Rodrigues at al. split documents on lines, and sent the lines to the model, we combine the lines of the document together, and then send larger chunks to the model, giving it more context to work with when training and predicting.
* Whilst the splitter model makes predictions at the token level, it outputs references by naively splitting on these tokens ([source](https://github.com/wellcometrust/deep_reference_parser/blob/master/deep_reference_parser/tokens_to_references.py)).
* Hyperparameters are passed to the model in a config (.ini) file. This is to keep track of experiments, but also because it is difficult to save the model with the CRF architecture, so it is necesary to rebuild (not re-train!) the model object each time you want to use it. Storing the hyperparameters in a config file makes this easier.
* The package ships with a [config file](https://github.com/wellcometrust/deep_reference_parser/blob/master/deep_reference_parser/configs/2020.3.19_multitask.ini) which defines the latest, highest performing model. The config file defines where to find the various objects required to build the model (index dictionaries, weights, embeddings), and will automatically fetch them when run, if they are not found locally.
* The model includes a command line interface inspired by [SpaCy](https://github.com/explosion/spaCy); functions can be called from the command line with `python -m deep_reference_parser` ([source](https://github.com/wellcometrust/deep_reference_parser/blob/master/deep_reference_parser/predict.py)).
* Python version updated to 3.7, along with dependencies (although more to do).

### Performance

On the validation set.

#### Finding references spans (splitting)

Current mode version: *2020.3.6_splitting*

|token|f1|
|---|---|
|b-r|0.8146|
|e-r|0.7075|
|i-r|0.9623|
|o|0.8463|
|weighted avg|0.9326|

#### Identifying reference components (parsing)

Current mode version: *2020.3.8_parsing*

|token|f1|
|---|---|
|author|0.9053|
|title|0.8607|
|year|0.0.8639|
|o|0.0.9340|
|weighted avg|0.9124|

#### Multitask model (splitting and parsing)

Current mode version: *2020.4.5_multitask*

|token|f1|
|---|---|
|author|0.9458|
|title|0.9002|
|year|0.8704|
|o|0.9407|
|parsing weighted avg|0.9285|
|b-r|0.9111|
|e-r|0.8788|
|i-r|0.9726|
|o|0.9332|
|weighted avg|0.9591|

#### Computing requirements

Models are trained on AWS instances using CPU only.

|Model|Time Taken|Instance type|Instance cost (p/h)|Total cost|
|---|---|---|---|---|
|Span detection|00:26:41|m4.4xlarge|$0.88|$0.39|
|Components|00:17:22|m4.4xlarge|$0.88|$0.25|
|MultiTask|00:42:43|m4.4xlarge|$0.88|$0.63|

## tl;dr: Just get me to the references!

```
# Install from github

pip install git+git://github.com/wellcometrust/deep_reference_parser.git#egg=deep_reference_parser


# Create references.txt with some references in it

cat > references.txt <<EOF
1 Sibbald, A, Eason, W, McAdam, J, and Hislop, A (2001). The establishment phase of a silvopastoral national network experiment in the UK. Agroforestry systems, 39, 39–53. 
2 Silva, J and Rego, F (2003). Root distribution of a Mediterranean shrubland in Portugal. Plant and Soil, 255 (2), 529–540. 
3 Sims, R, Schock, R, Adegbululgbe, A, Fenhann, J, Konstantinaviciute, I, Moomaw, W, Nimir, H, Schlamadinger, B, Torres-Martínez, J, Turner, C, Uchiyama, Y, Vuori, S, Wamukonya, N, and X. Zhang (2007). Energy Supply. In Metz, B, Davidson, O, Bosch, P, Dave, R, and Meyer, L (eds.), Climate Change 2007: Mitigation. Contribution of Working Group III to the Fourth Assessment Report of the Intergovernmental Panel on Climate Change, Cambridge University Press, Cambridge, United Kingdom and New York, NY, USA.
EOF


# Run the MultiTask model. This will take a little time while the weights and 
# embeddings are downloaded. The weights are about 300MB, and the embeddings 
# 950MB.

python -m deep_reference_parser split_parse -t "$(cat references.txt)"

# For parsing:

python -m deep_reference_parser parse "$(cat references.txt)"

# For splitting:

python -m deep_reference_parser split "$(cat references.txt)"

```

## The longer guide

### Installation

The package can be installed from github for now. Future versions may be available on pypi.

```
pip install git+git://github.com/wellcometrust/deep_reference_parser.git#egg=deep_reference_parser
```

### Config files

The package uses config files to store hyperparameters for the models. 

A [config file](https://github.com/wellcometrust/deep_reference_parser/blob/master/deep_reference_parser/configs/2019.12.0.ini) which describes the parameters of the best performing model ships with the package:

```
[DEFAULT]
version = 2020.3.19_multitask
description = Same as 2020.3.13 but with adam rather than rmsprop
deep_reference_parser_version = b61de984f95be36445287c40af4e65a403637692

[data]
test_proportion = 0.25
valid_proportion = 0.25
data_path = data/
respect_line_endings = 0
respect_doc_endings = 1
line_limit = 150
policy_train = data/multitask/2020.3.19_multitask_train.tsv
policy_test = data/multitask/2020.3.19_multitask_test.tsv
policy_valid = datamultitask/2020.3.19_multitask_valid.tsv
s3_slug = https://datalabs-public.s3.eu-west-2.amazonaws.com/deep_reference_parser/

[build]
output_path = models/multitask/2020.3.19_multitask/
output = crf
word_embeddings = embeddings/2020.1.1-wellcome-embeddings-300.txt
pretrained_embedding = 0
dropout = 0.5
lstm_hidden = 400
word_embedding_size = 300
char_embedding_size = 100
char_embedding_type = BILSTM
optimizer = rmsprop

[train]
epochs = 60
batch_size = 100
early_stopping_patience = 5
metric = val_f1
```

### Getting help

To get a list of the available commands run `python -m deep_reference_parser`

```
$ python -m deep_reference_parser
Using TensorFlow backend.

ℹ Available commands
parse, split, train 
```

For additional help, you can pass a command with the `-h`/`--help` flag:

```
$ python -m deep_reference_parser split --help
usage: deep_reference_parser split [-h]
                                   [-c]
                                   [-t] [-o None]
                                   text

    Runs the default splitting model and pretty prints results to console unless
    --outfile is parsed with a path. Can output either tokens (with -t|--tokens)
    or split naively into references based on the b-r tag (default).

    NOTE: that this function is provided for examples only and should not be used
    in production as the model is instantiated each time the command is run. To
    use in a production setting, a more sensible approach would be to replicate
    the split or parse functions within your own logic.
    

positional arguments:
  text                  Plaintext from which to extract references

optional arguments:
  -h, --help            show this help message and exit
  -c, --config-file     Path to config file
  -t, --tokens          Output tokens instead of complete references
  -o, --outfile         Path to json file to which results will be written


```

### Training your own models

To train your own models you will need to define the model hyperparameters in a config file like the one above. The config file is then passed to the train command as the only argument. Note that the `output_path` defined in the config file will be created if it doesn not already exist.

```
python -m deep_reference_parser train test.ini
```

Data must be prepared in the following tab separated format (tsv). We use [prodi.gy](https://prodi.gy) for annotations. Some utilities to help with manual annotations and various format conversions are available in the [prodigy](./prodigy/) module. Data for reference span detection follows an IOBE schema.

You must provide the train/test/validation data splits in this format in pre-prepared files that are defined in the config file.

```
References  o o
1   o   o
The	b-r title
potency	i-r title
of	i-r title
history	i-r title
was	i-r title
on	i-r title
display	i-r title
at	i-r title
a	i-r title
workshop    i-r title
held	i-r title
in	i-r title
February	i-r title
```

### Making predictions

If you wish to use the latest model that we have trained, you can simply run:

```
python -m deep_reference_parser split <input text>
```

If you wish to use a custom model that you have trained, you must specify the config file which defines the hyperparameters for that model using the `-c` flag:

```
python -m deep_reference_parser split -c new_model.ini <input text>
```

Use the `-t` flag to return the raw token predictions, and the `-v` to return everything in a much more user friendly format.

Note that the model makes predictions at the token level, but a naive splitting is performed by simply splitting on the `b-r` tags.

### Developing the package further

To create a local virtual environment and activate it:

```
make virtualenv

# to activate

source ./build/virtualenv/bin/activate
```

## Get the data, models, and embeddings

```
make data models embeddings
```

## Testing

The package uses pytest:

```
make test
```

## References

Rodrigues Alves, D., Colavizza, G., & Kaplan, F. (2018). Deep Reference Mining From Scholarly Literature in the Arts and Humanities. Frontiers in Research Metrics and Analytics, 3(July), 1–13. https://doi.org/10.3389/frma.2018.00021
