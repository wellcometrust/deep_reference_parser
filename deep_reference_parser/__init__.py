# Tensorflow and Keras emikt a very large number of warnings that are very 
# distracting on the command line. These lines here (while undesireable) 
# reduce the level of verbosity.

import sys
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from .common import download_model_artefact
from .deep_reference_parser import DeepReferenceParser
from .logger import logger
from .model_utils import get_config
from .reference_utils import (
    break_into_chunks,
    labels_to_prodigy,
    load_data,
    load_tsv,
    prodigy_to_conll,
    prodigy_to_lists,
    read_jsonl,
    read_pickle,
    write_json,
    write_jsonl,
    write_pickle,
    write_to_csv,
    write_txt,
)
from .tokens_to_references import tokens_to_references
