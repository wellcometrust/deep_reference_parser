#!/usr/bin/env python3
# coding: utf-8

"""
"""

import configparser
import itertools

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.utils import save_load_utils
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn_crfsuite import metrics

from .logger import logger

matplotlib.use("agg")


def get_config(path):
    """
    Returns the config object

    Note that empty values in the config will be returned as None
    """
    try:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(path)
        logger.debug("Loaded config from %s", path)
    except:
        logger.exception("Exception loading config file %s", path)

    return config


def merge_digits(datasets, digits_word):
    """
    Map digits to a symbolic token

    From: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    Args:
        data(list): The data to transform
        digit_word(str): The word to map digits to

    Returns:
        list: The data transformed data
    """

    logger.debug("Mapping digits to %s", digits_word)

    return [
        [
            [digits_word if character.isdigit() else character for character in word]
            for word in data
        ]
        for data in datasets
    ]


def encode_x(x, word2ind, max_len, ukn_words, padding_style):
    """
    Transform a data of words in a data of integers, where each entry as the
    same length.

    From: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    Args:
        x(list): The data to transform, for example X_train_w.
        word2ind(dict): Dictionary to retrieve the integer for each word in the
            data.
        max_len(int): The length of each entry in the returned data.
        ukn_words(str): Key, in the dictionary words-index, to use for words.
            not present in the dictionary.
        padding_style(str): Either `"pre"` or `"post"`. Defines padding style
            to be used to standardise the length of each entry.

    Returns:
        Transformed data.
    """
    logger.debug("Encoding %s training examples", len(x))

    # Encode: Map each words to the corresponding integer
    # TODO: Make this easier to understand.

    encoded = []

    for sentence in x:
        encoded_sentence = []

        for word in sentence:
            word_index = word2ind.get(word)

            if word_index:
                encoded_sentence.append(word_index)
            else:
                encoded_sentence.append(word2ind[ukn_words])
        encoded.append(encoded_sentence)

    # Pad: Each entry in the data must have the same length

    padded = pad_sequences(encoded, maxlen=max_len, padding=padding_style)

    return padded


def _encode(x, n):
    """
    Return an array of zeros, except for an entry set to 1 (one-hot-encode)

    Args:
        x: Index entry to set to 1.
        n(int): Length of the array to return.

    Returns
        np.array: The created array.
    """
    result = np.zeros(n)
    result[x] = 1

    return result


def encode_y(y, label2ind, max_len, padding_style):
    """
    Apply one-hot-encoding to each label in the dataset

    Each entry will have the same length.

    >>> label2ind = {Label_A:1, Label_B:2, Label_C:3}
    >>> max_len = 4
    >>> y = [[Label_A, Label_C], [Label_A, Label_B, Label_C]]
    >>> encode_pad_data_y(y, label2ind, max_len, padding_style="pre")
    [[[1,0,0], [0,0,1], [0,0,0], [0,0,0]], [[1,0,0], [0,1,0], [0,0,1]], [0,0,0]]

    Args:
        y(list): The data to encode, e.g. y_train_w.
        label2ind(dict):  Dictionary where each value in the data is mapped to a
            unique integer
        max_len: The length of each entry in the returned data
        padding_style: Padding style to use for having each entry in the data
            with the same length

    Returns:
        The transformed data
    """

    logger.debug("Encoding %s training examples", len(y))

    # Encode y (with pad)

    # Transform each label into its index and adding "pre" padding

    y_pad = [[0] * (max_len - len(yi)) + [label2ind[label] for label in yi] for yi in y]

    # One-hot-encode label

    max_label = max(label2ind.values()) + 1
    y_enc = [[_encode(c, max_label) for c in ey] for ey in y_pad]

    # Repad (to have numpy array)

    y_encode = pad_sequences(y_enc, maxlen=max_len, padding=padding_style)

    return y_encode


def character_index(X, digits_word):
    """
    Map each character present in the dataset into an unique integer.

    From: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    All digits are mapped into a single array.

    Args:
        X(list): Data to retrieve characters from.
        digits_word(str): Words regrouping all digits

    Returns:
        A dictionary where each character is maped into a unique integer, the
            maximum number of words in the data, the maximum of characters in
            a word.
    """

    # Create a set of all character

    all_chars = list(set([c for s in X for w in s for c in w]))

    # Create an index for each character. The index 1 is reserved for the
    # digits, substituted by digits_word.

    char2ind = {char: index for index, char in enumerate(all_chars, 2)}
    ind2char = {index: char for index, char in enumerate(all_chars, 2)}

    # To deal with out-of-vocabulary words

    char2ind.update({digits_word: 1})
    ind2char.update({1: digits_word})

    # For padding

    max_words = max([len(s) for s in X])
    max_char = max([len(w) for s in X for w in s])

    return char2ind, max_words, max_char


def character_data(X, char2ind, max_words, max_char, digits_word, padding_style):
    """
    For each word in the data, transform it into an array of characters.

    All character arrays will have the same length. All sequence will have the
    same array length. All digits will be maped to the same character array.

    If a character is not present in the dictionary character-index, discard it.

    From: https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    Args:
        X(list): The data
        chra2ind(dict): Dictionary where each character is mapped to a unique
            integer.
        max_words: Maximum number of words in a sequence.
        max_char: Maximum number of characters in a word.
        digits_word: Word regrouping all digits.
        padding_style: Padding style to use for having each entry in the data
            with the same length

    Returns:
        The transformed array.
    """

    # Transform each word into an array of characters (discards those oov)

    X_char = [
        [
            [char2ind[c] for c in w if c in char2ind.keys()]
            if w != digits_word
            else [1]
            for w in s
        ]
        for s in X
    ]

    # Pad words - Each words has the same number of characters

    X_char = pad_sequences(
        [pad_sequences(s, max_char, padding=padding_style) for s in X_char],
        max_words,
        padding=padding_style,
    )

    return X_char


def index_x(x, ukn_words):
    """
    Map each word in the given data to a unique integer.

    A special index will be kept for "out-of-vocabulary" words.

    Args:
        x(list): The data

    Returns:
        Two dictionaries: one where words are keys and indexes values, another
            one "reversed" (keys->index, values->words)
    """

    # Retrieve all words used in the data (with duplicates)

    all_text = [w for e in x for w in e]

    # Compute the unique words (remove duplicates)

    words = list(set(all_text))

    logger.debug("Number of entries: %s", len(all_text))
    logger.debug("Individual entries: %s", len(words))

    # Assign an integer index for each individual word

    word2ind = {word: index for index, word in enumerate(words, 2)}
    ind2word = {index: word for index, word in enumerate(words, 2)}

    # To deal with out-of-vocabulary words

    word2ind.update({ukn_words: 1})
    ind2word.update({1: ukn_words})

    # The index '0' is kept free in both dictionaries

    return word2ind, ind2word


def index_y(y):
    """
    Map each word in the given data to a unique integer.

    Args:
        y(list): The data

    Returns:
        Two dictionaries: one where words are keys and indexes values, another
            one "reversed" (keys->index, values->words).
    """

    # Unique attributes in the data, sort alphabetically

    labels_t1 = list(set([w for e in y for w in e]))
    labels_t1 = sorted(labels_t1, key=str.lower)

    logger.debug("Number of labels: %s", len(labels_t1))

    # Assign an integer index for each individual label

    label2ind = {label: index for index, label in enumerate(labels_t1, 1)}
    ind2label = {index: label for index, label in enumerate(labels_t1, 1)}

    # The index '0' is kept free in both dictionaries

    return label2ind, ind2label


def word2vec_embeddings(embedding_path, word2ind, word_embedding_size):
    """
    Convert word embeddings to dictionary

    Will return dict of the format: `{word: embedding vector}`.
    Only return words of interest. If the word isn't in the embedding, returns
    a zero-vector instead.

    Args:
        embedding_path(str): Path to embedding (which will be loaded).
        word2ind(dict): Dictionary {words: index}. The keys represented the
            words for each embeddings will be retrieved.
        word_embedding_size(int): Size of the embedding vectors.

    Returns:
        Array of embeddings vectors. The embeddings vector at position i
        corresponds to the word with value i in the dictionary param `word2ind`
    """

    # Pre-trained embeddings filepath

    ukn_index = "$UKN$"

    # Read the embeddings file

    embeddings_all = {}
    with open(embedding_path, "r") as file:
        logger.debug("Reading embedding from %s", embedding_path)

        for line in file:
            l = line.strip().split(" ")
            embeddings_all[l[0]] = l[1:]

    # Compute the embedding for each word in the dataset

    embedding_matrix = np.zeros((len(word2ind) + 1, word_embedding_size))

    for word, i in word2ind.items():
        if word in embeddings_all:
            embedding_matrix[i] = embeddings_all[word]

    # NOTE: Not sure why this is commented out. This came from the original
    # code by Giovanni.

    #        else:
    #           embedding_matrix[i] = embeddings_all[ukn_index]

    # Delete the word2vec dictionary from memory
    del embeddings_all

    return embedding_matrix


class Classification_Scores(Callback):
    """
    Add the F1 score on the testing data at the end of each epoch.

    In case of multi-outputs, compute the F1 score for each output layer and
    the mean of all F1 scores.

    Compute the training F1 score for each epoch. Store the results internally.
    Internally, the accuracy and recall scores will also be stored, both for
    training and testing dataset.

    The model's weigths for the best epoch will be save in a given folder.
    """

    def __init__(self, train_data, ind2label, model_save_path):
        """
        Args:
            train_data(list): The data used to compute training accuracy. One
                array of two arrays => [X_train, y_train].
            ind2label(dict): Dictionary of index-label to add tags label into
                results.
            model_save_path(str): Path to save the best model's weigths
        """
        self.train_data = train_data
        self.ind2label = ind2label
        self.model_save_path = model_save_path
        self.score_name = "val_f1"

    def on_train_begin(self, logs={}):
        self.test_report = []
        self.test_f1s = []
        self.test_acc = []
        self.test_recall = []
        self.train_f1s = []
        self.train_acc = []
        self.train_recall = []

        self.best_score = -1

        # Add F1-score as a metric to print at end of each epoch
        self.params["metrics"].append("val_f1")

        # In case of multiple outputs

        if len(self.model.layers) > 1:
            for output_layer in self.model.layers:
                self.params["metrics"].append("val_" + output_layer.name + "_f1")

    def compute_scores(self, pred, targ):
        """
        Compute the Accuracy, Recall and F1 scores between the two given arrays
            pred and targ (targ is the golden truth)
        """
        val_predict = np.argmax(pred, axis=-1)
        val_targ = np.argmax(targ, axis=-1)

        # Flatten arrays for sklearn
        predict_flat = np.ravel(val_predict)
        targ_flat = np.ravel(val_targ)

        # Compute scores

        labels = [x for x in np.unique(targ_flat) if x != 0]

        out = precision_recall_fscore_support(
            targ_flat, predict_flat, average="weighted", labels=labels
        )[:3]

        return out

    def compute_epoch_training_F1(self):
        """
        Compute and save the F1 score for the training data
        """

        in_length = len(self.model._input_layers)
        out_length = len(self.model._output_layers)
        predictions = self.model.predict(self.train_data[0])

        if len(predictions) != out_length:
            predictions = [predictions]

        vals_acc = []
        vals_recall = []
        vals_f1 = []

        # NOTE: It seems like this for loop is not actually necessary. This may
        # be the case for a single output model, but for a multitask model the
        # accuracy is calculated at the per prediction level, and then averaged?

        for i, pred in enumerate(predictions):
            _val_acc, _val_recall, _val_f1 = self.compute_scores(
                np.asarray(pred), self.train_data[1][i]
            )

            vals_acc.append(_val_acc)
            vals_recall.append(_val_recall)
            vals_f1.append(_val_f1)

        self.train_acc.append(sum(vals_acc) / len(vals_acc))
        self.train_recall.append(sum(vals_recall) / len(vals_recall))
        self.train_f1s.append(sum(vals_f1) / len(vals_f1))

    def classification_report(self, i, pred, targ, printPadding=False):
        """
        Compute the classification report for the predictions given.
        """

        # Hold all classification reports
        reports = []

        # The model predicts probabilities for each tag.
        # Retrieve the id of the most probable tag.

        pred_index = np.argmax(pred, axis=-1)

        # Reverse the one-hot encoding for target

        true_index = np.argmax(targ, axis=-1)

        # Index 0 in the predictions referes to padding

        ind2labelNew = self.ind2label[i].copy()
        ind2labelNew.update({0: "null"})

        # Compute the labels for each prediction

        pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]
        true_label = [[ind2labelNew[x] for x in b] for b in true_index]

        # CLASSIFICATION REPORTS

        reports.append("")

        if printPadding:
            reports.append("With padding into account")
            reports.append(
                metrics.flat_classification_report(true_label, pred_label, digits=4)
            )
            reports.append("")
            reports.append("----------------------------------------------")
            reports.append("")
            reports.append("Without the padding:")
        reports.append(
            metrics.flat_classification_report(
                true_label,
                pred_label,
                digits=4,
                labels=list(self.ind2label[i].values()),
            )
        )

        return "\n".join(reports)

    def on_epoch_end(self, epoch, logs={}):
        """
        At the end of each epoch, compute the F1 score for the validation data.

        In case of multi-outputs model, compute one value per output and average
        all to return the overall F1 score.

        Same model's weights for the best epoch.
        """
        self.compute_epoch_training_F1()

        # X data - to predict from

        in_length = len(self.model._input_layers)

        # Number of tasks

        out_length = len(self.model._output_layers)

        # Compute the model predictions

        predictions = self.model.predict(self.validation_data[:in_length])

        # In case of single output

        if len(predictions) != out_length:
            predictions = [predictions]

        vals_acc = []
        vals_recall = []
        vals_f1 = []
        reports = ""
        # Iterate over all output predictions

        for i, pred in enumerate(predictions):
            _val_acc, _val_recall, _val_f1 = self.compute_scores(
                np.asarray(pred), self.validation_data[in_length + i]
            )

            # Classification report

            reports += "For task " + str(i + 1) + "\n"
            reports += "===================================================================================="
            reports += (
                self.classification_report(
                    i, np.asarray(pred), self.validation_data[in_length + i]
                )
                + "\n\n\n"
            )

            # Add scores internally
            vals_acc.append(_val_acc)
            vals_recall.append(_val_recall)
            vals_f1.append(_val_f1)

            # Add F1 score to be log
            f1_name = "val_" + self.model.layers[i].name + "_f1"
            logs[f1_name] = _val_f1

        # Add classification reports for all the predicitions/tasks
        self.test_report.append(reports)

        # Add internally
        self.test_acc.append(sum(vals_acc) / len(vals_acc))
        self.test_recall.append(sum(vals_recall) / len(vals_recall))
        self.test_f1s.append(sum(vals_f1) / len(vals_f1))

        # Add to log
        f1_mean = sum(vals_f1) / len(vals_f1)
        logs["val_f1"] = f1_mean

        # Save best model's weights

        if f1_mean > self.best_score:
            self.best_score = f1_mean
            save_load_utils.save_all_weights(self.model, self.model_save_path)


def save_confusion_matrix(target, preds, labels, figure_path, figure_size=(20, 20)):
    """
    Generate two confusion matrices plots: with and without normalization.

    Args:
        target: Tags groud truth
        pred: Tags predictions
        labels(list): Predictions classes to use
        figure_path(str): Path the save figures
        figure_size(tuple): Size of the generated figures
    """

    # Compute confusion matrices

    cnf_matrix = confusion_matrix(target, preds)

    # Confusion matrix

    plt.figure(figsize=figure_size)
    plot_confusion_matrix(
        cnf_matrix, classes=labels, title="Confusion matrix, without normalization"
    )

    save_location = f"{figure_path}.png"
    logger.debug("Saving confusion matrix to %s", save_location)
    plt.savefig(save_location)

    # Confusion matrix  with normalization

    plt.figure(figsize=figure_size)
    plot_confusion_matrix(
        cnf_matrix, classes=labels, normalize=True, title="Normalized confusion matrix"
    )

    save_location = f"{figure_path}_normalized.png"
    logger.debug("Saving confusion matrix to %s", save_location)
    plt.savefig(save_location)


def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    printToFile=False,
):
    """
    FROM: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        if printToFile:
            print("Normalized confusion matrix")
    else:
        if printToFile:
            print("Confusion matrix, without normalization")

    if printToFile:
        print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def remove_padding_from_predictions(X, predictions, padding_type):
    """
    Remove padding from predictions

    Args:
        X(list): List of lists containing input tokens
        predictions(list): List of lists containing corresponding predictins.
        padding_type(list): Either `pre` or `post`, corresponds to whether the
            padding was added to the input tokens.
    """

    out = []

    for tokens, labels in zip(X, predictions):
        padding_len = len(labels) - len(tokens)

        if padding_type == "pre":
            out.append(labels[padding_len:])
        elif padding_type == "post":
            out.append(labels[:-padding_len])
        else:
            logger.error("padding_type must be one of ['pre', 'post']")

    return out
