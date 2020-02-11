#!/usr/bin/env python3
# coding: utf-8

"""
Based on

https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing
"""

import csv
import itertools
import os

import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, Bidirectional, Convolution1D, Dense, Dropout,
                          Embedding, Flatten, Input, MaxPooling1D,
                          TimeDistributed, concatenate)
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from sklearn_crfsuite import metrics

from deep_reference_parser.logger import logger
from deep_reference_parser.model_utils import (Classification_Scores, character_data,
                          character_index, encode_x, encode_y, index_x,
                          index_y, merge_digits,
                          remove_padding_from_predictions,
                          save_confusion_matrix, word2vec_embeddings)
from .reference_utils import load_tsv, read_pickle, write_pickle, write_to_csv


class DeepReferenceParser:
    """
    A Recurrent Neural Network for refrence finding

    Build, train, evaluate, and predict from a bi-direction LSTM with stacked
    CRF, based heavily on the model described by Rodrigues et al, the code for
    which is available here:

    https://github.com/dhlab-epfl/LinkedBooksDeepReferenceParsing

    This class, and the accompany utility functions in model_utils.py take the
    Rodridgues model, and wrap it in a more production friendly class.
    """

    def __init__(self, X_train=None, X_test=None, X_valid=None,
        y_train=None, y_test=None, y_valid=None, digits_word="$NUM$",
        ukn_words="out-of-vocabulary", padding_style="pre",
        output_path="data/model_output"):
        """
        Note that in the terminology used by Rodrigues, the development set is
        referred to as the test set, and the true test (hold out) set is
        referred to as the validation set.

        Args:
            X_train(str): List of lists containing training tokens.
            X_test(str): List of lists containing testing tokens.
            X_valid(str): List of lists containing validation tokens.
            y_train(str): List of lists containing training labels.
            y_test(str): List of lists containing testing labels.
            y_valid(str): List of lists containing validation labels.
            digits_word(str): Token to be used to replace digits.
            ukn_words(str): Token used to replace words not in the dictionary.
            padding_style(str): One of `"pre"` or `"post"`. Determines whether
                padding of sentences is added to the beginning or end of each
                sentence.
            output_path(str): Where to save model outputs.

        """

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test

        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

        self.X_train_merged = list()
        self.X_valid_merged = list()
        self.X_test_merged = list()

        self.X_train_encoded = list()
        self.X_valid_encoded = list()
        self.X_test_encoded = list()

        self.y_train_encoded = list()
        self.y_valid_encoded = list()
        self.y_test_encoded = list()

        self.X_train_char = list()
        self.X_valid_char = list()
        self.X_test_char = list()

        self.X_training = list()
        self.X_validation = list()
        self.X_testing = list()

        self.max_len = int()
        self.max_char = int()
        self.max_words = int()

        # Defined in prepare_data

        self.word2ind = {}
        self.ind2word = {}
        self.label2ind = {}
        self.ind2label = []

        # Defined in self.build_model()

        self.model = None
        self.digits_word = digits_word
        self.ukn_words = ukn_words
        self.padding_style = padding_style

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.weights_path = os.path.join(output_path, "weights.h5")


    def prepare_data(self, save=False):
        """
        Prepare data for training a model

        Args:
            Save(bool): If True, then data objects will be saved to
                `self.output_path`.
        """
        self.max_len = max([len(xx) for xx in self.X_train])

        self.X_train_merged, self.X_test_merged, self.X_valid_merged = merge_digits(
            [self.X_train, self.X_test, self.X_valid],
            self.digits_word
        )

        # Compute indexes for words+labels in the training data

        self.word2ind, self.ind2word = index_x(self.X_train_merged, self.ukn_words)
        self.label2ind, ind2label = index_y(self.y_train)

        # NOTE: The original code expected self.ind2label to be a list,
        # in case you are training a multi-task model. For this reason,
        # self.index2label is wrapped in a list.

        self.ind2label.append(ind2label)

        # Convert data into indexes data

        # Encode X variables

        # TODO: Save out the encoded data for loading in the train_model method

        self.X_train_encoded = encode_x(self.X_train_merged, self.word2ind,
            self.max_len, self.ukn_words, self.padding_style)

        self.X_test_encoded = encode_x(self.X_test_merged, self.word2ind,
            self.max_len, self.ukn_words, self.padding_style)

        self.X_valid_encoded = encode_x(self.X_valid_merged, self.word2ind,
            self.max_len, self.ukn_words, self.padding_style)


        logger.debug("Training set dimensions: %s", self.X_train_encoded.shape)
        logger.debug("Test set dimensions: %s", self.X_test_encoded.shape)
        logger.debug("Validation set dimensions: %s", self.X_valid_encoded.shape)

        # Encode y variables

        self.y_train_encoded = encode_y(self.y_train, self.label2ind,
            self.max_len, self.padding_style)

        self.y_test_encoded = encode_y(self.y_test, self.label2ind,
            self.max_len, self.padding_style)

        self.y_valid_encoded = encode_y(self.y_valid, self.label2ind,
            self.max_len, self.padding_style)

        logger.debug("Training target dimensions: %s", self.y_train_encoded.shape)
        logger.debug("Test target dimensions: %s", self.y_test_encoded.shape)
        logger.debug("Validation target dimensions: %s", self.y_valid_encoded.shape)

        # Create character level data

        # Create the character level data
        self.char2ind, self.max_words, self.max_char = character_index(
            self.X_train, self.digits_word
        )

        self.X_train_char = character_data(self.X_train, self.char2ind,
            self.max_words, self.max_char, self.digits_word, self.padding_style)

        self.X_test_char = character_data(self.X_test, self.char2ind,
            self.max_words, self.max_char, self.digits_word, self.padding_style)

        self.X_valid_char = character_data(self.X_valid, self.char2ind,
            self.max_words, self.max_char, self.digits_word, self.padding_style)

        self.X_training = [self.X_train_encoded, self.X_train_char]
        self.X_testing = [self.X_test_encoded, self.X_test_char]
        self.X_validation = [self.X_valid_encoded, self.X_valid_char]

        if save:

            # Save intermediate objects to data

            write_pickle(self.word2ind, "word2ind.pickle", path=self.output_path)
            write_pickle(self.ind2word, "ind2word.pickle", path=self.output_path)
            write_pickle(self.label2ind, "label2ind.pickle", path=self.output_path)
            write_pickle(self.ind2label, "ind2label.pickle", path=self.output_path)
            write_pickle(self.char2ind, "char2ind.pickle", path=self.output_path)

            maxes = {
                "max_words": self.max_words,
                "max_char": self.max_char,
                "max_len": self.max_len
            }

            write_pickle(maxes, "maxes.pickle", path=self.output_path)

    def load_data(self, out_path):
        """
        Loads the intermediate model objects created that are created and saved
        out by prepare_data. But not the data used to train the model.

        NOTE: This method is not yet fully tested.
        """

        self.word2ind = read_pickle("word2ind.pickle", path=out_path)
        self.ind2word = read_pickle("ind2word.pickle", path=out_path)
        self.label2ind = read_pickle("label2ind.pickle", path=out_path)
        self.ind2label = read_pickle("ind2label.pickle", path=out_path)
        self.char2ind = read_pickle("char2ind.pickle", path=out_path)

        maxes = read_pickle("maxes.pickle", path=out_path)

        self.max_len = maxes["max_len"]
        self.max_char = maxes["max_char"]
        self.max_words = maxes["max_words"]

        logger.debug("Setting max_len to %s", self.max_len)
        logger.debug("Setting max_char to %s", self.max_char)
        logger.debug("Setting max_words to %s", self.max_words)


    def build_model(self, output="crf", word_embeddings=None,
            pretrained_embedding="", word_embedding_size=100,
            char_embedding_type="BILSTM", char_embedding_size=50, dropout=0,
            lstm_hidden=32, optimizer="rmsprop"):
        """
        Build the bilstm for use in the training and predict methods.

        Args:
            output(str): One of `"crf"` or `"softmax"`. Sets the prediction
                layer of the model.
            word_embeddings(str): Path to word embeddings. If `None`, then word
                embeddings will not be used.
            pretrained_embedding(str): One of: `["", True, False]`
                by the model? Setting to `""` will not use any pre-embedding.
                Setting to `True` will use a pre-trained embedding to
                initialise the weights of an embedding layer which will be
                trained. Setting to `False` will use the supplied pre-trained
                embedding without any additional training.
            word_embedding_size(int): The size of the pre-trained word embedding
                to use. One of `[100, 300]`.
            char_embedding_type(str): Which architecture to use for the
                character embedding. One of `["CNN", "BILSTM"]`.
            char_embedding_size(int): Size of the character-level word
                representations.
            dropout(float): Degree of dropout to use, set to between `0` and
                `1`.
            lstm_hidden(int): Dimensionality of the hidden layer.
            optimizer(str): Which optimizer to use. One of
                `["adam", "rmsprop"]`.
        """

        inputs = []
        embeddings_list = []
        nbr_words = len(self.word2ind) + 1
        nbr_chars = len(self.char2ind) + 1
        out_size = len(self.ind2label) + 1

        if word_embeddings:

            word_input = Input((self.max_words,))
            inputs.append(word_input)

            # TODO: More sensible handling of options for pretrained embedding.
            # Currently it uses one of `["", True, False]` where:
            # "": Do not use pre-trained embedding
            # False: Use pre-trained weights in the embedding layer
            # True: Use the pre-trained weights as weight initialisers, and
            # train the embedding layer.

            if pretrained_embedding == "":

                word_embedding = Embedding(nbr_words, word_embedding_size)(word_input)

            else:

                embedding_matrix = word2vec_embeddings(
                    embedding_path=word_embeddings, word2ind=self.word2ind,
                    word_embedding_size=word_embedding_size
                )

                word_embedding = Embedding(
                    nbr_words, word_embedding_size, weights=[embedding_matrix],
                    trainable=pretrained_embedding, mask_zero=False)(word_input)

            embeddings_list.append(word_embedding)

        # Input - Characters Embeddings

        if self.max_char != 0:

            character_input = Input((self.max_words, self.max_char,))

            char_embedding = self.character_embedding_layer(
                char_embedding_type=char_embedding_type,
                character_input=character_input, nbr_chars=nbr_chars,
                char_embedding_size=char_embedding_size
            )

            embeddings_list.append(char_embedding)
            inputs.append(character_input)

        # Model - Inner Layers - BiLSTM with Dropout

        if len(embeddings_list) == 2:

            embeddings = concatenate(embeddings_list)

        else:

            embeddings = embeddings_list[0]

        model = Dropout(dropout)(embeddings)
        model = Bidirectional(
            LSTM(lstm_hidden, return_sequences=True, dropout=dropout)
        )(model)
        model = Dropout(dropout)(model)

        if output == "crf":

            # Output - CRF

            crfs = [[CRF(out_size),out_size] for out_size in [len(x)+1 for x in self.ind2label]]
            outputs = [x[0](Dense(x[1])(model)) for x in crfs]
            model_loss = [x[0].loss_function for x in crfs]
            model_metrics = [x[0].viterbi_acc for x in crfs]

        if output == "softmax":

            outputs = [Dense(out_size, activation='softmax')(model) for out_size in [len(x)+1 for x in self.ind2label]]
            model_loss = ['categorical_crossentropy' for x in outputs]
            model_metrics = None

        # Model

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=model_loss, metrics=model_metrics,
            optimizer=self.get_optimizer(optimizer)
        )

        # NOTE: It's not necessary to save the model as it needs to be recreated
        # each time using build_model()

        #model_json = model.to_json()
        #save_path = os.path.join(self.output_path, "model.json")
        #with open(save_path, "w") as json_file:
        #    json_file.write(model_json)

        self.model = model

        logger.debug(self.model.summary(line_length=150))


    def train_model(self, epochs=25, batch_size=100, early_stopping_patience=5,
            metric="val_f1"):
        """
        Train a model that has been built or loaded (loading is not currently
        supported).

        Args:
            epochs(int): Number of epochs to train the model for.
            batch_size(int): Size of batches when training the model.
            early_stopping_patience(int): Number of continuous epochs to
                tolerate without improvement before stopping the model.
            metric(str): Which metric to monitor for early stopping.
        """

        # Save the model to json format

        # Training Callbacks:

        callbacks = []

        # Use custom classification scores callback

        # NOTE: X lists are important for input here

        classification_scores = Classification_Scores(
            [self.X_training, [self.y_train_encoded]], self.ind2label,
            self.weights_path
        )

        callbacks.append(classification_scores)

        # EarlyStopping

        if early_stopping_patience:

            early_stopping = EarlyStopping(
                monitor=metric, patience=early_stopping_patience,
                mode='max'
            )
            callbacks.append(early_stopping)

        # Train the model. Keras's method argument 'validation_data' is
        # referred as 'testing data' in this code.

        hist = self.model.fit(
            x=self.X_training, y=[self.y_train_encoded],
            validation_data=[self.X_testing, [self.y_test_encoded]],
            epochs=epochs, batch_size=batch_size, callbacks=callbacks,
            verbose=1
        )

        logger.info(
            "Best F1 score: %s (epoch number %s)",
            early_stopping.best,
            1 + np.argmax(hist.history[metric])
        )

        # Save Training scores

        scores_file = os.path.join(self.output_path, "scores.csv")

        self.save_model_training_scores(scores_file, hist, classification_scores)

        # Print best testing classification report

        best_epoch = np.argmax(hist.history[metric])

        logger.info("Best model epoch:\n%s", classification_scores.test_report[best_epoch])

        # Best epoch results

        self.best_results = self.model_best_scores(classification_scores, best_epoch)


    def evaluate(self, load_weights=False, test_set=False, validation_set=False,
        print_padding=False, out_file=None):
        """
        Evaluate model results

        Args:
            load_weights(bool): Load model weights from disk?
            test_set(bool): Whether to test the model against the test set.
            validation_set(bool): Whether to test the model against the
                validation set.
            print_padding(bool): Should the confusion matrix include the
                the prediction of padding characters?
            out_file(str): File into which the predictions and targets will be
                saved. Defaults to `None` which saves nothing if not set.
        """

        if load_weights:

            self.load_weights()

        # Create confusion matrices

        if test_set:

            # NOTE: self.y_valid_encoded goes in a list here, as it would
            # under a multi-task scenario. This will need adjusting when
            # using this syntax for a multi-task model.

            for i, y_target in enumerate([self.y_test_encoded]):

                # Compute predictions, flatten

                predictions, target = self.compute_predictions(
                    X=self.X_testing,
                    y=y_target,
                    labels=self.ind2label[i],
                )

                flat_predictions = np.ravel(predictions)
                flat_target = np.ravel(target)

                # Generate confusion matrices

                labels = list(self.ind2label[i].values())
                figure_path = os.path.join(self.output_path,
                    f"confusion_matrix_test_{i + 1}")

                save_confusion_matrix(
                    target=flat_target,
                    preds=flat_predictions,
                    labels=labels,
                    figure_path=figure_path
                )

       # Validation dataset

        if validation_set:

       # Compute classification report

            # NOTE: self.y_valid_encoded goes in a list here, as it would
            # under a multi-task scenario. This will need adjusting when
            # using this syntax for a multi-task model.

            for i, y_target in enumerate([self.y_valid_encoded]):

               # Compute predictions, flatten

                predictions, target = self.compute_predictions(
                    X=self.X_validation,
                    y=y_target,
                    labels=self.ind2label[i],
                    nbrTask=i,
                )

                flat_predictions = np.ravel(predictions)
                flat_target = np.ravel(target)

                # Only for multi-task

                labels = list(self.ind2label[i].values())
                # NOTE: Expects y_train to be a list for multi-task

                if len(self.y_train_encoded) > 1:

                    logger.info("Metrics not taking padding into account:")
                    logger.info("\n%s",
                                metrics.flat_classification_report(
                                    [flat_target], [flat_predictions], digits=4,
                                    labels=list(self.ind2label[i].values()))
                                )

                    if print_padding:
                        logger.info("Metrics taking padding into account:")
                        logger.info("\n%s",
                                    metrics.flat_classification_report(
                                        [flat_target], [flat_predictions], digits=4)
                                    )

                        # Generate confusion matrices

                    figure_path = os.path.join(self.output_path,
                        f"confusion_matrix_validation_{i + 1}")

                    save_confusion_matrix(
                        target=flat_target,
                        preds=flat_predictions,
                        labels=labels,
                        figure_path=figure_path
                    )

                    if out_file:

                        tokens = list(itertools.chain.from_iterable(self.X_valid))

                        # Strip out the padding

                        target_len = np.mean([len(line) for line in target])
                        prediction_len = np.mean([len(line) for line in predictions])

                        # Strip out the nulls from the target

                        clean_target = [
                            [label for label in line if label != "null"]
                            for line in target
                        ]

                        # Strip out the nulls in the predictions that match the
                        # nulls in the target

                        clean_predictions = remove_padding_from_predictions(
                            clean_target,
                            predictions,
                            self.padding_style
                        )

                        # Record any token length mismatches.

                        num_mismatches = len(clean_target) - np.sum([len(x) == len(y) for x, y in zip(clean_target, clean_predictions)])

                        logger.info("Number of mismatches: %s", num_mismatches)

                        # Flatten the target and predicted into one list.

                        clean_target = list(itertools.chain.from_iterable(clean_target))
                        clean_predictions = list(itertools.chain.from_iterable(clean_predictions))
                        # NOTE: this needs some attention. The current outputs
                        # seem to have different lengths and will therefore be
                        # offset unequally. - Don't trust them!

                        logger.info("tokens: %s", len(tokens))
                        logger.info("target: %s", len(clean_target))
                        logger.info("predictions: %s", len(clean_predictions))

                        out = list(zip(tokens, clean_target, clean_predictions))

                        out_file_path = os.path.join(self.output_path, out_file)

                        logger.info("Writing results to %s", out_file_path)

                        with open(out_file_path, "w") as fb:
                            writer = csv.writer(fb, delimiter="\t")

                            for i in out:
                                writer.writerow(i)

    def character_embedding_layer(self, char_embedding_type, character_input,
        nbr_chars, char_embedding_size, cnn_kernel_size=2, cnn_filters=30,
        lstm_units=50):
        """
        Return layer for computing the character-level representations of words.

        There is two type of architectures:

            Architecture CNN:
                - Character Embeddings
                - Flatten
                - Convolution
                - MaxPool

            Architecture BILSTM:
                - Character Embeddings
                - Flatten
                - Bidirectional LSTM

        Args:
            char_embedding_type(str): Model architecture to use "CNN" or
                "BILSTM".
            character_input(int): Keras Input layer, size of the input.
            nbr_chars(int): Numbers of unique characters present in the data.
            char_embedding_size(int): size of the character-level word
                representations.
            cnn_kernel_size(int): For the CNN architecture, size of the kernel
                in the Convolution layer
            cnn_filters: For the CNN architecture, number of filters in the
                Convolution layer.
            lstm_units: For the BILSTM architecture, dimensionality of the
                output LSTM space (half of the Bidirectinal LSTM output space).

        Returns:
            Character-level representation layers.
        """
        embed_char_out = TimeDistributed(
            Embedding(nbr_chars, char_embedding_size),
            name='char_embedding'
        )(character_input)

        embed_char = TimeDistributed(Flatten())(embed_char_out)

        if char_embedding_type == "CNN":
            conv1d_out = TimeDistributed(
                Convolution1D(kernel_size=cnn_kernel_size,
                              filters=cnn_filters,
                              padding='same')
            )(embed_char)

            char_emb = TimeDistributed(
                MaxPooling1D(self.max_char)
            )(conv1d_out)

        if char_embedding_type == "BILSTM":
            char_emb = Bidirectional(
                LSTM(lstm_units, return_sequences=True)
            )(embed_char)

        return char_emb

    def get_optimizer(self, optimizer, learning_rate=0.001, decay=0.0):
        """
        Return the optimizer needeed to compile Keras models.

        Args:
            optimizer_type(str): Type of optimizer. Two types supported:
                'Adam' and 'RMSprop'.
            learning_rate(float): float >= 0. Learning rate.
            decay(float):float >= 0. Learning rate decay over each update

        Returns:
            The optimizer to use directly into keras model compiling function.
        """

        if optimizer == "adam":
            return Adam(lr=learning_rate, decay=decay)

        if optimizer == "rmsprop":
            return RMSprop(lr=learning_rate, decay=decay)


    def save_model_training_scores(self, filename, hist, classification_scores):
        """
        Create a .csv file containg the model training metrics for each epoch

        Args:
            filename(str): Path and name of the .csv file without csv extension
            hist: Default model training history returned by Keras
            classification_scores: Classification_Scores instance used as
                callback in the model's training.
        """
        csv_values = []

        csv_columns = ["Epoch", "Training Accuracy", "Training Recall",
                       "Training F1", "Testing Accuracy", "Testing Recall",
                       "Testing F1"]

        # Epoch column

        csv_values.append(hist.epoch)

        # Training metrics

        csv_values.append(classification_scores.train_acc)
        csv_values.append(classification_scores.train_recall)
        csv_values.append(classification_scores.train_f1s)

        # Testing metrics

        csv_values.append(classification_scores.test_acc)
        csv_values.append(classification_scores.test_recall)
        csv_values.append(classification_scores.test_f1s)

        # Create file

        write_to_csv(filename, csv_columns, zip(*csv_values))

    def model_best_scores(self, classification_scores, best_epoch):
        """
        Return the metrics from best epoch

        Return list has the format:

        ```
        [
            "Best epoch", "Training Accuracy", "Training Recall",
            "Training F1", "Testing Accuracy", "Testing Recall",
            "Testing F1"
        ]
        ```

        Args:
            classification_scores: Classification_Scores instance used as
                callback in the model's training.
        best_epoch(int): Best training epoch index

        Returns:
            list: Best epoch training metrics in format:
        """

        best_values = []
        best_values.append(1 + best_epoch)

        best_values.append(classification_scores.train_acc[best_epoch])
        best_values.append(classification_scores.train_recall[best_epoch])
        best_values.append(classification_scores.train_f1s[best_epoch])

        best_values.append(classification_scores.test_acc[best_epoch])
        best_values.append(classification_scores.test_recall[best_epoch])
        best_values.append(classification_scores.test_f1s[best_epoch])

        return best_values

    def compute_predictions(self, X, y, labels, nbrTask=-1):
        """
        Compute the predictions and ground truth

        Args:
            X: Data
            y: Ground truth
            nbrTask: Relates to which task is in use

        Returns:
            The predictions and ground truth ready to be compared, flatten
            (1-d array).
        """

        # Compute training score

        pred = self.model.predict(X)

        # For multi-task

        if len(self.model.outputs) > 1:
            pred = pred[nbrTask]

        pred = np.asarray(pred)

        # Compute validation score
        pred_index = np.argmax(pred, axis=-1)

        # Reverse the one-hot encoding
        true_index = np.argmax(y, axis=-1)

        # Index 0 in the predictions refers to padding

        # TODO: `{0: "null"}` ... 🙄

        new_labels = labels.copy()
        new_labels.update({0: "null"})

        # Compute the labels for each prediction

        pred_label = [[new_labels[x] for x in a] for a in pred_index]
        true_label = [[new_labels[x] for x in b] for b in true_index]

        return pred_label, true_label


    def prepare_X_data(self, X):
        """
        Convert data to encoded word and character indexes

        TODO: Create a more generic function that can also be used in
        `self.prepare_data()`.

        Expects data in the following format:

        ```
        [
            ["word", "word", "word", "word"],
            ["word", "word", "word", "word"],
            ["word", "word", "word", "word"],
        ]

        ```

        Args:
            X(list): Data to be encoded.

        Returns:
            list: List containing two elements: A version of X where the digits
                have been converted to `self.digits_words`, and a second list
                containing the word and character embeddings.
        """

        X_merged = merge_digits([X], self.digits_word)[0]

        # Encode X variables

        X_encoded = encode_x(
            X_merged,
            self.word2ind,
            self.max_len,
            self.ukn_words,
            self.padding_style,
        )


        X_char = character_data(
            X,
            self.char2ind,
            self.max_words,
            self.max_char,
            self.digits_word,
            self.padding_style
        )

        X = [X_encoded, X_char]

        return X_merged, [X_encoded, X_char]


    def load_weights(self):
        """
        Load and initialise weights
        """

        if not self.model:

        # Assumes that model has been buit with build_model!

            logger.exception(
                "No model. you must build the model first with build_model"
            )

        # NOTE: This is not required if incldue_optimizer is set to false in
        # load_all_weights.

        # Run the model for one epoch to initialise network weights. Then load
        # full trained weights

        #self.model.fit(x=self.X_testing, y=self.y_test_encoded,
        #    batch_size=2500, epochs=1)

        logger.debug("Loading weights from %s", self.weights_path)

        save_load_utils.load_all_weights(self.model, self.weights_path,
            include_optimizer=False)


    def predict(self, X, load_weights=False):
        """
        Make predictions using a trained model
g        Expects data in the following format:

        ```
        [
            ["word", "word", "word", "word"],
            ["word", "word", "word", "word"],
            ["word", "word", "word", "word"],
        ]

        ```

        Args:
            X(list): Data to be encoded.
            load_weights(bool): Should the weights be loaded from disk?
            out_file(str): File into which the predictions and targets will be
                saved. Defaults to `None` which saves nothing if not set.

        Returns:
            list: A list of token predictions.
        """

        if load_weights:

            self.load_weights()

        _, X_combined = self.prepare_X_data(X)

        pred = self.model.predict(X_combined)

        pred = np.asarray(pred)

        # Compute validation score

        pred_index = np.argmax(pred, axis=-1)

        # NOTE: indexing ind2label[0] will only work in the case of making
        # predictions with a single task model.

        ind2labelNew = self.ind2label[0].copy()

        # Index 0 in the predictions refers to padding

        ind2labelNew.update({0: "null"})

        # Compute the labels for each prediction
        pred_label = [[ind2labelNew[x] for x in a] for a in pred_index]

        # Flatten data

        # Remove the padded tokens. This is done by counting the number of
        # tokens in the input example, and then removing the additional padded
        # tokens that are added before this. It has to be done this way because
        # the model can predict padding tokens, and sometimes it gets it wrong
        # so if we remove all padding tokens, then we end up with mismatches in
        # the length of input tokens and the length of predictions.

        out = remove_padding_from_predictions(X, pred_label, self.padding_style)

        return out
