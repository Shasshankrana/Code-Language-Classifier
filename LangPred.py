"""Machine learning model for programming identification"""

import os
import gc
import logging
from pathlib import Path
from math import ceil
import json

import numpy as np
import tensorflow as tf

from FeatureExtract import extract, CONTENT_SIZE

from Proccess import (search_files, extract_from_files, read_file)

# Settings list
# LOGGER = logging.getLogger(__name__)

_NEURAL_NETWORK_HIDDEN_LAYERS = [256, 64, 16]
_OPTIMIZER_STEP = 0.05

_FITTING_FACTOR = 20
_CHUNK_PROPORTION = 0.2
_CHUNK_SIZE = 1000

class Predictor:

    def __init__(self, model_dir=None):

        # trained model dir
        self.model_dir = os.curdir

        #: tells if current model is the default model
        #self.is_default = model_data[1]

        #: supported languages with associated extensions
        with open('languages.json') as f:
            self.languages = json.load(f)

        n_classes = len(self.languages)
        feature_columns = [
            tf.contrib.layers.real_valued_column('', dimension=CONTENT_SIZE)]

        self._classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=_NEURAL_NETWORK_HIDDEN_LAYERS,
            n_classes=n_classes,
            linear_optimizer=tf.train.RMSPropOptimizer(_OPTIMIZER_STEP),
            dnn_optimizer=tf.train.RMSPropOptimizer(_OPTIMIZER_STEP),
            model_dir=self.model_dir)

    def language(self, text):
        # predict language name
        values = extract(text)
        input_fn = _to_func([[values], []])
        proba = next(self._classifier.predict_proba(input_fn=input_fn))
        proba = proba.tolist()

        # Order the languages from the most probable to the least probable
        positions = np.argsort(proba)[::-1]
        names = np.sort(list(self.languages))
        names = names[positions]
        
        return names[0]

    def learn(self, input_dir):
        """Learning model"""

        languages = self.languages
        extensions = [ext for exts in languages.values() for ext in exts]
        print (extensions)
        files = search_files(input_dir, extensions)
        nb_files = len(files)
        chunk_size = min(int(_CHUNK_PROPORTION * nb_files), _CHUNK_SIZE)

        batches = _pop_many(files, chunk_size)

        evaluation_data = extract_from_files(next(batches), languages)

        accuracy = 0
        total = ceil(nb_files / chunk_size) - 1
        print("Start learning")
        for pos, training_files in enumerate(batches, 1):

            training_data = extract_from_files(training_files, languages)

            steps = int(_FITTING_FACTOR * len(training_data[0]) / 100)

            self._classifier.fit(input_fn=_to_func(training_data), steps=steps)

            # evaluation
            accuracy = self._classifier.evaluate(
                input_fn=_to_func(evaluation_data), steps=1)['accuracy']

        return accuracy

def _pop_many(items, chunk_size):
    while items:
        yield items[0:chunk_size]

        # Avoid memory overflow
        del items[0:chunk_size]
        gc.collect()

def _to_func(vector):
    return lambda: (
        tf.constant(vector[0], name='const_features'),
        tf.constant(vector[1], name='const_labels'))
