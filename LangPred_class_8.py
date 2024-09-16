import os
import gc
import json
from pathlib import Path
from math import ceil

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

from FeatureExtract import extract, CONTENT_SIZE
from Process import (search_files, extract_from_files)

_NEURAL_NETWORK_HIDDEN_LAYERS = [256, 64, 16]
_OPTIMIZER_STEP = 0.05

_FITTING_FACTOR = 20
_CHUNK_PROPORTION = 0.2
_CHUNK_SIZE = 1000


class Predictor:
    def __init__(self, model_dir='ckpt_model'):
        # Create the directory if it doesn't exist
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        # trained model dir
        self.model_dir = model_dir

        #: supported languages with associated extensions
        with open('languages.json') as f:
            self.languages = json.load(f)

        n_classes = len(self.languages)
        self.model = self._build_model(n_classes)


    def _build_model(self, n_classes):
        model = Sequential()
        model.add(
            Dense(_NEURAL_NETWORK_HIDDEN_LAYERS[0], activation='relu',
                  input_shape=(CONTENT_SIZE,)))
        for units in _NEURAL_NETWORK_HIDDEN_LAYERS[1:]:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(optimizer=RMSprop(_OPTIMIZER_STEP),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def language(self, text):
        # Predict language name
        values = extract(text)
        proba = self.model.predict(np.array([values]))[0]

        # Order the languages from the most probable to the least probable
        positions = np.argsort(proba)[::-1]
        names = np.sort(list(self.languages))
        names = names[positions]

        return names[0]

    def learn(self, input_dir):
        """Learning model"""

        languages = self.languages
        extensions = [ext for exts in languages.values() for ext in exts]
        print(extensions)
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
            X, y = training_data
            steps = int(_FITTING_FACTOR * len(X) / 100)
            self.model.fit(np.array(X), np.array(y), epochs=steps,
                           verbose=1)

            # evaluation
            eval_X, eval_y = evaluation_data
            loss, acc = self.model.evaluate(np.array(eval_X),
                                            np.array(eval_y))
            accuracy = acc

        return accuracy

    def save_model(self):
        """Save the model to the model directory"""
        self.model.save(os.path.join(self.model_dir, 'model.h5'))

    def load_model(self):
        """Load the model from the model directory"""
        self.model = tf.keras.models.load_model(
            os.path.join(self.model_dir, 'model.h5'))


def _pop_many(items, chunk_size):
    while items:
        yield items[0:chunk_size]
        # Avoid memory overflow
        del items[0:chunk_size]
        gc.collect()
