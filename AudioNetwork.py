from serpent.machine_learning.context_classification.context_classifier import ContextClassifier

from serpent.utilities import SerpentError
import tensorflow as tf

try:
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    from keras.layers import (Activation, Convolution1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
    from keras.models import Model, load_model
    from keras.callbacks import ModelCheckpoint
    from keras.utils import Sequence, to_categorical
    from keras import backend as K



except ImportError:
    raise SerpentError("Setup has not been been performed for the ML module. Please run 'serpent setup ml'")

import skimage.transform

import serpent.cv
import numpy as np
import random
import os
import shutil
import IPython
import pandas as pd
# To load and read audio files
import librosa
SAMPLE_RATE = 44100
from Config import Config, DataGenerator
config = Config(sampling_rate=SAMPLE_RATE, audio_duration=2, use_mfcc=False)


def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

class ContextClassifierError(BaseException):
    pass

class AudioNetwork(ContextClassifier):

    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape

        self.training_generator = None
        self.validation_generator = None
	
    def train(self, epochs=3, autosave=False, validate=True):
        if validate and (self.training_generator is None or self.validation_generator is None):
            self.prepare_generators()


        inp = Input(shape=self.input_shape)
        x = Convolution1D(16, 9, activation='tanh', padding="valid")(inp)
        x = Convolution1D(16, 9, activation='tanh', padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)
        
        x = Convolution1D(32, 3, activation='tanh', padding="valid")(x)
        x = Convolution1D(32, 3, activation='tanh', padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)
    
        x = Convolution1D(32, 3, activation='tanh', padding="valid")(x)
        x = Convolution1D(32, 3, activation='tanh', padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)
		
        x = Convolution1D(256, 3, activation='tanh', padding="valid")(x)
        x = Convolution1D(256, 3, activation='tanh', padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(64, activation='tanh')(x)
        x = Dense(1028, activation='tanh')(x)
		
        predictions = Dense(2, activation='softmax')(x)
        self.classifier = Model(inputs=inp, outputs=predictions)

        self.classifier.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy",auc_roc]
        )

        callbacks = []

        if autosave:
            callbacks.append(ModelCheckpoint(
                "datasets/audio_classifier_{epoch:02d}-{val_loss:.2f}.model",
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='auto',
                period=1
            ))

        self.classifier.fit_generator(
            self.training_generator,
            samples_per_epoch=self.training_sample_count,
            nb_epoch=epochs,
            validation_data=self.validation_generator,
            nb_val_samples=self.validation_sample_count,
            class_weight={0: 4., 1: 1.},
            callbacks=callbacks
        )

    def validate(self):
        pass

    def predict(self, input_frame):

        def audio_norm(data):
            np.nan_to_num(data, copy=False)
            max_data = np.max(data)
            min_data = np.min(data)
            data = (data - min_data) / (max_data - min_data + 1e-6)
            return data - 0.5

        source_min = 0

        input_frame = np.array(serpent.cv.normalize(
            input_frame,
            source_min,
            source_max=1,
            target_min=-1,
            target_max=1
        ), dtype="float32")

        np.nan_to_num(input_frame,copy=False)
        class_probabilities = self.classifier.predict(input_frame[None, :, :])[0]
        print(class_probabilities)
        max_probability_index = np.argmax(class_probabilities)
        max_probability = class_probabilities[1]

        return max_probability

    def save_classifier(self, file_path):
        if self.classifier is not None:
            self.classifier.save(file_path)

    def load_classifier(self, file_path):
        self.classifier = load_model(file_path, custom_objects={'auc_roc': auc_roc})
	
    def prepare_generators(self):
        trainingLabels = []
        trainingIDs = []
        files = os.listdir('datasets/current/training/yes_jump/')
        for file in files:
            if file.endswith(".wav"):
                trainingLabels.append('yes_jump')
                trainingIDs.append('/yes_jump/' +file)
        files = os.listdir('datasets/current/training/no_jump/')
        for file in files:
            if file.endswith(".wav"):
                trainingLabels.append('no_jump')
                trainingIDs.append('/no_jump/' +file)
                    
        validationLabels = []
        ValidtionIDS = []
        files = os.listdir('datasets/current/validation/yes_jump/')
        for file in files:
            if file.endswith(".wav"):
                validationLabels.append('yes_jump')
                ValidtionIDS.append('/yes_jump/' +file)
        files = os.listdir('datasets/current/validation/no_jump/')
        for file in files:
            if file.endswith(".wav"):
                validationLabels.append('no_jump')
                ValidtionIDS.append('/no_jump/' + file)
        	
        print(trainingIDs);		
        def audio_norm(data):
            np.nan_to_num(data, copy=False)
            max_data = np.max(data)
            min_data = np.min(data)
            data = (data-min_data)/(max_data-min_data+1e-6)
            return data-0.5
	
        self.training_generator = DataGenerator(config, 'datasets/current/training', trainingIDs, 
		                            trainingLabels, batch_size=32, preprocessing_fn=audio_norm)
        self.validation_generator = DataGenerator(config, 'datasets/current/validation', ValidtionIDS, 
                                    validationLabels, batch_size=32, preprocessing_fn=audio_norm)

    def executable_train(epochs=3, autosave=False, classifier="AudioNetwork", validate=True):
        context_paths = list()

        for root, directories, files in os.walk("datasets/audio/collect_frames_for_training".replace("/", os.sep)):
            if root != "datasets/audio/collect_frames_for_training".replace("/", os.sep):
                break

            for directory in directories:
                context_paths.append(f"datasets/audio/collect_frames_for_training/{directory}".replace("/", os.sep))

        if not len(context_paths):
            raise ContextClassifierError("No Context Frames found in 'datasets/audio/collect_frames_for_training'...")

        serpent.datasets.create_training_and_validation_sets(context_paths)

        context_path = random.choice(context_paths)
        frame_path = None

        for root, directories, files in os.walk(context_path):
            for file in files:
                if file.endswith(".wav"):
                    frame_path = f"{context_path}/{file}"
                    break
            if frame_path is not None:
                break

        frame, _ = librosa.core.load(frame_path, sr=SAMPLE_RATE)
        np.nan_to_num(frame,copy=False)
        frame.shape
	
        audionetwork = AudioNetwork(input_shape=(config.audio_length, 1))
        audionetwork.train(epochs=epochs, autosave=autosave, validate=validate)
        audionetwork.validate()

        AudioNetwork.save_classifier(audionetwork, "datasets/pretrained_audio_classifier.model")
        print("Success! Model was saved to 'datasets/pretrained_audio_classifier.model'")

        # check this:
        #(X,y) = self.training_generator[0]
