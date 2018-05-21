from serpent.machine_learning.context_classification.context_classifier import ContextClassifier
from serpent.utilities import SerpentError
import tensorflow as tf

try:
    from keras.preprocessing.image import ImageDataGenerator
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    from keras.layers import (Input, Dense, GlobalAveragePooling2D, Convolution2D,
                              BatchNormalization, Flatten, GlobalMaxPool2D, MaxPool2D,
                              concatenate, Activation)
    from keras.models import Model, load_model
    from keras.callbacks import ModelCheckpoint
    from keras.utils import Sequence, to_categorical
    from keras import backend as K
    from keras.callbacks import Callback



except ImportError:
    raise SerpentError("Setup has not been been performed for the ML module. Please run 'serpent setup ml'")

import skimage.transform

import serpent.cv
import numpy as np
import random
import os

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

class ImageNetwork(ContextClassifier):

    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape

        self.training_generator = None
        self.validation_generator = None

    def train(self, epochs=3, autosave=False, validate=True):
        if validate and (self.training_generator is None or self.validation_generator is None):
            self.prepare_generators()

        #Not sure what this input shoul be, Input() is not working
        #self.input_shape gives an error it should be a tensor.
        #serpent give some input stuff from exsisting model, not sure how this works

        inp = Input(shape=self.input_shape)
        #inp = self.input_shape
        x = Convolution2D(32, (8, 8), strides=4, padding="same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Convolution2D(64, (4, 4), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Convolution2D(64, (3, 3), strides=1, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)

        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        predictions = Dense(len(self.training_generator.class_indices), activation='softmax')(x)
        self.classifier = Model(inputs=inp, outputs=predictions)



       #This loads an existing model, thats why you need 3 channels. We want our own model
       # base_model = InceptionV3(
       #     weights="imagenet",
       #     include_top=False,
       #     input_shape=self.input_shape
       # )

        #output = base_model.output
        #output = GlobalAveragePooling2D()(output)
        #output = Dense(1024, activation='relu')(output)

        #predictions = Dense(len(self.training_generator.class_indices), activation='softmax')(output)
        #self.classifier = Model(inputs=base_model.input, outputs=predictions)

        #for layer in base_model.layers:
        #    layer.trainable = False

        self.classifier.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy", auc_roc]
        )

        callbacks = []

        if autosave:
            callbacks.append(ModelCheckpoint(
                "datasets/context_classifier_{epoch:02d}-{val_loss:.2f}.model",
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
            class_weight={0:100,1:1},
            callbacks=callbacks
        )

    def validate(self):
        pass

    def predict(self, input_frame):
        source_min = 0

        if str(input_frame.dtype) == "uint8":
            source_max = 255
        elif str(input_frame.dtype) == "float64":
            source_max = 1

        input_frame = np.array(serpent.cv.normalize(
            input_frame,
            source_min,
            source_max,
            target_min=-1,
            target_max=1
        ), dtype="float32")

        class_probabilities = self.classifier.predict(input_frame[None, :, :, :])[0]

        max_probability_index = np.argmax(class_probabilities)
        max_probability = class_probabilities[max_probability_index]

        if max_probability < 0.5:
            return None

        return max_probability_index


    def save_classifier(self, file_path):
        if self.classifier is not None:
            self.classifier.save(file_path)

    def load_classifier(self, file_path):
        self.classifier = load_model(file_path, custom_objects={'auc_roc': auc_roc})

    def prepare_generators(self):
        training_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.training_generator = training_data_generator.flow_from_directory(
            "datasets/current/training",
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32
        )

        self.validation_generator = validation_data_generator.flow_from_directory(
            "datasets/current/validation",
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32
        )



    def executable_train(epochs=3, autosave=False, classifier="ImageNetwork", validate=True):
        context_paths = list()

        for root, directories, files in os.walk("datasets/collect_frames_for_training".replace("/", os.sep)):
            if root != "datasets/collect_frames_for_training".replace("/", os.sep):
                break

            for directory in directories:
                context_paths.append(f"datasets/collect_frames_for_training/{directory}".replace("/", os.sep))

        if not len(context_paths):
            raise ContextClassifierError("No Context Frames found in 'datasets/collect_frames_for_datasets'...")

        serpent.datasets.create_training_and_validation_sets(context_paths)

        context_path = random.choice(context_paths)
        frame_path = None

        for root, directories, files in os.walk(context_path):
            for file in files:
                if file.endswith(".png"):
                    frame_path = f"{context_path}/{file}"
                    break
            if frame_path is not None:
                break

        frame = skimage.io.imread(frame_path)

        imagenetwork = ImageNetwork(input_shape=(60,80,3))
        imagenetwork.train(epochs=epochs, autosave=autosave, validate=validate)
        imagenetwork.validate()

        ImageNetwork.save_classifier(imagenetwork, "datasets/pretrained_classifier.model")
        print("Success! Model was saved to 'datasets/pretrained_classifier.model'")