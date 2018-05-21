from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import numpy as np
import keyboard
import os
import sys
sys.path.append(os.getcwd())
from ImageNetwork import ImageNetwork
from PIL import Image

from time import time

class SerpentDasherGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):

        classifier_path = f"datasets/pretrained_classifier.model"

        classifier = ImageNetwork(
            input_shape=(60, 80, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        classifier.load_classifier(classifier_path)

        self.machine_learning_models["classifier"] = classifier

    def handle_play(self, game_frame):

        small_im = game_frame.eighth_resolution_frame
        start = time()
        prediction = self.machine_learning_models["classifier"].predict(small_im)
        end = time()
        print(prediction)
        if (prediction == 0) :
            self.input_controller.tap_key(KeyboardKey.KEY_UP)
        #print("time : " + str(end-start))