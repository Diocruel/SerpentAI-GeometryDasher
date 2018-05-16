from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import sys
sys.path.append(os.getcwd())
from ImageNetwork import ImageNetwork

from time import time

class SerpentDasherGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):

        context_classifier_path = f"datasets/pretrained_classifier.model"

        context_classifier = ImageNetwork(
            input_shape=(60, 80, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):

        eightframe = game_frame.eighth_resolution_frame
        start = time()
        context = self.machine_learning_models["context_classifier"].predict(eightframe)
        end = time()
        if (context == 'jump') :
            self.input_controller.tap_key(KeyboardKey.KEY_UP)
        print(context)
        print("time : " + str(end-start))