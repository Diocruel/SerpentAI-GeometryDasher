from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import sys
sys.path.append(os.getcwd())
from ImageNetwork import ImageNetwork

from time import time

from serpent.sprite import Sprite
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
        print('Space is spressed: ',str(keyboard.is_pressed('space')))

        # for i, game_frame in enumerate(self.game_frame_buffer.frames):
        #     self.visual_debugger.store_image_data(
        #         game_frame.frame,
        #         game_frame.frame.shape,
        #         str(i)
        #     )
        self.input_controller.tap_key(KeyboardKey.KEY_UP)
