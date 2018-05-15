from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import _thread as thread
import numpy as np
import time
from PIL import Image
from datetime import datetime

class SerpentDasherGameAgent(GameAgent):
    global timestamp
    global frame_count
    global key_pressed

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
    frame_count = 0
    key_pressed = False

    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\jump\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\no_jump\\"), exist_ok=True)
    #open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt","w+")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        pass

    def handle_play(self, game_frame):

        def save_game_frame(frame,frame_cnt):
            if not (key_pressed or old_key_pressed):
                frame.save("datasets\\" + timestamp + "\\no_jump\\" + str(frame_cnt) + ".png")
                print("Writing to no_jump")
            else:
                frame.save("datasets\\" + timestamp + "\\jump\\" + str(frame_cnt) + ".png")
                print("Writing to jump")

        global timestamp
        global frame_count
        global key_pressed

        #Visual debugger
        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.grayscale_frame,
                game_frame.grayscale_frame.shape,
                str(i)
            )

        small_im = game_frame.eighth_resolution_frame
        gray_im = Image.fromarray(small_im).convert("L")
        old_key_pressed = key_pressed
        key_pressed = keyboard.is_pressed('space')
        thread.start_new_thread(save_game_frame,(gray_im,frame_count,))
        frame_count += 1
