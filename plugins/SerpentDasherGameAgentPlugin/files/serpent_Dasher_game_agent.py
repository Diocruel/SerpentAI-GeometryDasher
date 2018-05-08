from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
from PIL import Image
from datetime import datetime

class SerpentDasherGameAgent(GameAgent):
    global timestamp
    global frame_count
    global key_presses

    key_presses = []
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
    frame_count = 0

    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\jump\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\no_jump\\"), exist_ok=True)
    #open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt","w+")

    keyboard.hook_key('space',lambda k: key_presses.append("Space pressed at frame: "+ str(frame_count) + "\n"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        print("Frames and key_presses are being recorded")
        global timestamp
        global frame_count
        global key_presses
        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.grayscale_frame,
                game_frame.grayscale_frame.shape,
                str(i)
            )
        im = Image.fromarray(game_frame.frame)


        #f = open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt", "a+")
        #for j in key_presses:
        #    f.write(j)
        if not key_presses:
            im.save("datasets\\" + timestamp + "\\no_jump\\"+ str(frame_count) + ".png")
        else:
            im.save("datasets\\" + timestamp + "\\jump\\" + str(frame_count) + ".png")
        key_presses = []
        frame_count += 1
