from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import _thread as thread
import numpy as np
import time
from PIL import Image
from datetime import datetime
import sys
sys.path.append(os.getcwd())
from ImageNetwork import ImageNetwork

class SerpentRecorderGameAgent(GameAgent):
    global timestamp
    global frame_count
    global key_pressed
    global RemovedB
    global removeFramesFilePath
    
    RemovedB = False
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
    frame_count = 0
    key_pressed = False

    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\jump\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\no_jump\\"), exist_ok=True)
    #open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt","w+")
    removeFramesFilePath = os.getcwd()+"\\datasets\\remove\\"+timestamp[:-1]+".txt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        

    def setup_play(self):
        context_classifier_path = f"datasets/context_classifier.model"

        context_classifier = ImageNetwork(
            input_shape=(60, 80, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

    def handle_play(self, game_frame):
        global RemovedB
        prediction = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        global frame_count
        global timestamp
        global key_pressed
        old_key_pressed = key_pressed
        key_pressed = keyboard.is_pressed('space')
        
        if prediction != 1:
            RemovedB = False
            def save_game_frame(frame,frame_cnt):
            
                if not (key_pressed or old_key_pressed):
                    frame.save("datasets\\" + timestamp + "\\no_jump\\" + str(frame_cnt) + ".png")
                    print("Writing to no_jump")
                else:
                    frame.save("datasets\\" + timestamp + "\\jump\\" + str(frame_cnt) + ".png")
                    print("Writing to jump")
          
            
        
            #Visual debugger
            for i, game_frame in enumerate(self.game_frame_buffer.frames):
                self.visual_debugger.store_image_data(
                    game_frame.grayscale_frame,
                    game_frame.grayscale_frame.shape,
                    str(i)
                )
        
            small_im = game_frame.eighth_resolution_frame
            gray_im = Image.fromarray(small_im).convert("L")
            thread.start_new_thread(save_game_frame,(gray_im,frame_count,))
            frame_count += 1
        else:
            print('Game Over')
            if not RemovedB:
                RemovedB = True
                print(frame_count)
                def game_over(frame_cnt):
                    global removeFramesFilePath
                    removeFramesFile = open(removeFramesFilePath,"a+")
                    removeFramesFile.write(str(frame_cnt)+"\n")
                
                thread.start_new_thread(game_over,(frame_count,))
            #ONLY FOR TESTING SHOULD BE REMOVED LATER
            #frame_count +=1       
             