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
    
    
    RemovedB = False
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
    frame_count = 0
    key_pressed = False

    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\jump\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\no_jump\\"), exist_ok=True)
    #open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt","w+")
    global removeFramesFile
    removeFramesFile = open(os.getcwd()+"\\datasets\\remove\\"+timestamp[:-1]+".txt","a+")
    
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
        #full_frame = game_frame.frame
        #image_data = full_frame[...,np.newaxis]
        #query_sprite = Sprite("QUERY",image_data = image_data)
        
        #sprite_name = self.identify_by_signature_colors(query_sprite)

        global RemovedB
        prediction = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        #if prediction == 1:
        #    print("Game over :(")
        #else:
        #    print("Still in it :)")
        global frame_count
        global timestamp
        
        
        if prediction != 1:
        #if sprite_name == 'UNKNOWN'
            RemovedB = False
            def save_game_frame(frame,frame_cnt):
            
                if not (key_pressed or old_key_pressed):
                    frame.save("datasets\\" + timestamp + "\\no_jump\\" + str(frame_cnt) + ".png")
                    print("Writing to no_jump")
                else:
                    frame.save("datasets\\" + timestamp + "\\jump\\" + str(frame_cnt) + ".png")
                    print("Writing to jump")
          
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
        else:
            #print('Gameover')
            if not RemovedB:
                RemovedB = True
                print(frame_count)
                def game_over(frame_cnt):
                    global removeFramesFile
                    removeFramesFile.write(str(frame_count)+"\n")
                
                thread.start_new_thread(game_over,(frame_count,))
            #ONLY FOR TESTING SHOULD BE REMOVED LATER
            #frame_count +=1       
            
    #def identify_by_signature_colors(self, query_sprite, score_threshold=25, debug=False):
    #    top_sprite_score = 0
    #    top_sprite_match = None
    #
    #    for sprite_name, sprite in self.game.sprites.items():
    #        for sprite_signature_colors in sprite.signature_colors:
    #            for query_sprite_signature_colors in query_sprite.signature_colors:
    #                signature_color_score = int((len(query_sprite_signature_colors & sprite_signature_colors) / len(sprite_signature_colors)) * 100)
    #
    #                if debug:
    #                    print(sprite_name, signature_color_score)
    #
    #                if signature_color_score > top_sprite_score:
    #                    top_sprite_score = signature_color_score
    #                    top_sprite_match = sprite_name
    #    return top_sprite_match if top_sprite_score >= score_threshold else "UNKNOWN"                
    