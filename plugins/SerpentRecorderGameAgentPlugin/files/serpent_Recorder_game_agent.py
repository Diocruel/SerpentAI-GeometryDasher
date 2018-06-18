from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import _thread as thread
import numpy as np
import time
import multiprocessing
import pyaudio
import wave
from PIL import Image
from datetime import datetime
import sys
sys.path.append(os.getcwd())
from ImageNetwork import ImageNetwork

global FORMAT
global defaultframes
global recordtime

recordtime = 10
FORMAT = pyaudio.paInt16
defaultframes = 512


def record():
    global FORMAT
    global defaultframes
    global recordtime

    device_info = {}
    useloopback = False

    # Use module
    p = pyaudio.PyAudio()

    # Get input or default
    device_id = 4
    print("")

    # Get device info
    try:
        device_info = p.get_device_info_by_index(device_id)
    except IOError:
        print("Couldn't use audio device, please set correct device_id")
        exit()

    # Choose between loopback or standard mode
    is_input = device_info["maxInputChannels"] > 0
    is_wasapi = (p.get_host_api_info_by_index(device_info["hostApi"])["name"]).find("WASAPI") != -1
    if is_input:
        print("Selection is input using standard mode.\n")
    else:
        if is_wasapi:
            useloopback = True
            print("Selection is output. Using loopback mode.\n")
        else:
            print("Selection is output and does not support loopback mode. Quitting.\n")
            exit()

    # recordtime = int(input("Record time in seconds [" + str(recordtime) +"]: ") or recordtime)

    # Open stream
    channelcount = device_info["maxInputChannels"] if (
        device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
    stream = p.open(format=pyaudio.paInt16,
                    channels=channelcount,
                    rate=int(device_info["defaultSampleRate"]),
                    input=True,
                    frames_per_buffer=defaultframes,
                    input_device_index=device_info["index"],
                    as_loopback=useloopback)

    mydir = os.getcwd()
    os.makedirs(os.path.dirname(mydir + "\\audio\\"), exist_ok=True)
    os.makedirs(os.path.dirname(mydir + "\\audio\\raw\\"), exist_ok=True)
    subdir = "audio\\raw"

    print("recording...")
    while not keyboard.is_pressed('q'):
        frames = []
        stream.start_stream()
        start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        for i in range(0, int(int(device_info["defaultSampleRate"]) / defaultframes * recordtime)):
            frames.append(stream.read(defaultframes))

        stream.stop_stream()
        filename = "%s.wav" % start_time

        wavefilepath = os.path.join(mydir, subdir, filename)
        waveFile = wave.open(wavefilepath, 'wb')
        waveFile.setnchannels(channelcount)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(int(device_info["defaultSampleRate"]))
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
    print("finished recording")

    # stop Recording
    stream.close()
    p.terminate()

class SerpentRecorderGameAgent(GameAgent):
    global frame_count
    global key_pressed
    global audio_file
    global audio_thread
    global RemovedB
    
    RemovedB = False
    
    frame_count = 0
    key_pressed = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        

    def setup_play(self):

        global timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
        

        
        
        global removeFramesFilePath
        
        os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\"), exist_ok=True)
        os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp))
        os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\jump\\"))
        os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\" + timestamp + "\\no_jump\\"))
        #open(os.getcwd() + "\\datasets\\" + timestamp + "presses.txt","w+")
        removeFramesFilePath = os.getcwd()+"\\datasets\\remove\\"+timestamp[:-1]+".txt"
    
        global audio_file
        global audio_thread

        context_classifier_path = f"plugins/SerpentRecorderGameAgentPlugin/files/ml_models/context_classifier.model"

        context_classifier = ImageNetwork(
            input_shape=(60, 80, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

        context_classifier.load_classifier(context_classifier_path)

        self.machine_learning_models["context_classifier"] = context_classifier

        # Start audio recording
        audio_thread = multiprocessing.Process(target=record)
        audio_thread.start()
        # Reset audio - jump file
        open(os.getcwd() + "\\audio\\raw\\timestamps.txt", 'w').close()

    def handle_play(self, game_frame):
        prediction = self.machine_learning_models["context_classifier"].predict(game_frame.frame)
        global RemovedB
        global timestamp
        global frame_count
        global key_pressed
        global audio_file
        global audio_thread

        # #Visual debugger
        # for i, game_frame in enumerate(self.game_frame_buffer.frames):
        #     self.visual_debugger.store_image_data(
        #         game_frame.grayscale_frame,
        #         game_frame.grayscale_frame.shape,
        #         str(i)
        #     )

        old_key_pressed = key_pressed
        key_pressed = keyboard.is_pressed('space')
        
        if prediction != 1:
            RemovedB = False

            def save_game_frame(frame,frame_cnt):
                audio_file = open(os.getcwd() + "\\audio\\raw\\timestamps.txt", 'a')
                if not (key_pressed or old_key_pressed):
                    frame.save("datasets\\" + timestamp + "\\no_jump\\" + timestamp[:-1] + "_" + str(frame_cnt) + ".png")
                    audio_file.write(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')) + " n " + str(frame_cnt) + "\n")
                    print("Writing to no_jump")
                else:
                    frame.save("datasets\\" + timestamp + "\\jump\\" + timestamp[:-1] + "_" + str(frame_cnt) + ".png")
                    audio_file.write(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')) + " j " + str(frame_cnt) + "\n")
                    print("Writing to jump")
                audio_file.close()
            
        
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
                    removeFramesFile.close()


                thread.start_new_thread(game_over,(frame_count,))
            #ONLY FOR TESTING SHOULD BE REMOVED LATER
            #frame_count +=1
