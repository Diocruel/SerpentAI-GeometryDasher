from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
import keyboard
import os
import sys
sys.path.append(os.getcwd())
import _thread as thread
import numpy as np
import time
import multiprocessing
import pyaudio
import wave
from collections import deque
from AudioNetwork import AudioNetwork
from ImageNetwork import ImageNetwork

def record(frames):
    global FORMAT
    global defaultframes
    defaultframes = 512
    FORMAT = pyaudio.paInt16
    device_info = {}
    useloopback = False
    
    # Use module
    p = pyaudio.PyAudio()
    
    # Get input or default
    device_id = 5
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
    
    print("recording...")
    #print(int(device_info["defaultSampleRate"])) #44100
    while not keyboard.is_pressed('q'):
        stream.start_stream()
        #start_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        #for i in range(0, int(int(device_info["defaultSampleRate"]) / defaultframes * recordtime)):
        frames.put(np.fromstring(stream.read(defaultframes), 'Float32'))
       # print(list(frames))
    
    
    # stop Recording
    stream.stop_stream()
    stream.close()
    p.terminate()

class SerpentCombineGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global frames
        global dq 
        dq = deque([0]*88200, maxlen=88200)
        global total_frames 
        total_frames = 0
        global correct_frames
        correct_frames = 0
        global incorrect_frames 
        incorrect_frames = 0
        frames = multiprocessing.Queue()
        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        audio_thread = multiprocessing.Process(target=record, args=(frames,))
        audio_thread.start()

    def setup_play(self):
        image_classifier_path = f"datasets/pretrain_classifier_super_final.model"
        audio_classifier_path = f"datasets/pretrained_audio_classifier.model"

        audio_classifier = AudioNetwork(
            input_shape=(88200, 1))
        image_classifier = ImageNetwork(
            input_shape=(60, 80, 3))  
       
        image_classifier.load_classifier(image_classifier_path)
        audio_classifier.load_classifier(audio_classifier_path)

        self.machine_learning_models["image_classifier"] = image_classifier 
        self.machine_learning_models["audio_classifier"] = audio_classifier

    def handle_play(self, game_frame):
        eightframe = game_frame.eighth_resolution_frame
        image_prediction = self.machine_learning_models["image_classifier"].predict(eightframe)
                
        global frames
        audioframe = []
        global dq 
        global total_frames
        global correct_frames
        global incorrect_frames
        key_pressed = keyboard.is_pressed('space')
       
        while not frames.empty(): 
            dq.extend(frames.get())
        audioframe = np.array(list(dq))

        if len(audioframe) == 88200 :
            audioframe = audioframe[..., np.newaxis]
            print(audioframe.shape)
            audio_prediction = self.machine_learning_models["audio_classifier"].predict(audioframe)
            prediction = (image_prediction + audio_prediction)/2
            if key_pressed and prediction > 0.5 :
                total_frames = total_frames + 1
                correct_frames = correct_frames + 1
            elif not(key_pressed) and prediction <= 0.5 :
                total_frames = total_frames + 1
                correct_frames = correct_frames + 1
            else :
                total_frames = total_frames + 1
                incorrect_frames = incorrect_frames + 1
                
            print("audio", audio_prediction, "image", image_prediction, "total", prediction)
            print("Accuracy", total_frames, "correct", correct_frames, "incorrect", incorrect_frames)
        #if (prediction > 0.5) :
        #    self.input_controller.tap_key(KeyboardKey.KEY_UP)
       
