from serpent.game_agent import GameAgent
import keyboard
import os
import _thread as thread
import numpy as np
import time
import multiprocessing
import pyaudio
import wave
from collections import deque
from AudioNetwork import AudioNetwork

 
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
        
class SerpentAudioGameAgent(GameAgent):
    
    def __init__(self, **kwargs):
     
        super().__init__(**kwargs)
        global frames
        global dq 
        dq = deque([], maxlen=88200)
        frames = multiprocessing.Queue()
        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play
        audio_thread = multiprocessing.Process(target=record, args=(frames,))
        audio_thread.start()
       
    
    def setup_play(self):
        classifier_path = f"datasets/pretrained_audio_classifier.model"

        classifier = AudioNetwork(
            input_shape=(88200, 1))  # Replace with the shape (rows, cols, channels) of your captured context frames

        classifier.load_classifier(classifier_path)

        self.machine_learning_models["classifier"] = classifier

    def handle_play(self, game_frame):
        global frames
        audioframe = []
        global dq 
       
        while not frames.empty(): 
            dq.extend(frames.get())
        audioframe = np.array(list(dq))
        if len(audioframe) == 88200 :
            audioframe = audioframe[..., np.newaxis]
            np.nan_to_num(audioframe, copy=False)
            audioframe[audioframe==0] = 1
            prediction = self.machine_learning_models["classifier"].predict(audioframe)
            #print("Prediction: " + str(prediction))
            #if (prediction == 1) :
                #self.input_controller.tap_key(KeyboardKey.KEY_UP)
