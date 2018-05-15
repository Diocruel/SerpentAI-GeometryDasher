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
    stream.stop_stream()
    stream.close()
    p.terminate()

class SerpentDasherGameAgent(GameAgent):
    global timestamp
    global frame_count
    global key_pressed
    global audio_file

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
        global audio_file
        # Start audio recording
        p = multiprocessing.Process(target=record)
        p.start()
        # Open audio time stamp file


    def handle_play(self, game_frame):

        def save_game_frame(frame,frame_cnt):
            audio_file = open(os.getcwd() + "\\audio\\raw\\timestamps.txt", 'a')
            if not key_pressed:
                time.sleep(0.03)
                if not (key_pressed or old_key_pressed):
                    frame.save("datasets\\" + timestamp + "\\no_jump\\" + str(frame_cnt) + ".png")
                    audio_file.write(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')) + " n\n")
                    # print("Writing to no_jump")
                else:
                    frame.save("datasets\\" + timestamp + "\\jump\\" + str(frame_cnt) + ".png")
                    audio_file.write(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')) + " j\n")
                    # print("Writing to jump")
            else:
                frame.save("datasets\\" + timestamp + "\\jump\\" + str(frame_cnt) + ".png")
                audio_file.write(str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')) + " j\n")
                # print("Writing to jump")
            audio_file.close()


        global timestamp
        global frame_count
        global key_pressed
        global audio_file

        small_im = game_frame.eighth_resolution_frame
        gray_im = Image.fromarray(small_im).convert("L")
        old_key_pressed = key_pressed
        key_pressed = keyboard.is_pressed('space')
        thread.start_new_thread(save_game_frame,(gray_im,frame_count,))
        frame_count += 1

