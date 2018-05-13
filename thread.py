import threading
import pyaudio
import wave
import sounddevice as sd
import keyboard
import sys
import os
import time


FORMAT = pyaudio.paInt16
defaultframes = 512
recordtime = 0.5
WAVE_OUTPUT_FILENAME = "file.wav"


def record(event):
    
    mydir = os.path.dirname(__file__)
    subdir = 'audio'
    p = pyaudio.PyAudio()
    
    #Select Device
    print ("Available devices:\n" )
    for i in range(0, p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print (str(info["index"]) + ": \t %s \n \t %s \n" % (info["name"], p.get_host_api_info_by_index(info["hostApi"])["name"]))
    
    device_id = -1
    for i in range(0, p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if "WASAPI" in (p.get_host_api_info_by_index(info["hostApi"])["name"]):
            device_id = i
    
    #Handle wasapi not found
    if device_id == -1:
        print ( "Couldn't find wasapi. Quitting." )
        exit()

    
    #Get device info
    try:
        device_info = p.get_device_info_by_index(device_id)
    except IOError:
        device_info = p.get_device_info_by_index(default_device_index)
        print ("Selection not available, using default.")
    
    #Choose between loopback or standard mode
    is_input = device_info["maxInputChannels"] > 0
    is_wasapi = (p.get_host_api_info_by_index(device_info["hostApi"])["name"]).find("WASAPI") != -1
    if is_input:
        print ("Selection is input using standard mode.\n")
    else:
        if is_wasapi:
            useloopback = True;
            print ( "Selection is output. Using loopback mode.\n")
        else:
            print ("Selection is input and does not support loopback mode. Quitting.\n")
            exit()
    
    #Open stream
    channelcount = device_info["maxInputChannels"] if (device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
    stream = p.open(format=FORMAT, channels=channelcount,
                    rate=int(device_info["defaultSampleRate"]), input=True,
                    frames_per_buffer=defaultframes,
                    input_device_index = device_info["index"],
                    as_loopback = useloopback)

    print("recording...")
    framecounter = 0
    while not event.wait(0):
        frames = []
        stream.start_stream()
        for i in range(0, int(int(device_info["defaultSampleRate"]) / defaultframes * recordtime)):
            frames.append(stream.read(defaultframes))
            
        stream.stop_stream()
        filename = "%s.wav" %framecounter
    
        wavefilepath = os.path.join(mydir, subdir, filename)
        waveFile = wave.open(wavefilepath,'wb')
        waveFile.setnchannels(channelcount)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(int(device_info["defaultSampleRate"]))
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        framecounter+=1
    print("finished recording")
    
    # stop Recording
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
  event = threading.Event()
  thread = threading.Thread(target=record, args=(event,))
  thread.start()

  counter = 0
  while counter<100000:
    print(counter)
    counter+=1
  
  #input("Press Enter to stop recording.")
  event.set()
  thread.join()
  print("done")


main()