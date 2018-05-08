import pyaudio
import sounddevice as sd
import wave
import keyboard
import sys


def save(i):

    filename = "out%s.wav" %i

    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channelcount)
    waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(int(device_info["defaultSampleRate"]))
    waveFile.writeframes(b''.join(recorded_frames))
    waveFile.close()

def stop():
    stream.stop_stream()
    stream.close()

    #Close module
    p.terminate()    


defaultframes = 512


recorded_frames = []
device_info = {}
useloopback = False
recordtime = 0.5

#Use module
p = pyaudio.PyAudio()

#Set default to first in list or ask Windows
try:
    default_device_index = p.get_default_input_device_info()
except IOError:
    default_device_index = -1

#Select Device
print("Available devices:\n")
for i in range(0, p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(str(info["index"]) + ": \t %s \n \t %s \n" % (info["name"], p.get_host_api_info_by_index(info["hostApi"])["name"]))

    if default_device_index == -1:
        default_device_index = info["index"]

#Handle no devices available
if default_device_index == -1:
    print("No device available. Quitting.")
    exit()


#Get input or default
device_id = int(input("Choose device [" + str(default_device_index) + "]: ") or default_device_index)
print("")

#Get device info
try:
    device_info = p.get_device_info_by_index(device_id)
except IOError:
    device_info = p.get_device_info_by_index(default_device_index)
    print("Selection not available, using default.")

#Choose between loopback or standard mode
is_input = device_info["maxInputChannels"] > 0
is_wasapi = (p.get_host_api_info_by_index(device_info["hostApi"])["name"]).find("WASAPI") != -1
if is_input:
    print("Selection is input using standard mode.\n")
else:
    if is_wasapi:
        useloopback = True;
        print("Selection is output. Using loopback mode.\n" )
    else:
        print("Selection is output and does not support loopback mode. Quitting.\n")
        exit()

#recordtime = int(input("Record time in seconds [" + str(recordtime) +"]: ") or recordtime)

#Open stream
channelcount = device_info["maxInputChannels"] if (device_info["maxOutputChannels"] < device_info["maxInputChannels"]) else device_info["maxOutputChannels"]
stream = p.open(format = pyaudio.paInt16,
                channels = channelcount,
                rate = int(device_info["defaultSampleRate"]),
                input = True,
                frames_per_buffer = defaultframes,
                input_device_index = device_info["index"],
                as_loopback = useloopback)
                
#Start Recording
print ( "Starting... Press q to stop recording")
print("Won't start recording till it hears a sound")
def start():
    stream.start_stream()
    #while not keyboard.is_pressed('q'):
    for i in range(0, int(int(device_info["defaultSampleRate"]) / defaultframes * recordtime)):
        recorded_frames.append(stream.read(defaultframes))

for i in range(3):        
    try:
        recorded_frames = []
        keyboard.stash_state()
        start()
    except KeyboardInterrupt:
        save(i)
    stream.stop_stream()
    save(i)
