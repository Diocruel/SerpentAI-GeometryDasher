# DL-Final-Assignment
Final assignment for CS4180: Deep Learning at the TU Delft

# Audio

[Download this](https://github.com/intxcc/pyaudio_portaudio/releases/download/1.1/PyAudio-0.2.11-cp36-cp36m-win_amd64.whl)
pip install PyAudio-0.2.11-cp36-cp36m-win_amd64.whl

## Testing the file
python audio_test.py

It will only start recording once audio is actually playing over your speakers.
It will record until the *q*-key is pressed, which will loop 3 times and write the output to out0/1/2.wav
