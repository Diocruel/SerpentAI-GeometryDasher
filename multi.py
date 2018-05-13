import threading
import pyaudio
import wave
import sounddevice as sd
import keyboard
import sys
import os
import time

from datetime import datetime
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool


# See https://pymotw.com/2/multiprocessing/basics.html
def action1():
    for i in range(1,10):
        print(str(datetime.now()))
        time.sleep(1)


def action2():
    for i in range(1,10):
        print("pizza #", str(i), "!")
        time.sleep(2)


if __name__ == "__main__":
    jobs = []
    p1 = multiprocessing.Process(target=action1)
    p2 = multiprocessing.Process(target=action2)

    jobs.append(p1)
    jobs.append(p2)

    print("Starting first thread at:", str(datetime.now()))
    p1.start()
    time.sleep(2)
    print("Starting second thread at:", str(datetime.now()))
    p2.start()
    print("Started both threads at:", str(datetime.now()))
    # Wait till processes have been completed
    p1.join()
    p2.join()

    print("Done, exiting program at:", str(datetime.now()))