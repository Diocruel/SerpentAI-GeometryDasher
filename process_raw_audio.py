import numpy as np
import os
import datetime
from pathlib import Path
from pydub import AudioSegment


def find_files(path, pathList, extension, subFolders = True):
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

    path:        Base directory to find files
    pathList:    A list that stores all paths
    extension:   File extension to find
    subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
    """

    try:   # Trapping a OSError:  File permissions problem I believe
        for entry in os.scandir(path):
            if entry.is_file() and entry.path.endswith(extension):
                pathList.append(entry.path)
            elif entry.is_dir() and subFolders:   # if its a directory, then repeat process as a nested function
                pathList = findFilesInFolder(entry.path, pathList, extension, subFolders)
    except OSError:
        print('Cannot access ' + path + '. Probably a permissions error')

    return pathList


def check_jump(string_to_check):
    if string_to_check == "j":
        return true
    else:
        return false


if __name__ == "__main__":
    # Constants
    audio_feature_length = 0.5  # in seconds
    date_format = '%Y-%m-%d-%H-%M-%S-%f'  # in string format


    print("Processing raw audio")
    wav_files = []
    raw_audio_loc = os.getcwd() + "\\audio\\raw\\"
    wav_files = find_files(raw_audio_loc, wav_files, ".wav", False)

    # Get starting times for audio recording
    start_times = []
    for f in wav_files:
        time_rec, file_extension = os.path.splitext(f)
        time_rec = time_rec[len(raw_audio_loc):]
        date_rec = datetime.datetime.strptime(time_rec, '%Y-%m-%d-%H-%M-%S-%f')
        print(str(date_rec))
        start_times.append(date_rec)

    # Get jump time stamps
    p_jump = Path(os.getcwd() + "\\audio\\raw\\timestamps.txt")
    jump_times = p_jump.read_text().splitlines()
    jump_i = 0
    print(jump_times[jump_i])
    current_jump_time = datetime.datetime.strptime(jump_times[jump_i].split()[0], date_format)
    print(str(current_jump_time))

    # each audio file use start times to find first relevant jump action
    for i in range(0,len(wav_files)):
        # Read in audio
        fragment = AudioSegment.from_wav(wav_files[i])
        print(len(fragment))

        # Find start
        while current_jump_time + audio_feature_length > start_times[i] & jump_i < len(jump_times):
            pcurrent_jump_time = datetime.datetime.strptime(jump_times[jump_i].split()[0], date_format)
            jump_i += 1


    # fs1, y1 = scipy.io.wavfile.read(filename)
    # l1 = numpy.array([[7.2, 19.8], [35.3, 67.23], [103, 110]])
    # l1 = ceil(l1 * fs1)  # get integer indices into the wav file - careful of end of array reading with a check for
    # # greater than y1.shape
    # newWavFileAsList = []
    # for elem in l1:
    #     startRead = elem[0]
    #     endRead = elem[1]
    #     if startRead >= y1.shape[0]:
    #         startRead = y1.shape[0] - 1
    #     if endRead >= y1.shape[0]:
    #         endRead = y1.shape[0] - 1
    #     newWavFileAsList.extend(y1[startRead:endRead])