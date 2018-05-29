import numpy as np
import os
import datetime
from pathlib import Path
from pydub import AudioSegment


def find_files(path, path_list, extension, sub_folders = True):
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

    path:        Base directory to find files
    pathList:    A list that stores all paths
    extension:   File extension to find
    subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
    """

    try:   # Trapping a OSError:  File permissions problem I believe
        for entry in os.scandir(path):
            if entry.is_file() and entry.path.endswith(extension):
                path_list.append(entry.path)
            elif entry.is_dir() and sub_folders:   # if its a directory, then repeat process as a nested function
                path_list = find_files(entry.path, path_list, extension, sub_folders)
    except OSError:
        print('Cannot access ' + path + '. Probably a permissions error')

    return path_list


def check_jump(string_to_check):
    if string_to_check == "j":
        return true
    else:
        return false


if __name__ == "__main__":
    # Constants
    audio_feature_length = 2000  # in milliseconds
    date_format = '%Y-%m-%d-%H-%M-%S-%f'  # in string format based on agent settings

    # Create directories
    process_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S\\')
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\audio\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\audio\\" + process_time), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\audio\\" + process_time + "\\jump\\"), exist_ok=True)
    os.makedirs(os.path.dirname(os.getcwd() + "\\datasets\\audio\\" + process_time + "\\no_jump\\"), exist_ok=True)

    print("Reading in files ...")
    wav_files = []
    raw_audio_loc = os.getcwd() + "\\audio\\raw\\"
    wav_files = find_files(raw_audio_loc, wav_files, ".wav", False)

    # Get starting times for audio recording
    start_times = []
    for f in wav_files:
        time_rec, file_extension = os.path.splitext(f)
        time_rec = time_rec[len(raw_audio_loc):]
        date_rec = datetime.datetime.strptime(time_rec, '%Y-%m-%d-%H-%M-%S-%f')
        print(str(f))
        start_times.append(date_rec)

    # Get jump time stamps
    p_jump = Path(os.getcwd() + "\\audio\\raw\\timestamps.txt")
    jump_times = p_jump.read_text().splitlines()

    print("\nFound " + str(len(wav_files)) + " sound fragments and " + str(len(jump_times)) + " frames timestamps.")

    # Set first jump
    jump_i = 0
    current_jump_time = datetime.datetime.strptime(jump_times[jump_i].split()[0], date_format)
    duration_fragments = datetime.timedelta(milliseconds=audio_feature_length)
    current_jump_time_with_offset = current_jump_time + duration_fragments

    frame_counter_j = 0
    frame_counter_nj = 0
    frame_counter_j_t = 0
    frame_counter_nj_t = 0
    # each audio file use start times to find first relevant jump action
    for i in range(0,len(wav_files)):
        # Read in audio
        print("\tCreating audio frames for audio fragment " + str(i+1) + ", out of " + str(len(wav_files)))
        fragment = AudioSegment.from_wav(wav_files[i])
        fragment_duration = len(fragment)
        fragment_end_time = start_times[i] + datetime.timedelta(milliseconds=fragment_duration)

        # Find first possible starting frame
        while (current_jump_time_with_offset < start_times[i]) & (jump_i < len(jump_times) - 1):
            jump_i += 1
            current_jump_time = datetime.datetime.strptime(jump_times[jump_i].split()[0], date_format)
            current_jump_time_with_offset = current_jump_time + duration_fragments

        # Label all frames within audio fragment
        while (current_jump_time < fragment_end_time) & (jump_i < len(jump_times) - 1):
            diff_start = current_jump_time - start_times[i] - duration_fragments
            start_time_frame_ms = np.floor((diff_start.days * 86400000) + (diff_start.seconds * 1000) + (diff_start.microseconds / 1000))

            # Print information about fragment
            # print("Timestamp " + str(current_jump_time))
            # print("Jump j/n: " + str(jump_times[jump_i].split()[1]))
            # print("Start audio frame " + str(start_times[i]))
            # print("Start time in ms: " + str(start_time_frame_ms))
            # print("end time in ms: " + str(start_time_frame_ms+audio_feature_length))
            # print("Duration in ms: " + str(start_time_frame_ms+audio_feature_length-start_time_frame_ms))
            # print(" ")

            # Classify frame and write audio frame
            frame = fragment[start_time_frame_ms:(start_time_frame_ms+audio_feature_length)]
            if jump_times[jump_i].split()[1] == "j":
                class_directory = "\\jump\\"
                frame_counter_j += 1
            else:
                frame_counter_nj += 1
                class_directory = "\\no_jump\\"
            frame_counter = frame_counter_j + frame_counter_nj
            frame.export(os.getcwd() + "\\datasets\\audio\\" + process_time + class_directory + str(i+1)+ "_" +str(frame_counter) + ".wav", format="wav")

            # Update to next jump timestamp
            jump_i += 1
            current_jump_time = datetime.datetime.strptime(jump_times[jump_i].split()[0], date_format)
            current_jump_time_with_offset = current_jump_time + duration_fragments

        # Indicate that fragment is completed and update counters
        print("\tDone with file number " + str(i + 1) + ", created " + str(frame_counter_nj+frame_counter_j) + " frames, with " + str(frame_counter_j) +  " jump_frames and " + str(frame_counter_nj) + " no jump frames.\n")
        frame_counter_j_t += frame_counter_j
        frame_counter_j = 0
        frame_counter_nj_t += frame_counter_nj
        frame_counter_nj = 0
    print("Done with all fragments. Created in total " + str(frame_counter_nj_t+frame_counter_j_t) + " frames, of which " + str(frame_counter_j_t) + " are jump frames and " + str(frame_counter_nj_t) +  " are non-jump frames.")
    print("Output can be found in: " + str(os.path.dirname(os.getcwd() + "\\datasets\\audio\\" + process_time)))
    print("Do not forget to delete the audio in raw data folder before recording!")