import numpy as np
import scipy.io.wavfile


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
        print('Cannot access ' + path +'. Probably a permissions error')

    return pathList


if __name__ == "__main__":
    print("Processing raw audio")
    wav_files = []
    raw_audio_loc = os.getcwd() + "\\audio\\raw\\"

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