import os
import numpy as np
import librosa
from AudioNetwork import AudioNetwork
import cv2
from sklearn.metrics import roc_auc_score


def audio_norm(data):
    np.nan_to_num(data, copy=False)
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5


if __name__ == '__main__':

    classifier_path = f"datasets/pretrained_audio_classifier.model"
    test_path_no_jump = f"datasets/current/training/no_jump"
    test_path_jump = f"datasets/current/training/yes_jump"

    classifier = AudioNetwork(
        input_shape=(22050, 1))  # Replace with the shape (rows, cols, channels) of your captured context frames

    classifier.load_classifier(classifier_path)

    no_jump = np.zeros([172, 1])
    yes_jump = np.ones([231, 1])

    y_true = np.append(no_jump,yes_jump)

    no_jump = no_jump - 1
    yes_jump = yes_jump - 2

    i = 0
    directory = os.getcwd() + test_path_no_jump
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audioframe, _ = librosa.core.load(directory+file, sr=44100)
                audioframe = audio_norm(audioframe)
                audioframe = audioframe[..., np.newaxis]
                np.nan_to_num(audioframe, copy=False)
                audioframe[audioframe == 0] = 1
                prediction = classifier.predict(audioframe)
                no_jump[i,0] = prediction
                i = i + 1


    i = 0
    directory = os.getcwd() + test_path_jump
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                audioframe, _ = librosa.core.load(directory+file, sr=44100)
                audioframe = audio_norm(audioframe)
                audioframe = audioframe[..., np.newaxis]
                np.nan_to_num(audioframe, copy=False)
                audioframe[audioframe == 0] = 1
                prediction = classifier.predict(audioframe)
                yes_jump[i,0] = prediction
                i = i + 1


    error_no_jump = sum(no_jump)
    correct_no_jump = no_jump.shape[0] - error_no_jump
    correct_yes_jump = sum(yes_jump)

    acc = float(correct_no_jump + correct_yes_jump)/(no_jump.shape[0] + yes_jump.shape[0])

    y_pred = np.append(no_jump,yes_jump)

    auc_score = roc_auc_score(y_true,y_pred)

    print(auc_score)
    print(acc)






