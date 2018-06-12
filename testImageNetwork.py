from ImageNetwork import ImageNetwork
import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
    classifier_path = f"datasets/pretrained_classifier.model"

    classifier = ImageNetwork(
        input_shape=(60, 80, 3))  # Replace with the shape (rows, cols, channels) of your captured context frames

    classifier.load_classifier(classifier_path)

    no_jump = np.zeros([1339,1])
    yes_jump = np.ones([869, 1])

    y_true = np.append(no_jump,yes_jump)

    no_jump = no_jump - 1
    yes_jump = yes_jump - 2
    i = 0

    directory = os.getcwd() + "\\datasets\\test_set\\no_jump\\"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                im = cv2.imread(directory + file)
                prediction = classifier.predict(im)
                no_jump[i,0] = prediction
                i = i + 1

    i = 0
    directory = os.getcwd() + "\\datasets\\test_set\\yes_jump\\"
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".png"):
                im = cv2.imread(directory + file)
                prediction = classifier.predict(im)
                yes_jump[i, 0] = prediction
                i = i + 1


    error_no_jump = sum(no_jump)
    correct_no_jump = no_jump.shape[0] - error_no_jump
    correct_yes_jump = sum(yes_jump)

    acc = float(correct_no_jump + correct_yes_jump)/(no_jump.shape[0] + yes_jump.shape[0])

    y_pred = np.append(no_jump,yes_jump)

    auc_score = roc_auc_score(y_true,y_pred)

    print(auc_score)
    print(acc)