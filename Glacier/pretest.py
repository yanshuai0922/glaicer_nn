import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
#import imutils
import cv2
import copy
import scipy.io as sio
import Path
import random

import time

norm_size = 9
#filepath = "/home/ys/ys0922/research_python2/CNN_Keras/test_0110_0321/Sample_0110/"
filepath = "/home/ys/ys0922/research_python2/CNN_Keras/Select_Channel/sar/1022_1106_1201/val/"


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",
        help="path to trained model model", default="sar_1022_1106_1201.h5")#conv2_fc1
    ap.add_argument("-dtrain", "--dataset_train",
        help="path to input dataset_train", default= filepath + "00001")
    args = vars(ap.parse_args())
    return args

def normalize(data):
    size = np.shape(data)
    if img_i != 1:
        data = np.array(data, dtype="float") / 255
    return data

def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(Path.list_files(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        img = sio.loadmat(imagePath)
        image = img['SmallImage']
        image = np.array(image, dtype="float")
        image = normalize(image)
        # image = min_max_scaler.fit_transform(image)
        data.append(image)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    return data

def predict(args, train_data):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])

    # load the image
    a = train_data
    if img_i == 1:
        (num, w, h) = a.shape
    else:
        (num, w, h, s) = a.shape
    #print num
    for i in range(numfiles):
        zero = 0
        one = 0
        for j in range(i * (num // numfiles), i * (num // numfiles) + (num // numfiles)):
            if img_i == 1:
                crop = a[j, :, :]
                image = np.transpose(crop[None], (1, 2, 0))
            else:
                crop = a[j, :, :, :]
                image = crop
            #image = np.transpose(crop[None], (1, 2, 0))
            image = np.expand_dims(image, axis=0)
            result = model.predict(image)[0]
            proba = np.max(result)
            label = str(np.where(result == proba)[0])
            if label == '[0]':
                zero = zero + 1
            else:
                one = one + 1
        with open('./../select_result/sar/train/sar_1.txt', 'a') as f:
            f.write(str(zero) + ',')
            f.write(str(one) + '\n')
        print zero, one


# python predict.py --model traffic_sign.model -i 2.png -s
if __name__ == '__main__':
    time_start = time.time()
    img_i = input("please input number if you input 1 mean sar else mean msi:")
    numfiles = input("please input number to select:")
    args = args_parse()
    train_file_path = args["dataset_train"]
    train_data = load_data(train_file_path)
    predict(args, train_data)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')