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

import time

import Path
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

norm_size = 9
filepath = "/home/ys/ys0922/research_python2/CNN_Keras/msi_sar/Image/"

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",
        help="path to trained model model", default="msi_sar_0907_1022_1201.h5")
    ap.add_argument("-i", "--image",
        help="path to input image", default=filepath + "msi1106_sar1104.mat")
    args = vars(ap.parse_args())
    return args

def normalize(data):
    size = np.shape(data)
    if img_type != 1:
        data = np.array(data, dtype="float") / 255
        #for i in range(size[2]):
            #b = data[:,:,i]
            #b = b - np.min(b)
            #data[:, :, i] = np.array(b / np.max(b), dtype="float")
    return data

def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    
    #load the image
    #image = cv2.imread(args["image"])
    img = sio.loadmat(args["image"])
    Image = img['SmallImage']
    image = np.array(Image, dtype="float")
    if img_type == 1:
        h, w = np.shape(image)
    else:
        h, w, _ = np.shape(image)

    padding_h = (h // norm_size + 1) * norm_size
    padding_w = (w // norm_size + 1) * norm_size
    if img_type != 1:
        padding_img = np.zeros((padding_h, padding_w, 7), dtype=np.uint8)
        padding_img[0:h, 0:w, :] = image[:, :, :]
    else:
        padding_img = np.zeros((padding_h, padding_w), dtype=np.float64)
        padding_img[0:h, 0:w] = image[:, :]
    mask_whole = np.zeros((padding_h, padding_w), dtype=np.float64)
    for i in range(padding_h // norm_size):
        for j in range(padding_w // norm_size):
            crop = padding_img[i * norm_size:i * norm_size + norm_size, j * norm_size:j * norm_size + norm_size]
            if img_type != 1:
                ch, cw, _ = np.shape(crop)
            else:
                ch, cw = np.shape(crop)
            if ch != norm_size or cw != norm_size:
                print 'invalid size!'
                continue
            # pre-process the image for classification
            crop = np.array(crop, dtype="float")
            crop = normalize(crop)
            if img_type == 1:
                crop = np.transpose(crop[None], (1, 2, 0))
            crop = np.expand_dims(crop, axis=0)

            # classify the input image
            result = model.predict(crop)[0]

            proba = np.max(result)
            label = str(np.where(result == proba)[0])
            #print label
            if label == '[1]':
                pred = np.ones((norm_size, norm_size)) * 255
            else:
                pred = np.zeros((norm_size, norm_size))

            mask_whole[i * norm_size:i * norm_size + norm_size, j * norm_size:j * norm_size + norm_size] = pred[:, :]

    cv2.imwrite('./3_1.png', mask_whole[0:h, 0:w])


#python predict.py --model traffic_sign.model -i 2.png -s
if __name__ == '__main__':
    time_start = time.time()
    img_type = input("please input data type if you input 1 means sar else means msi:")
    args = args_parse()
    predict(args)
    time_end = time.time()
    print('time cost', time_end - time_start, 's')