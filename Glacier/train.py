# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

import time

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
#from sklearn.model_selection import train_test_split
#from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
#from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
#from Net import LeNet
from NetModel.Net_conv2_fc1 import LeNet

import Path
import scipy.io as sio
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

#filepath = "/home/ys/ys0922/research_python2/CNN_Keras/msi/Sample/"
filepath = "/home/ys/ys0922/research_python2/CNN_Keras/Select_Channel/msi_sar/0907_1022_1201/"
#filepath = "/home/ys/PycharmProjects_ys/ys0922/research_python2/CNN_Keras/Sample/"

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test",
        help="path to input dataset_test", default= filepath + "val")
    ap.add_argument("-dtrain", "--dataset_train",
        help="path to input dataset_train", default= filepath + "train")
    ap.add_argument("-m", "--model",
        help="path to output model", default="msi_sar_0907_1022_1201.h5")
    ap.add_argument("-p", "--plot", type=str, default="msi_sar_0907_1022_1201.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 10 #30
INIT_LR = 1e-3
#INIT_LR = 1e-4
#BS = 128
BS = 32
CLASS_NUM = 2
norm_size = 9

def normalize(data):
    size = np.shape(data)
    if img_type != 1:
        data = np.array(data, dtype="float") / 255
        #for i in range(size[2]):
            #b = data[:,:,i]
            #b = b - np.min(b)
            #data[:, :, i] = np.array(b / np.max(b), dtype="float")
            #print data[:, :, i]
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
        #print image.shape
        #image = min_max_scaler.fit_transform(image)
        data.append(image)


        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])       
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)                         
    return data,labels
    


def train(aug,trainX,trainY,testX,testY,args):
    # initialize the model
    print("[INFO] compiling model...")

    modelcheck = ModelCheckpoint(args['model'], monitor='val_acc', save_best_only=True, mode='max')
    callable = [modelcheck]

    model = LeNet.build(width=norm_size, height=norm_size, depth=7, classes=CLASS_NUM) # 7 6 1
    #model = LeNet.build(width=norm_size, height=norm_size, depth=1, classes=CLASS_NUM)
    #opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS, nesterov=True)
    opt = Adam(lr=INIT_LR,  decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1) # callbacks=callable,

    #with open('sample_9.txt', 'w') as f:
    #    f.write(str(H.history))

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on glacier classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    

if __name__=='__main__':
    time_start = time.time()
    img_type = input("please input data type if you input 1 means sar else means msi:")
    args = args_parse()
    train_file_path = args["dataset_train"]
    test_file_path = args["dataset_test"]
    trainX,trainY = load_data(train_file_path)
    testX,testY = load_data(test_file_path)
    print(trainX.shape)
    if img_type == 1:
        trainx = np.transpose(trainX[None], (1, 2, 3, 0))
        testx = np.transpose(testX[None], (1, 2, 3, 0))
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)
        train(aug,trainx,trainY,testx,testY,args)
    else:
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)
        train(aug, trainX, trainY, testX, testY, args)
        time_end = time.time()
    print('time cost', time_end - time_start, 's')
