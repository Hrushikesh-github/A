from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

import sys
sys.path.append("/home/hrushikesh/dl4cv/callbacks")

from minivggnet import MiniVGGNet
from trainingmonitor import TrainingMonitor
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,
        help="path to the output directory")
#output directory to store figure and serialized JSON training history
args=vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))
#can be used to kill the process in case it goes awry

print("[INFO] loading CIFAR-10 data")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb=LabelBinarizer()
# convert the labels from integers to vectors
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
"dog", "frog", "horse", "ship", "truck"]

#intialize the optimizer and model
print("[INFO] compiling model")
opt=SGD(lr=0.01, momentum=0.9, nesterov=True)
model= MiniVGGNet.build(width=32,height=32,depth=3, classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,
        metrics=["accuracy"])

# construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
 
