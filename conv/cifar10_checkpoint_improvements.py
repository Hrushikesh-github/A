from sklearn.preprocessing import LabelBinarizer
#from sklearn.metrics import classification_report

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#import sys
#sys.path.append("/home/hrushikesh/dl4cv/callbacks")

from minivggnet import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-w","--weights",required=True,
        help="path to the weights directory")
#output directory to store figure and serialized JSON training history
args=vars(ap.parse_args())

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

fname=os.path.sep.join([args["weights"],
    "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint= ModelCheckpoint(fname, monitor="val_loss", mode="min",
        save_best_only=True, verbose=1)
callbacks=[checkpoint]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
batch_size=64, epochs=40, callbacks=callbacks, verbose=2)
