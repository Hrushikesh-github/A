from sklearn.preprocessing import LabelBinarizer
#to encode integer labels as vector labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from sklearn import datasets
from sklearn.datasets import fetch_openml

import numpy as np
import argparse
import matplotlib.pyplot as plt



ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,
        help="path to the output loss/accuracy plot")
args=vars(ap.parse_args())

print("[INFO] loading MNIST (full) dataset...")
dataset=fetch_openml('mnist_784',version=1,cache=False)

data=dataset.data.astype("float")/255.0
(trainX, testX, trainY, testY)=train_test_split(data, dataset.target, test_size=0.25)

#converting the labels from integers to vectors-one-hot encoding
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

#define the 784-256-128-10 architecture using Keras

model=Sequential() #instantiate our network
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))
#softmax to obtain normalized class probabilites

#train the model using SGD

print("[INFO] training network")
sgd=SGD(0.01) #initialize SGD optimizer using learning rate 0.01
#the batch size is set in .fit method of the model
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])
H=model.fit(trainX, trainY, validation_data=(testX, testY),
        epochs=100, batch_size=128)
#H is a dictionary, which we'll use to plot

#evaluate the network
print("[INFO] evaluating network..")
predictions=model.predict(testX, batch_size=128)
#return class label probabilites for every data, size is (X,10) X-> # of data
print(classification_report(testY.argmax(axis=1), 
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))

#metric is: it's a function that, given predicted values and ground truth values from examples, provides you with a scalar measure of a "fitness" of your model, to the data you have. So, as you may see, a loss function is a metric, but the opposite doesn't always hold


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])





