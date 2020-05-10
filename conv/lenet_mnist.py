from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.datasets import fetch_openml

import tensorflow as tf

print("[INFO] loading MNIST (full) dataset...")
dataset=fetch_openml('mnist_784',version=1,cache=False)

data=dataset.data
if K.image_data_format()=="channels_first":
    data=data.reshape(data.shape[0],1,28,28)
else:
    data=data.reshape(data.shape[0], 28, 28, 1)

(trainX, testX, trainY, testY)=train_test_split(data/255.0, dataset.target, test_size=0.25)

#converting the labels from integers to vectors-one-hot encoding
le=LabelBinarizer()
trainY=le.fit_transform(trainY)
testY=le.transform(testY)

print("[INFO] compiling model...")
opt=SGD(lr=0.01)
model=LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Other test.......................")
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#train the network
print("[INFO] training network..")
H=model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()



