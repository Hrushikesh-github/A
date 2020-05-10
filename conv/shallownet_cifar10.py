from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from shallownet import ShallowNet
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

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
opt=SGD(lr=0.01)
model= ShallowNet.build(width=32,height=32,depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
        optimizer=opt,metrics=["accuracy"])

#train the network
print("[INFO] training network..")
H=model.fit(trainX, trainY, validation_data=(testX, testY), 
        batch_size=32, epochs=40, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
