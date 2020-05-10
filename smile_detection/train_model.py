from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import utils

import sys
sys.path.append("/home/hrushikesh/dl4cv/conv")

from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,
        help="path to input dataset of faces")
ap.add_argument("-m","--model",required=True,help="path to output model")
args=vars(ap.parse_args())

import tensorflow as tf

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#initialize the list of data and labels
data=[]
labels=[]

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image=cv2.imread(imagePath)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image=imutils.resize(image, width=28)
    image=img_to_array(image)
    data.append(image)

    #extract class labels from image path and update the labels list
    label=imagePath.split("/")[-3]
    label="smiling" if label=="positives" else "not_smiling"
    labels.append(label)

data=np.array(data, dtype="float")/255.0
labels=np.array(labels)

le=LabelEncoder().fit(labels)
labels=utils.to_categorical(le.transform(labels),2)

classTotals=labels.sum(axis=0)# here gives [9475,3690]
classWeight=classTotals.max()/classTotals#gives [1,2.56]

(trainX,testX, trainY, testY)=train_test_split(data,labels,
        test_size=0.1, stratify=labels,random_state=42)

print("[INFO] compiling model")
model=LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

print("[INFO] training network...")
H= model.fit(trainX, trainY, validation_data=(testX, testY),
        class_weight=classWeight, batch_size=64, epochs=15, verbose=1)

print("[INFO] evaluating network...")
predictions=model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),
    target_names=le.classes_))

#save the model
print("[INFO] serializing the network...")
model.save(args["model"])

# plot the training + testing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()





