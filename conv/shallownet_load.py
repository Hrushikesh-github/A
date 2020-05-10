import sys
sys.path.append('/home/hrushikesh/dl4cv/datasets')
sys.path.append('/home/hrushikesh/dl4cv/preprocessing')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from simpledatasetloader import SimpleDatasetLoader
from shallownet import ShallowNet

from tensorflow import keras

from imutils import paths
import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to pre-trained model")
args = vars(ap.parse_args())
# initialize the class labels
classLabels = ["cat", "dog", "panda"]

# grab the list of images in the dataset then randomly sample
# indexes into the image paths list
print("[INFO] sampling images...")


imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), 
#print(len(imagePaths))->3000
imagePaths = imagePaths[idxs]#check last in case u can't comprehend

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model=keras.models.load_model(args["model"])

print("[INFO] predicting..")
preds=model.predict(data, batch_size=32).argmax(axis=1)

#loop over the sample images
for (i,imagePath) in enumerate(imagePaths):
    #load the example image, draw the prediction, and display it to our screen
    image=cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),(10,30),
            cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,0),2)
    cv2.imshow("Image",image)
    cv2.waitKey(0)

'''
>>> c=np.random.random((3,3))
>>> a=c([2,2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'numpy.ndarray' object is not callable
>>> a=c[2,2]
>>> print(c)
[[ 0.30546908  0.31734987  0.1431588 ]
 [ 0.70247997  0.5869213   0.41610264]
 [ 0.267811    0.47380929  0.29428189]]
>>> print(a)
0.294281892128
>>> a=c[[2,2]]
>>> print(a)
[[ 0.267811    0.47380929  0.29428189]
 [ 0.267811    0.47380929  0.29428189]]
>>> a=c[[2,1]]
>>> print(a)
[[ 0.267811    0.47380929  0.29428189]
 [ 0.70247997  0.5869213   0.41610264]]
>>> a.shape
(2, 3)

so a=c[[#1,#2,...#10]] would give (10,1) matrix
'''
