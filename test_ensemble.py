import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse
import glob 
import os

ap=argparse.ArgumentParser()
ap.add_argument("-m","--models",required=True,help="path to models directory")
args=vars(ap.parse_args())
(testX,testY)=cifar10.load_data()[1]
testX=testX.astype("float")/255.0
# convert the labels from integers to vectors
lb=LabelBinarizer()
testY=lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"]

modelPaths=os.path.sep.join([args["models"],"*.hdf5"])
print("before",modelPaths)
modelPaths=list(glob.glob(modelPaths))
models=[]
print(modelPaths)

for (i,modelPath) in enumerate(modelPaths):
    print("[INFO] loading model {}/{}".format(i+1,len(modelPaths)))
    models.append(load_model(modelPath ,compile=True))

print("[INFO] evaluating the ensemble...")
predictions=[]
for model in models:
    predictions.append(model.predict(testX,batch_size=64))

predictions=np.average(predictions,axis=0)
report=classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1),target_names=labelNames)
print(report)
f=open("/home/hrushikesh/dl4cv/ensemble_models/2/plots_graphs/ensemble_model.txt","w")
f.write(report)
f.close()
