# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py
from sklearn.tree import DecisionTreeClassifier


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
        help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
        help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
        help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

db=h5py.File(args["db"],"r") # r is read only mode
i=int(db["labels"].shape[0]*0.75)

data=db["features"][:]
data_labels=db["labels"][:]
train_X=data[:i]
test_X=data[i:]
print("trainX shape:",train_X.shape)
print("testX shape:",test_X.shape)
train_Y=data_labels[:i]
test_Y=data_labels[i:]

print("trainY shape:",train_Y.shape)
print("testY shape:",test_Y.shape)

print("testY",test_Y[:20])


# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3,n_jobs=args["jobs"])


model.fit(train_X,train_Y)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# evaluate the model

'''
print("n_classes:\n",forest.n_classes_)  # 3
print("classes:\n",forest.classes_)      # [0,1,2]
print("n_features:\n",forest.n_features_) # 25088
print("N-outputs:",forest.n_outputs_)    # 1
'''

print("[INFO] evaluating...")                 

preds=model.predict(test_X)
#print(preds.shape)

print("# of labels:",db["labels"][i:].shape)
print(classification_report(test_Y,preds,target_names=db["label_names"]))

# serialize the model to disk
print("[INFO] saving the model...")
f=open(args["model"],"wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

#close the database
db.close()

