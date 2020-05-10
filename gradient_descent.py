from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))
    #returns b/w 0 to 1

def predict(X,W):
    #step function,if # >0.5 -> 1 otherwise 0
    #here # is the return from sigmoid_Activation
    #function, not the original #
    preds= sigmoid_activation(X.dot(W))
    preds[preds<=0.5]=0
    preds[preds>0]=1

    return preds

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
help="learning rate")
args = vars(ap.parse_args())    

# generate a 2-class classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X,y)=make_blobs(n_samples=1000, n_features=2,centers=2,cluster_std=1.5,
        random_state=1)
print("Original Shape of y was:", y.shape)
y=y.reshape((y.shape[0],1))#making it a 1d matrix rather than a vector

#print("size of X is:", X.shape)#(1000,2)
#print("size of y is:",y.shape)#(1000,1)

X=np.c_[X, np.ones((X.shape[0]))]#bias trick
#print("size of X after biasing is:", X.shape)#(1000,3)

(trainX, testX, trainY, testY)=train_test_split(X,y, test_size=0.5, random_state=42)

#print(trainY) gives only 0 or 1.
#print("shape of trainY is:", trainY.shape) #(500,1)
W=np.random.randn(X.shape[1],1) #our weighted matrix
#print("shape of weighted matrix is:",W.shape)
#shape of weighted matrix is: (3, 1)
losses=[]

for epoch in np.arange(0, args["epochs"]):
    preds=sigmoid_activation(trainX.dot(W))
    #i.e scoring_function=sigmoid_activation(linear_scoring_function)
    #and preds stores the value
    error=preds- trainY
    #the error matrix(here vector though) values will be b/w -1 and 1
    #print("dim of error matrix is:",error.shape) (500,1)
    loss=np.sum(error**2)#compute least square error over our predictions,
    print(loss)
    print(loss.shape)
    #a simple loss typically used for binary classification problems
    losses.append(loss)
     
    '''
    the gradient descent update is the dot product between our
    features and the error of the predictions.
    in the update stage, all we need to do is "nudge" the weight
    matrix in the negative direction of the gradient (hence the
    term "gradient descent" by taking a small step towards a set
    of "more optimal" parameters
    '''
    gradient=trainX.T.dot(error) #trainX.T makes it transpose of trainX
    #derived mathematically, check notes      

    W+= -args["alpha"]*gradient
    

    #check to see if an update should be displayed
    if epoch==0 or (epoch+1)%5==0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch+1),loss))

print("[INFO] evaluating")
preds=predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

