from neuralnetwork import NeuralNetwork
import numpy as np

#constructing the XOR dataset
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[1,0],[0,1],[0,1],[1,0]])

#defining and traing our neural network
nn=NeuralNetwork([2,2], alpha=0.5)

nn.fit(X,y, epochs=20000)

for (x,target) in zip(X,y):

    pred=nn.predict(x)[0][0]
    step=1 if pred>0.5 else 0
    print("[INFO] data={},ground-truth={}, pred={:.4f}, step={}".format(x,
        target[0], pred, step))
'''

putting no hidden layer we get the following
[INFO] data=[0 0],ground-truth=0, pred=0.5161, step=1
[INFO] data=[0 1],ground-truth=1, pred=0.5000, step=1
[INFO] data=[1 0],ground-truth=1, pred=0.4839, step=0
[INFO] data=[1 1],ground-truth=0, pred=0.4678, step=0

'''
