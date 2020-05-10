from perceptron import Perceptron
import numpy as np

# construc the OR dataset

X= np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])#our target

print("[INFO] training perceptron...")
p=Perceptron(X.shape[1], alpha=0.1) #instantiating
p.fit(X,y, epochs=20)

print("[INFO] testing perceptron")

for (x, target) in zip(X,y):
    pred=p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0],pred))

'''
This will fail because xor is nonlinear but the mapping function that we obtain
from perceptron algorithm will always be linear
It is linear because it is eventually sum of dot products
'''
