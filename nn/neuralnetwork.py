import numpy as np

class NeuralNetwork:
    def __init__(self,layers,alpha=0.1):
        #initializing list of weights matrices, then store network 
        #architecture and learning rate
        self.W=[]
        self.layers=layers#[2,2,1] implies our 1st layer(imput) has 2 nodes, 
        #our hidden layer has 2 and our output layer has 1
        self.alpha=alpha
        #start looping from index of 1st layer but stop 
        #before we reach the last two layers
        for i in np.arange(0, len(layers)-2):
            w=np.random.randn(layers[i]+1, layers[i+1]+1)
            #+1 for bias for both the layers neurons
            
            self.W.append(w/ np.sqrt(layers[i]))
        #the last case is special as no bias at output
        w=np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork:{}".format("-".join(str(l) for l in self.layers)) 
    #A printable representation of the object

    def sigmoid(self, x):
# compute and return the sigmoid activation value for a
# given input value
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
# compute the derivative of the sigmoid function ASSUMING
# that ‘x‘ has already been passed through the ‘sigmoid‘
# function
        return x * (1 - x)

    def fit(self, X,y, epochs=1000, displayUpdate=200):
        X=np.c_[X, np.ones((X.shape[0]))]#bias 
        
        for epoch in np.arange(0,epochs):
            for (x,target) in zip(X,y):
                self.fit_partial(x,target)

            
            if epoch==0 or (epoch+1)%displayUpdate == 0:
                loss=self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.6f}".format(epoch+1,loss))

    def fit_partial(self, x, y):
        #here x-the individual data point  which is returned by zip is a array
        A=[np.atleast_2d(x)]#A is a list for storing output activations
        #FEEDFORWARD
        for layer in np.arange(0, len(self.W)):
            net= A[layer].dot(self.W[layer])

#for xor data:
#shape of x is (3,)
#shape of A[0] is (1,3)
#shape of W[0] is (3, 3)

            
            out=self.sigmoid(net)
            A.append(out)
        #FORWARD PROPAGATION DONE, now should do BACKWARD PROPAGATION

        #BACKPROPAGATION
        error=A[-1]-y 
        #error is here,also equal to 'gradient' of cost function w.r.t A[-1] as we take  C=0.5*(A[-1]-y)^2, thus 
        D=[error*self.sigmoid_deriv(A[-1])]

        for layer in np.arange(len(A) - 2, 0, -1):
# the delta for the current layer is equal to the delta
# of the *previous layer* dotted with the weight matrix
# of the current layer, followed by multiplying the delta
# by the derivative of the nonlinear activation function
# for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        D=D[::-1]
            
       
        for layers in np.arange(0, len(self.W)):
            self.W[layers]+= -self.alpha * (A[layers].T.dot(D[layers]) )
#(A[layers].T.dot(D[layers])) completes the chain rule and gives the gradient
                #the above line is our gradient descent
    
    def predict(self, X, addBias=True):
# initialize the output prediction as the input features -- this
# value will be (forward) propagated through the network to
# obtain the final prediction
        p = np.atleast_2d(X)
# check to see if the bias column should be added
        if addBias:
# insert a column of 1’s as the last entry in the feature
# matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
# loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
# computing the output prediction is as simple as taking
# the dot product between the current activation value ‘p‘
# and the weight matrix associated with the current layer,
# then passing this value through a nonlinear activation
# function
            p = self.sigmoid(np.dot(p, self.W[layer]))
# return the predicted value
        return p

    def calculate_loss(self, X, targets):
# make predictions for the input data points then compute
# the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

    # return the loss
        return loss
