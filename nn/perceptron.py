import numpy as np

class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W=np.random.randn(N+1)/np.sqrt(N)#+1 in shape of W for bias trick
        self.alpha=alpha
# we divide by sqrt of # of inputs to scale our weight matrix leading to faster
# convergence(a common technique)
 
#N is # of columns in our input feature vector
#here, in bitwise datasets, we'll set N equal to two as there are two inputs
#alpha is learning rate

    def step(self,x):
        return 1 if x>0 else 0



#to train the perceptron we will use the following function called 'fit'    
    def fit(self,X,y,epochs=10):
        #there is a threshold for firing the perceptron, adding bias to sum of
        #w.x resolves makes threshold=0(i.e bias=-threshold).Now we apply our
        #bias trick
        X=np.c_[X, np.ones((X.shape[0]))]
        #shape of x,W will be same 
        for epoch in np.arange(0, epochs):
            for (x,target) in zip(X,y):
                p=self.step(np.dot(x,self.W))
                #p is a integer 
                if p!= target:
                    error=p-target
                    self.W+=-self.alpha*error*x
        

    #note that X used below is same as x used above, the output of this 
    #function is p
    def predict(self,X, addBias=True):#put True because X above is not returned
        #print("before",X.shape)#(2,) for or dataset
        X=np.atleast_2d(X)
        #print("after",X.shape)#(1,2) for or dataset

        if addBias:
            X=np.c_[X,np.ones((X.shape[0]))]

        return self.step(np.dot(X, self.W))
