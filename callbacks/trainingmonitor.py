from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

#A class that logs our loss and accuracy to disk
class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
    
        #store the output path for the figure, path to serialized JSON file 
        #and starting epoch
    
        super(TrainingMonitor, self).__init__()
        self.figPath=figPath
        self.jsonPath=jsonPath
        self.startAt=startAt
    
    def on_train_begin(self, logs={}):
        self.H={} #initialize the history dictionary, H->history of losses
        #if JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H= json.loads(open(self.jsonPath).read())
                # check to see if a starting epoch was supplied
                if self.startAt> 0:
                    #loop over the entries in the history log and trim any
                    #entries that are past the starting point
                    for k in self.H.keys():
                        self.H[k]=self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        #loop over the logs and update the loss, accuracy etc for 
        #entire training process
        for (k,v) in logs.items():
            l=self.H.get(k,[])
            l.append(v)
            self.H[k]=l
        #check to see if the training history should be serialized to a file
        if self.jsonPath is not None:
            f=open(self.jsonPath,"w")
            f.write(json.dumps(str(self.H)))
            f.close()
     #ensuring atleast 2 epochs have passes before plotting(epoch starts at 2)
        if len(self.H["loss"])>1:
            #plot training loss and accuracy
            N=np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title("Training Loss and Accuracy [Epoch {}".format
                    (len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            
            #save the fig
            plt.savefig(self.figPath)
           
        

