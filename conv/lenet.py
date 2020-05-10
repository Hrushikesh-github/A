from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#The Sequential class indicates that our network will be feedforward and layers will be added to the class sequentially, one on top of the other. The Dense class on is the implementation of our fully-connected layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K #to access the .json file

class LeNet:
    def build(width, height, depth, classes):
        model=Sequential()
        inputShape=(height, width, depth)

        if K.image_data_format()=="channels_first":
            inputShape= (depth, height, width)
        
        model.add(Conv2D(20,(5,5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        #1st set of CONV-> RELU-> POOL layers, lets define 2nd one
        model.add(Conv2D(50, (5, 5), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        #softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model



