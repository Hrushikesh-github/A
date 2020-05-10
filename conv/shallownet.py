from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#The Sequential class indicates that our network will be feedforward and layers will be added to the class sequentially, one on top of the other. The Dense class on is the implementation of our fully-connected layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten

from tensorflow.keras import backend as K #to access the .json file

class ShallowNet:
    def build(width, height, depth, classes):
        model=Sequential()
        inputShape=(height,width,depth)

        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)
        #define the 1st(and only) CONV=> RELU layer
        model.add(Conv2D(32, (3,3), padding="same",input_shape=inputShape))
        # layer will have 32 filters(K) each of 3*3(i.e F*F)
        #same padding ensures size of input and output from convolution is same
        model.add(Activation("relu"))
        #after convolution we apply ReLU activation
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
