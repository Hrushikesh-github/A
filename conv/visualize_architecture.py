from lenet import LeNet
from tensorflow.keras.utils import plot_model

model=LeNet.build(28,28,1,10)#For MNIST digits classification
plot_model(model, to_file="lenet.png", show_shapes=True)


