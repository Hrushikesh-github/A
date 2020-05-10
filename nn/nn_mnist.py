from neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
#to encode integer labels as vector labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

#loading the MNIST dataset and apply min/max scaling to scale the pixel intensity values to range [0,1]
#each image is 8*8 pixels 

digits=datasets.load_digits()
data=digits.data.astype("float")
data= (data-data.min())/(data.max()-data.min())#normalizing
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))

# for above, it is 1797*64

(trainX,testX,trainY,testY)=train_test_split(data,digits.target, test_size=0.25)

#convert the labels from integers to vectors

trainY= LabelBinarizer().fit_transform(trainY)
testY=LabelBinarizer().fit_transform(testY)

#train the network
print("[INFO] training network...")
nn=NeuralNetwork([trainX.shape[1], 32, 16,10])
print("[INFO] {}".format(nn))
nn.fit(trainX,trainY, epochs=1000)

print("[INFO] evaluating network...")
predictions=nn.predict(testX)#matirx of 450*10 for testing data
predictions=predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))
