import numpy as np
import cv2
import dataset
import imutils

p = "/home/hrushikesh/book_opencv/Adrian Rosebrock - Practical Python and OpenCV, 3rd Edition + Case studies (2016)/Books/Case Studies, 3rd Edition/code/digits/data/digits.csv"

data = np.genfromtxt(p, delimiter=',',dtype="uint8")
target = data[:, 0]
data = data[:, 1:].reshape(data.shape[0], 28, 28)
print(data.shape)

images = data[:30,:,:]
print(images.shape)

for i in range(10, 20):
    image = imutils.resize(images[i,:,:],width=200)
#    cv2.imshow("image{}".format(i), image)
    image_deskew = dataset.deskew(image,200)
    stack = np.hstack([image, image_deskew])
    cv2.imshow("Image Deskewed",stack)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

