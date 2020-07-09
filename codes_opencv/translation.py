import numpy as np
import argparse 
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original",image)

print("dimensions of image is:", image.shape)
a=image[:2]
print("dimensions of a=image[:2] is:", a.shape)
print("image.shape[:2] value is:",image.shape[:2])

M=np.float32([[1,0,25],[0,1,50]])
shifted=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
cv2.imshow("shifted down and right",shifted)

M=np.float32([[1,0,-50],[0,1,-90]])
shifted=cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))
cv2.imshow("shifted Up and Left",shifted)

shifted=imutils.translate(image,0,100)
cv2.imshow("Shifted Down", shifted)
cv2.waitKey(0)
