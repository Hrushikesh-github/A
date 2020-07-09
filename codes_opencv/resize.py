import numpy as np
import argparse
import imutils
import cv2
#difference b/w imutils resize and cv2 resize is imutils ensures aspect ratio
#but it should only be given one input either height/width.

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original",image)
#all belowtake care of aspect ratio
r=150/image.shape[1]
dim=(150, int(image.shape[0]*r))

resized=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)

r=50.0/image.shape[0]
dim=(int(image.shape[1]*r),50)

resized=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
cv2.imshow("Resized (Height)",resized)

resized=imutils.resize(image,height=110)
print(resized.shape)
cv2.imshow("Resized via Function",resized)
cv2.waitKey(0)
