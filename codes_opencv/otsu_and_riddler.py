import numpy as np
import argparse
import cv2
import mahotas

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(image,(5,5),0)
cv2.imshow("Image",image)
cv2.imshow("Blurred image",blurred)


#use Otsu when there are two peaks in the grayscale histogram of the image — one for the background, another for the foreground.
#Otsu’s method assumes there are two peaks in the grayscale
#histogram of the image. It then tries to find an optimal
#value to separate these two peaks – thus our value of T.
T=mahotas.thresholding.otsu(blurred)
print("Otsu's threshold: {}".format(T))
thresh=image.copy()
print(thresh==image)
thresh[thresh>T]=255
thresh[thresh<255]=0
thresh=cv2.bitwise_not(thresh)
cv2.imshow("Otsu",thresh)
#another method which gives similar result as Otsu's
T=mahotas.thresholding.rc(blurred)
print("Riddler-Calvard: {}".format(T))
thresh=image.copy()
thresh[thresh>T]=255
thresh[thresh<255]=0#it is less than 255 not T because all nos > T are converted#to 255
thresh=cv2.bitwise_not(thresh)
cv2.imshow("Riddler-Calvard",thresh)
cv2.waitKey(0)


