import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(image,(5,5),0)
cv2.imshow("Image",image)
cv2.imshow("Blurred image",blurred)
'''
The first argument we
supply is our blurred, grayscale image. Then, we need to
provide two values: threshold1 and threshold2.
Any edge pixel larger than threshold2 is considered
to be an edge. Any value below threshold1 is consid-
ered not to be an edge. Values in between threshold1
and threshold2 are either classified as edges or non-edges
based on their intensities are "connected"
'''
canny=cv2.Canny(image, 20, 220)#modify manually
cv2.imshow("Canny", canny)
cv2.waitKey(0)

# There is an article in medium that explains canny edge detection properly
