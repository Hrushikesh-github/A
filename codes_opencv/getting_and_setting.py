from __future__ import print_function
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original",image)

(b,g,r)=image[0,0]
print("Pixel at (0,0)- Red: {}, Green: {}, Blue: {}".format(r,g,b))
image[0,0]=(0,0,255)
(b,g,r)=image[0,0]
print("Pixel at (0,0)- Red: {}, Green: {}, Blue: {}".format(r,g,b))

pix_value=image[219,90]
print("pix value at x=90,y=219 is",pix_value)
corner=image[0:100, 0:400]
cv2.imshow("Corner",corner)
image[0:400, 0:100]=(0,255,0)
cv2.imshow("Updated",image)
cv2.waitKey(0)
cv2.imwrite("newimage.jpg", image)

