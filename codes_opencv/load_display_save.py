from __future__ import print_function
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
args=vars(ap.parse_args())

# load the image
image=cv2.imread(args["image"])

print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {} pixels".format(image.shape[2]))

# display the image
cv2.imshow("image",image)
cv2.waitKey(0)

# save the image
cv2.imwrite("newimage.png", image)
#for jpg format
