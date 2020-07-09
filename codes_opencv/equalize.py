import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt


ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eq=cv2.equalizeHist(image)

cv2.imshow("Histogram Equlization",np.hstack([image,eq]))
cv2.imshow("Equalized",eq)
cv2.waitKey(0)

hist_before=cv2.calcHist([image],[0],None,[256],[0,256])
hist_after=cv2.calcHist([eq],[0],None,[256],[0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist_before,color="b")
plt.plot(hist_after,color="r")
plt.xlim([0,256])
plt.show()

