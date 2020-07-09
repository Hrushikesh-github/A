from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image")
args=vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Original",image)
cv2.waitKey(0)

chans=cv2.split(image)
colors=("b","g","r")
plt.figure()
plt.title("'Flattened' Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")

for (chan,color) in zip(chans,colors):
    hist=cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color=color)
    plt.xlim([0,256])
plt.show()


fig=plt.figure()

ax=fig.add_subplot(131)
hist=cv2.calcHist([chans[1],chans[0]],[0,1],None,[32,32],[0,256,0,256])
#hist is 2d numpy array and here it is 32*32
p=ax.imshow(hist, interpolation="nearest")
#imshow will display data as an image
ax.set_title("2D color histogram for g and B")
plt.colorbar(p)

ax=fig.add_subplot(132)
hist=cv2.calcHist([chans[1],chans[2]],[0,1],None,[32,32],[0,256,0,256])
p=ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color histogram for g and R")
plt.colorbar(p)

ax=fig.add_subplot(133)
hist=cv2.calcHist([chans[0],chans[2]],[0,1],None,[32,32],[0,256,0,256])
p=ax.imshow(hist, interpolation="nearest")
ax.set_title("2D color histogram for R and B")
plt.colorbar(p)

print("2D histogram: {},with {} value".format(hist.shape,hist.flatten().shape[0]))
#flatten makes it 1d, and using shape would give total no of values
plt.show()

hist = cv2.calcHist([image], [0, 1, 2],None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(hist.shape, hist.flatten().shape[0]))
#plt.show() will not give any result, it can't be visualized
