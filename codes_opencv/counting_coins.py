import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,help = "Path to the image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imwrite("Original.png",image)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow("Image",image)
cv2.imshow("Blurred image",blurred)

edged=cv2.Canny(blurred,30,150)
cv2.imshow("Edges",edged) #note that edged is binary image, not gray scale

( cnts,_)=cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# copy because image is destroyed afterwards
#RETR_EXTERNAL to retrieve only outermost contours and RETR_LIST for all contour
#print(len(cnts[1]))
#print(_.shape==edged.shape)
print("I count {} coins in this image".format(len(cnts)))

coins=image.copy()
cv2.drawContours(coins,cnts,-1,(155,255,155),(2))
# '-1' indicates we want to draw all contours and last argument is thickness
cv2.imshow("Coins",coins)
cv2.waitKey(0)

for (i,c) in enumerate(cnts): #is the number of iterations variable 
    (x,y,w,h)= cv2.boundingRect(c)

    print("Coin #{}".format(i+1))
    coin=image[y:y+h, x:x+w]
    cv2.imshow("Coin", coin)

    mask=np.zeros(image.shape[:2],dtype="uint8")
    ((centerX, centerY),radius)= cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)),int(radius),255,-1)
    mask=mask[y:y+h, x:x+w]
    cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask=mask))
    cv2.waitKey(0)
cv2.imwrite("Contours.png",coins)
cv2.imwrite("Edge_Detection.png",edged)
