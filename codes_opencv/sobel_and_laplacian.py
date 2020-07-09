import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#blurred=cv2.GaussianBlur(image,(5,5),0) #not necessary
cv2.imshow("Image",image)

lap=cv2.Laplacian(image,cv2.CV_64F)
lap=np.uint8(np.absolute(lap))
cv2.imshow("Laplacian",lap)

sobelX=cv2.Sobel(image,cv2.CV_64F,1,0)
sobelY=cv2.Sobel(image,cv2.CV_64F,0,1)
(minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
print(minVal, maxVal) # returns -813 822 

sobelX=np.uint8(np.absolute(sobelX))
sobelY=np.uint8(np.absolute(sobelY))
#taking absolute will result in edges being shown white irrespective of whether they are postive slope or negative slope
(minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
print(minVal, maxVal) # returns 0 255

sobelCombined=cv2.bitwise_or(sobelX,sobelY)

cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Combined", sobelCombined)
cv2.waitKey(0)
'''
Why are we using a 64-bit float now?
The reason involves the transition of black-to-white and
white-to-black in the image.
Transitioning from black-to-white is considered a posi-
tive slope, whereas a transition from white-to-black is a
negative slope. If you remember our discussion of image
arithmetic in Chapter 6, you’ll know that an 8-bit unsigned
integer does not represent negative values. Either it will be
clipped to zero if you are using OpenCV or a modulus operation will be performed using NumPy.


The short answer here is that if you don’t use a floating
point data type when computing the gradient magnitude
image, you will miss edges, specifically the white-to-black
transitions.
In order to ensure you catch all edges, use a floating point
data type, then take the absolute value of the gradient im-
age and convert it back to an 8-bit unsigned integer, as in
Line 15. This is definitely an important technique to take
note of – otherwise you’ll be missing edges in your image!
'''
