import cv2
import numpy as np

canvas=np.zeros((300,300,3), dtype="uint8")
i=0
j=0
while i<296:
    while j<296:
        if (((i+j)%2)==0):
            canvas[j:j+5,i:i+5]=(0,0,255)
        j+=5
    j=0    
    i+=5
cv2.circle(canvas,(150,150),50,(0,255,0),-1)
cv2.imshow("canvas",canvas)
cv2.waitKey(0)

