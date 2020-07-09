'''The code is used to detect and track the path of the object. I hace presumed the object detected to be a rectangle
but the code can be modified to a circle. The reason I didn't do is because I didn't have any circular objects like 
a ball to verify the code
'''
import numpy as np
import argparse
import time
from collections import deque
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video", help="path to the (optional) video file")
ap.add_argument("-b","--buffer", type=int, default=64, help="max buffer size")
args=vars(ap.parse_args())

# Provide lower and upper bounds of the colour to be detected. Google palette may help 
green_yellowLower=np.array([29,86,6], dtype="uint8")
green_yellowUpper=np.array([64,255,255],dtype="uint8")

pts = deque(maxlen=args["buffer"])
red = (0,0,255)

if not args.get("video",False):
    camera=cv2.VideoCapture(0)

else:
    camera=cv2.VideoCapture(args["video"])

while True:
    (grabbed,frame)=camera.read()

    if not grabbed:
        break

    green_yellow = cv2.GaussianBlur(frame,(3,3),0)
    green_yellow = cv2.inRange(green_yellow, green_yellowLower, green_yellowUpper)
    #This function gives a thresholded image, with pixels falling within 
    #range are set to white and rest into black
    # perform a series of dilations and erosions to remove any small blobs left in the mask
    green_yellow = cv2.erode(green_yellow, None, iterations=2)
    green_yellow = cv2.dilate(green_yellow, None, iterations=2)
    
    ( cnts,_)=cv2.findContours(green_yellow.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    # Only if any desired countour has been detected, proceed
    if len(cnts)>0:
        cnt=sorted(cnts,key=cv2.contourArea, reverse=True)[0]
        #sorts based on contour area, generally output is ascending but
        #here as we use reverse=True, we get descending order

        rect=np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
    
        cv2.drawContours(frame,[rect],-1,(0,0,255),2)

        (cx, cy) = (rect[0] + rect[1]) // 2
        #cv2.circle(frame,(centerX,centerY),5, red)
        pts.appendleft((cx,cy))

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i-1] is None or pts[i] is None:
            continue
        
        # otherwise, compute the thickness of the line and draw the connecting line
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Tracking",frame)
    cv2.imshow("Binary",green_yellow)

    time.sleep(0.025)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

