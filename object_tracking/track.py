import numpy as np
import argparse
import time
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video", help="path to the (optional) video file")
args=vars(ap.parse_args())

# Provide lower and upper bounds of the colour to be detected. Google palette may help 
green_yellowLower=np.array([29,86,6], dtype="uint8")
green_yellowUpper=np.array([64,255,255],dtype="uint8")

if not args.get("video",False):
    camera=cv2.VideoCapture(0)

else:
    camera=cv2.VideoCapture(args["video"])

while True:
    (grabbed,frame)=camera.read()

    if not grabbed:
        break

    green_yellow=cv2.inRange(frame, green_yellowLower, green_yellowUpper)
    #This function gives a thresholded image, with pixels falling within 
    #range are set to white and rest into black
    green_yellow=cv2.GaussianBlur(green_yellow,(3,3),0)

    
    ( cnts,_)=cv2.findContours(green_yellow.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts)>0:
        cnt=sorted(cnts,key=cv2.contourArea, reverse=True)[0]
        #sorts based on contour area, generally output is ascending but
        #here as we use reverse=True, we get descending order

        rect=np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame,[rect],-1,(0,0,255),2)

    cv2.imshow("Tracking",frame)
    cv2.imshow("Binary",green_yellow)

    time.sleep(0.025)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

