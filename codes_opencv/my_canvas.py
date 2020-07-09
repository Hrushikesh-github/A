import numpy as np
import cv2

canvas=np.zeros((300,400,3),dtype="uint8")
canvas[1:70]=(29,86,6)#light yellow
canvas[120:190]=(64,255,255)
canvas[190:260]=(0,204,204)
canvas[260:300]=(0,153,153)
cv2.imshow("my_canvas",canvas)
cv2.waitKey(0)

