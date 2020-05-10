import imutils 
import cv2

class AspectAwarePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        #storing the target image's width,height and interpolation method 
        #used when resizing
        self.width=width
        self.height=height
        self.inter=inter

    def preprocess(self,image):
        (h,w)=image.shape[:2]
        dW=0
        dH=0
        if w<h:
            image=imutils.resize(image, width=self.width, inter=self.inter)
            dH=int((image.shape[0]-self.height)/2.0)
        else:
            image=imutils.resize(image,height=self.height,inter=self.inter)
            dW=int((image.shape[0]-self.width)/2.0)

        (h,w)=image.shape[:2]
        image=image[dH:h-dH,dW:w-dW]
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
    #when cropping, maybe +- one pixel, therefore we make a call to cv2.resize
    #to ensure our output img has desired width and height
# when resizing we do it to the least dimension, otherwise it is not cropping but extending
