import imutils
import numpy as np
import mahotas
import cv2

def load_digits(datasetPath):

    #the path to where the MNIST sample dataset resides
    data = np.genfromtxt(datasetPath, delimiter=",", dtype="uint8")
    #dataset is unloaded and stored as unsigned 8 bit ints, same as our image
    #whose pixel intensities ranges from 0 to 255
    target=data[:,0]#target range is from 0 to 9
    data=data[:,1:].reshape(data.shape[0],28,28)
    #then split it into data and labels/targets and return them as tuples
    return (data, target)

def deskew(image,width):
    (h,w)=image.shape[:2]
    moments=cv2.moments(image)
    # moments contains statistical info regarding the distribution of locatoin of white pixels in the image
    # something related to centroid, moment of inertia etc
    #deskew the image by some affine transformations
    skew=moments["mu11"]/moments["mu02"]

    M=np.float32([[1,skew,-0.5*w*skew],[0,1,0]])#the direction in which image is going to be deskewed
    image=cv2.warpAffine(image, M, (w,h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    #here w,h is resulting width and height of deskewed image and flags is in which 
    #technique image will be deskewed

    image=imutils.resize(image,width=width)

    return image

def center_extent(image,size):
    (eW,eH)=size

    if image.shape[1]>image.shape[0]:
        image=imutils.resize(image,width=eW)
    else:
        image=imutils.resize(image,height=eH)

    extent=np.zeros((eH,eW),dtype="uint8")

    offsetX=(eW-image.shape[1])//2
    offsetY=(eH-image.shape[0])//2
    extent[offsetY:offsetY+image.shape[0], offsetX:offsetX+image.shape[1]]=image
    #try to visualize what's happening, its simple

    CM=mahotas.center_of_mass(extent)#weighted mean of white pixels in image
    (cY, cX) = np.round(CM).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    #M = np.float32([[1, 0, dX], [0, 1, dY]])
    #extent = cv2.warpAffine(extent, M, size)
    extent=imutils.translate(extent,dX,dY)
    #the above two lines are nothing but translation
    return extent
