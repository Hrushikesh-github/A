from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K): #numpy arrays of image and kernel
    #grab the dim of image and kernel
    (iH,iW)=image.shape[:2]
    (kH,kW)=K.shape[:2]

    pad=(kW-1)//2
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output=np.zeros((iH,iW),dtype="float")
    '''
     copyMakeBorder( src, dst, top, bottom, left, right, borderType, value )
     The arguments are:
src: Source image
dst: Destination image
top, bottom, left, right: Length in pixels of the borders at each side of the image. 
BorderType: Define what type of border is applied. It can be constant or replicate for this example.
value: If borderType is BORDER_CONSTANT, this is the value used to fill the border pixels.
Replicated border: The border will be replicated from the pixel values at the edges of the original image.    
    '''
    for y in np.arange(pad, iH+pad):
        for x in np.arange(pad, iW+pad):
            roi=image[y-pad:y+pad+1, x-pad:x+pad+1]
            #perform the actual convultion between ROI and kernel
            k=(roi*K).sum()
#roi->region of interest
            output[y-pad,x-pad]=k
            #store the convolved value in the (x,y) coordinate of output image
            #note that it is relative positioning in output matrix
    output=rescale_intensity(output, in_range=(0,255))
    output=(output*255).astype("uint8")

    return output

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
[0, -1, 0],
[-1, 5, -1],
[0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
[0, 1, 0],
[1, -4, 1],
[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
[-1, 0, 1],
[-2, 0, 2],
[-1, 0, 1]), dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array((
[-1, -2, -1],
[0, 0, 0],
[1, 2, 1]), dtype="int")

# construct an emboss kernel
emboss = np.array((
[-2, -1, 0],
[-1, 1, 1],
[0, 1, 2]), dtype="int")

kernelBank = (
("small_blur", smallBlur),
("large_blur", largeBlur),
("sharpen", sharpen),
("laplacian", laplacian),
("sobel_x", sobelX),
("sobel_y", sobelY),
("emboss", emboss))

# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# loop over the kernels
for (kernelName, K) in kernelBank:
    # apply the kernel to the grayscale image using both our custom
    # ‘convolve‘ function and OpenCV’s ‘filter2D‘ function
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)
    # show the output images
    cv2.imshow("Original", gray)
    cv2.imshow("{} - convole".format(kernelName), convolveOutput)
    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
