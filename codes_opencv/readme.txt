This folder contains codes from practical python and opencv

First we start with load_display_save.py which we can understand what it contains from it's name

Pixel are the raw building blocks of an image. There is no finer granuality than the pixel.
image.shape is given by height * width where height is y-axis value. OpenCV reads image as an numpy matrix
The origin is located at top left and axes run along right and down respectively. OpenCV stores rgb values at a pixel in the form of (b,g,r). getting_setting.py works on this

drawing.py shows how we can draw circles, lines and rectangles. Their syntax

We next look into image processing, starting with image transformations of translating, rotating, resizing, flipping and cropping
translation.py, rotation.py, resize.py are the files but all these three are included in imutils.py as functions, which is better to check out
difference b/w imutils resize and cv2 resize is imutils ensures aspect ratio but it should only be given one input either height/width.
flipping.py and cropping.py is about flipping and cropping an image

Then comes image arithmetic where we discuss about 
modulus operation (adding 10 to 250 would give 255)
wrap around (adding 10 to 250 would give 4)
There is 'no correct way', we can decide how to manipulate the pixels. cv2.add() ensures modulus operation whereas np.uint8() + np.uint8() ensures wrap around. Check arithmetic.py for more details

Pixel value close to 255 is lighter and close to 0 is darker
if A is a numpy array, then [A] will represent a list

Bitwise operations- And Or Xor and Not are covered and note that Not is only for a single image, bitwise.py
Masking is a very useful feature which utilizes the bitwise_and ingeniously(the syntax is different though, check code),masking.py.the key point of masks is that they allow
us to focus our computation only on regions of the image
that interests us.

A color image consists of multiple channels, Red, Green and Blue component. We can access these components via indexing into Numpy arrays and also through cv2.split
For example, in a picture of a wave
The Red channel is very dark. This makes sense,
because an ocean scene has very few red colors in it. The
red colors present are either very dark, and thus not repre-
sented, or very light, and likely part of the white foam of
the wave as it crashes down.
Finally, the Blue channel (Bottom-Left) is extremely light,
and near pure white in some locations. This is because
shades of blue are heavily represented in our image.
We can merge these in another way too, which doesn't give enough perception
We also looked into how images appear in different color spaces without much info, check colorspaces.py
We can also obsever this from colour histograms.A histogram represents
the distribution of pixel intensities (whether color or gray-
scale) in an image. Running color_histogram.py can give more easily interpretable results. grayscale_histogram.py is for gray scale images
histogram_with_mask.py shows the result for a masked region in the image 

Then we come across blurring, an important topic. I have included notes at the end of blurring.py
Thresholding, which is the binarization of an image.We apply blurring before thresholding to remove some high frequency edges in the image we are not concerned with. 
simple_thresholding.py does simple thresholding( where we input the value of threshold)
One of the downsides of using simple thresholding meth-
ods is that we need to manually supply our threshold value
T. Not only does finding a good value of T require a lot of
manual experiments and parameter tunings, it’s not very helpful if the image exhibits a lot of range in pixel intensi-
ties.

In order to overcome this problem, we can use adap-
tive thresholding, which considers small neighbors of pixels
and then finds an optimal threshold value T for each neigh-
bor. This method allows us to handle cases where there
may be dramatic ranges of pixel intensities and the optimal
value of T may change for different parts of the image.
use Otsu when there are two peaks in the grayscale histogram of the image — one for the background, another for the foreground.
Otsu’s method assumes there are two peaks in the grayscale
histogram of the image. It then tries to find an optimal
value to separate these two peaks – thus our value of T.
Codes can be found in adaptive_thresholding.py and otsu_and_riddler.py

Edge detection is done using Laplacian and Sobel but we don't get crisp edges and thus use canny edge detector, laplacian_and_sobel.py and canny.py. You can find some info abt these in the respective files
note that edged is binary image, not gray scale

counting_coins.py uses many OpenCV features, like cv2.findContours, cv2.drawContours, cv2.boundingRect and cv2.minEnclosingCircle
