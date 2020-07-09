import numpy as np
import argparse
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to the image")
arg=vars(ap.parse_args())

image=cv2.imread(arg["image"])
cv2.imshow("Original",image)

blurred=np.hstack([cv2.blur(image,(3,3)),cv2.blur(image,(5,5)),cv2.blur(image,(7,7))])
cv2.imshow("Averaged",blurred)

blurred=np.hstack([cv2.GaussianBlur(image,(3,3),0),cv2.GaussianBlur(image,(5,5),0),cv2.GaussianBlur(image,(7,7),0)])
cv2.imshow("Gaussian",blurred)

blurred=np.hstack([cv2.medianBlur(image,3),cv2.medianBlur(image,5),cv2.medianBlur(image,7)])
cv2.imshow("Median",blurred)

blurred=np.hstack([cv2.bilateralFilter(image,5,21,21),cv2.bilateralFilter(image,7,31,31),cv2.bilateralFilter(image,9,41,41)])
cv2.imshow("Bilateral",blurred)

cv2.waitKey(0)

'''
Averaging:
As the name suggests, we are going to define a k × k slid-
ing window on top of our image, where k is always an odd
number. This window is going to slide from left-to-right
and from top-to-bottom. The pixel at the center of this ma-
trix (we have to use an odd number, otherwise there would
not be a true “center”) is then set to be the average of all
other pixels surrounding it.

Gaussian:
Gaussian blurring is similar to average blurring, but instead of
using a simple mean, we are now using a weighted mean,
where neighborhood pixels that are closer to the central
pixel contribute more “weight” to the average.
The end result is that our image is less blurred, but more
naturally blurred, than using the average method

Median:
Traditionally, the median blur method has been most ef-
fective when removing salt-and-pepper noise. This type of
noise is exactly what it sounds like: imagine taking a photo-
graph, putting it on your dining room table, and sprinkling
salt and pepper on top of it. Using the median blur method,
you could remove the salt and pepper from your image.

When applying a median blur, we first define our kernel
size k. Then, as in the averaging blurring method, we con-
sider all pixels in the neighborhood of size k × k. But, unlike
the averaging method, instead of replacing the central pixel
with the average of the neighborhood, we instead replace
the central pixel with the median of the neighborhood.
Median blurring is more effective at removing salt-and-
pepper style noise from an image because each central pixel
is always replaced with a pixel intensity that exists in the
image.
Averaging and Gaussian methods can compute means or
weighted means for the neighborhood – this average pixel
intensity may or may not be present in the neighborhood.
But by definition, the median pixel must exist in our neigh-
borhood. By replacing our central pixel with a median
rather than an average, we can substantially reduce noise.

Bilateral:
Thus far, the intention of our blurring methods has been
to reduce noise and detail in an image; however, we tend to
lose edges in the image.
In order to reduce noise while still maintaining edges, we
can use bilateral blurring. Bilateral blurring accomplishes
this by introducing two Gaussian distributions.
The first Gaussian function only considers spatial neigh-
bors, that is, pixels that appear close together in the ( x, y )
coordinate space of the image. The second Gaussian then
models the pixel intensity of the neighborhood, ensuring
that only pixels with similar intensity are included in the
actual computation of the blur.
Overall, this method is able to preserve edges of an im-
age, while still reducing noise. The largest downside to this
method is that it is considerably slower than its averaging,
Gaussian, and median blurring counterparts.
'''
