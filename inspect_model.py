from tensorflow.keras.applications import VGG16
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


ap=argparse.ArgumentParser()
ap.add_argument("-i","--include-top",type=int, default=1,
        help="whether or not to include top of CNN")
args=vars(ap.parse_args())

#load the VGG16 network
print("[INFO] loading network...")
model=VGG16(weights="imagenet",include_top=args["include_top"]>0)
print("[INFO] showing layers")

# loop over the layers in network and display them to the console

for (i,layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i,layer.__class__.__name__))


