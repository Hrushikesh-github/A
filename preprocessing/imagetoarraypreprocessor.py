#does nothing, may reverse order to depth,width and height
from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None): #input->'channels_first' to change
        self.dataFormat=dataFormat

    def preprocess(self,image):
        return img_to_array(image,data_format=self.dataFormat)

