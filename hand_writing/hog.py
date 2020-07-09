from skimage import feature

#feature package contains many methods to extract features from images
class HOG:
    def __init__(self, orientations=9, pixelsPerCell=(8,8), cellsPerBlock=(3,3),transform=False):
        
        ## store the number of orientations, pixels per cell,
    	# cells per block, and whether or not power law	
        # compression should be applied

        self.orientations=orientations## of bins,i.e # of gradident orientation 
        self.pixelsPerCell=pixelsPerCell
        self.cellsPerBlock=cellsPerBlock
        self.transform=transform

    def describe(self,image):
        #compute HOG for the image
        hist=feature.hog(image,
                orientations=self.orientations,
                pixels_per_cell=self.pixelsPerCell,
                cells_per_block=self.cellsPerBlock,
                transform_sqrt=self.transform)
        #return HOG features
        return hist


