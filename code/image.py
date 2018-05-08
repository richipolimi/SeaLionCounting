import cv2
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from config import *
from pathlib import Path
from region_proposal import RegionProposal

UNKNOWN = 0
RED = 1
MAGENTA = 2
GREEN = 3
BLUE = 4
BROWN = 5

class Image(object):

    rp = RegionProposal()

    def __init__(self, image_id):
        self.real_path = TRAIN_DIR + str(image_id) + ".jpg"  # Path to the non-dotted training image
        self.dotted_path = TRAIN_DOTTED_DIR + str(image_id) + ".jpg"  # Path to the dotted training image
        self.boxes_path = BOXES_DIR + str(image_id) + '.box'  # Path to the boxes file
        self.coord_path = COORDINATES_DIR + str(image_id) + ".coor"  # Path to the boxes file

    
    def get_boxes(self):
    """ Generates a list of Box instances from non-dotted training image """

        # generate boxes if they are not already present in a file
        boxes_file = Path(self.boxes_path)
        if not boxes_file.is_file():
            rects = Image.rp.generate_boxes(self.real_path)
            np.savetxt(self.boxes_path, rects, fmt='%d')

        # read file
        boxes = np.loadtxt(self.boxes_path, dtype=int)
        return boxes


    def get_coordinates(self):
    """ Generate a list of coordinate pairs from dotted training image """
    
        # generate boxes if they are not already present in a file
        coord_file = Path(self.coord_path)
        if not coord_file.is_file():
            # read the Train and Train Dotted images
            image_1 = cv2.imread(self.real_path)
            image_2 = cv2.imread(self.dotted_path)
            img1 = cv2.GaussianBlur(image_1, (5, 5), 0)
            plt.imshow(img1)
            # absolute difference between Train and Train Dotted
            image_3 = cv2.absdiff(image_1, image_2)
            mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)

            mask_1[mask_1 < 50] = 0
            mask_1[mask_1 > 0] = 255
            image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)

            # convert to grayscale to be accepted by skimage.feature.blob_log
            image_6 = np.max(image_4, axis=2)

            # detect blobs
            blobs = skimage.feature.blob_log(image_6, min_sigma=3, max_sigma=7, num_sigma=1, threshold=0.05)

            res = [None] * len(blobs)

            for i, blob in enumerate(blobs):
                # get the coordinates for each blob
                y, x, s = blob

                # get the color of the pixel from Train Dotted in the center of the blob
                b, g, R = img1[int(y)][int(x)][:]

                cls = UNKNOWN
                # decision tree to pick the class of the blob by looking at the color in Train Dotted
                if R > 225 and b < 25 and g < 25:  # RED
                    cls = RED
                elif R > 225 and b > 225 and g < 25:  # MAGENTA
                    cls = MAGENTA
                elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
                    cls = GREEN
                elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
                    cls = BLUE
                elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
                    cls = BROWN

                #if cls == UNKNOWN:
                 #   raise RuntimeError("Proposed dot could not be classified")

                res[i] = (x, y, cls)
            np.savetxt(self.coord_path, np.array(res), fmt='%d')

        # read file
        coord = np.loadtxt(self.coord_path, dtype=int)
        return coord


class Box(object):
    def __init__(self, x, y, w, h, contains=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.contains = contains  # True if box contains sea lion

if __name__ == '__main__':
    im = Image('42')
   # print(im.get_boxes().shape)
    print(im.get_coordinates().shape)