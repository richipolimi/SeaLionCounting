import cv2
import numpy as np
import skimage.feature
from config import *
from pathlib import Path
from region_proposal import RegionProposal
import os
from os.path import basename, splitext

UNKNOWN = 0
RED = 1
MAGENTA = 2
GREEN = 3
BLUE = 4
BROWN = 5

class Image(object):

    rp = RegionProposal()
    MismatchedImg_file_path = DATASET_DIR + "MismatchedTrainImages.txt"

    def __init__(self, image_id, dataset):
        """
        Args:
            image_id: ID of the image.
            dataset: "TRAIN" or "TEST", determines which
            folder to look inside.
        """
        self.image_id = image_id

        if dataset == "TRAIN":
            real_dir = TRAIN_DIR
            dotted_dir = TRAIN_DOTTED_DIR
        elif dataset == "TEST":
            real_dir = TEST_DIR
            dotted_dir = TEST_DOTTED_DIR
        else:
            raise ValueError("dataset argument can only be 'TRAIN' or 'TEST'")

        self.real_path = real_dir + str(image_id) + ".jpg"  # Path to the non-dotted training image
        self.dotted_path = dotted_dir + str(image_id) + ".jpg"  # Path to the dotted training image
        self.boxes_path = BOXES_DIR + str(image_id) + '.box'  # Path to the boxes file
        self.coord_path = COORDINATES_DIR + str(image_id) + ".coor"  # Path to the coordinates file

    
    def get_boxes(self):
        """ 
        Generates a list of box vectors from non-dotted training image. 
        Returns:
            # TODO: show structure of returned numpy array here.
        """

        # generate boxes if they are not already present in a file
        boxes_file = Path(self.boxes_path)
        if not boxes_file.is_file():
            rects = Image.rp.generate_boxes(self.real_path)
            np.savetxt(self.boxes_path, rects, fmt='%d')

        # read file
        boxes = np.loadtxt(self.boxes_path, dtype=int)
        return boxes


    def get_coordinates(self):
        """ Generate a list of coordinate pairs from dotted image """
    
        # generate boxes if they are not already present in a file
        coord_file = Path(self.coord_path)
        if not coord_file.is_file():
            # read the Train and Train Dotted images
            image_1 = cv2.imread(self.dotted_path)
            image_2 = cv2.imread(self.real_path)
            img1 = cv2.GaussianBlur(image_1, (5, 5), 0)

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

            res = []

            for i, blob in enumerate(blobs):
                # get the coordinates for each blob
                y, x, s = blob
                y = int(y)
                x = int(x)

                # get the color of the pixel from Train Dotted in the center of the blob
                b, g, R = img1[y][x][:]

                dot_class = UNKNOWN
                # decision tree to pick the class of the blob by looking at the color in Train Dotted
                if R > 210 and b < 25 and g < 25:  # RED
                    dot_class = RED
                elif R > 210 and b > 225 and g < 25:  # MAGENTA
                    dot_class = MAGENTA
                elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
                    dot_class = GREEN
                elif R < 75 and 140 < b < 210 and g < 75:  # BLUE
                    dot_class = BLUE
                elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
                    dot_class = BROWN

                if dot_class == UNKNOWN:
                    print("Proposed dot could not be classified in image %d" % self.image_id)
                    #raise RuntimeError("Proposed dot could not be classified")
                else:
                    res.append([x, y, dot_class])

            np.savetxt(self.coord_path, np.array(res), fmt='%d')

        # read file
        coord = np.loadtxt(self.coord_path, dtype=int)
        if coord.ndim == 1 and coord.size != 0:
            coord = np.expand_dims(coord, axis=0)
        return coord

    """
    Return a list of image_id taken from the train folder
    and without the mismatched images.
    """
    @staticmethod
    def get_dataset():
        all_files = os.listdir(TRAIN_DIR)
        all_imgs = np.array([], dtype=int)
        mismatched = np.loadtxt(Image.MismatchedImg_file_path, dtype=int)

        for img in all_files:
            base_name = splitext(basename(img))[0]
            ext = splitext(basename(img))[-1]
            if ext == '.jpg' and base_name.isdigit():
                all_imgs = np.append(all_imgs, int(base_name))

        return np.sort(np.setdiff1d(all_imgs, mismatched))

if __name__ == '__main__':
    im = Image('42', "TRAIN")
    print(im.get_coordinates().shape)
