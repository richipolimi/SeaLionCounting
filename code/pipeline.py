from image import Image
import cv2
from itertools import combinations
import keras
import numpy as np
from filter_box import filter_by_size
from region_proposal import RegionProposal
from config import *

import os


class Pipeline(object):
    def __init__(self, classifier):
        """ Takes classifier and """
        self.classifier = classifier

    def evaluate_img(self, image, shape):
        """
        Generate all possible boxes for image and
        return boxes for image that contain a sea lion.
        """

        image = cv2.imread(img.real_path)
        boxes = image.get_boxes()
        positives = []

        boxes, _ = filter_by_size(boxes, 30, 100)

        for box in boxes:
            x, y, w, h = box
            sub_img = image[y: y + h, x: x + w]
            crop_img = cv2.resize(sub_img, shape)
            pattern = crop_img.ravel().astype(np.float64) / 255
            output = self.classifier.predict(pattern[None])
            if output > 0.5:
                positives.append(box)

        blacklist = set()
        for combo in combinations(range(len(positives)), 2):
            box1 = positives[combo[0]]
            box2 = positives[combo[1]]

            # check if one box is contained in the other

            if (combo[0] in blacklist or combo[1] in blacklist):
                continue

            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            overlap_area = overlap_x * overlap_y

            if overlap_area > 0:
                overlap_norm = overlap_area / (w1 * h1)
                if overlap_norm > 0.9:
                    blacklist.add(combo[1])
                else:
                    overlap_norm = overlap_area / (w2 * h2)
                    if overlap_norm > 0.9:
                        blacklist.add(combo[0])
        result = []
        for i, positive in enumerate(positives):
            if i not in blacklist:
                result.append(positive)

        return result

    def sea_lions_in_img(self, image):
        """
        Counts the number of sea lions in image
        by looking at the dotted image.
        Args:
            image: Image instance
            dataset: "TRAIN" or "TEST"
        """
        
        coords = img.get_coordinates() 
        no_of_lions = 0
        for coord in coords:
            _, _, dot_class = coord
            if not dot_class == "GREEN":
                no_of_lions += 1
        return no_of_lions

    def mse(self, dataset, shape = (50, 50)):
        """ 
        Calculates mean squared error over images in dataset. 
        Args:
            dataset: "TEST" or "TRAIN"
        """
        sum_squares = 0
        images = 0
        if dataset == "TEST":
            dataset_path = TEST_DIR
        elif dataset == "TRAIN":
            dataset_path = TRAIN_DIR
        # For each image ID in dataset
        for file in os.scandir(dataset_path):
            image_id, ext = os.path.splitext(file.name)
            if (ext == ".jpg"):
                img = Image(image_id, dataset)
                images += 1
                # Compare sum for image with sea_lions_in_img() result. That is the error
                output_count = len(self.evaluate_img(img, shape))
                target_count = self.sea_lions_in_img(img, dataset)
                error = output_count - target_count
                # Square the error and add to total
                squared_error = error ** 2
                sum_squares += squared_error

        # Divide by the number of image IDs
        mse = sum_squares / images
        return mse

if __name__ == "__main__":
    rp = RegionProposal()
    model = keras.models.load_model("../Model/2layers.mod")
    pipeline = Pipeline(model)
    positives = pipeline.evaluate_img(41, (50, 50))
    rp.display(Image(41).real_path, np.vstack(positives), n=len(positives))
    #pipeline.mse("../TrainSmall2/Train")
