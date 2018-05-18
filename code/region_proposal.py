import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from config import *

"""
A Region Proposal object that use selective search to generate a set of
boxes given an image.
"""
class RegionProposal(object):

    """
    Initilize the object. the mode attribute refers to the selective
    search and can be either 'quality' or 'fast'
    """
    def __init__(self, mode='quality', n_threads=1, chunk_width=724, chunk_height=516, overlapping=100):
        self.init_open_cv(n_threads)
        self.mode = mode
        self.chunk_width = chunk_width
        self.chunk_height = chunk_height
        self.overlapping = overlapping

    """
    Initialize OpenCV
    """
    def init_open_cv(self, n_thread=1):
        # speed-up using multithreads
        cv2.setUseOptimized(True)
        cv2.setNumThreads(n_thread)


    """
    Given an image generates a grid containing x, y, w, h of partial overlapping sub images
    """
    def generate_patch(self, img, patch_width, patch_height, overlapping):
        img_width, img_height = img.shape[1], img.shape[0]
        patches = np.zeros((0,4), dtype=int)
        for y in range(0, img_height - overlapping, patch_height - overlapping):
            for x in range(0, img_width - overlapping , patch_width - overlapping):
                width = np.min([patch_width, img_width - x])
                height = np.min([patch_height, img_height - y])
                patches = np.vstack([patches, [x, y, width, height]])
        return patches

    """
    Given an image generates a an array of partial overlapping sub images
    """
    def generate_sub_images(self, img):
        images = []
        position = []
        grid = self.generate_patch(img, self.chunk_width, self.chunk_height, self.overlapping)
        for coord in grid:
            x, y, w, h = coord
            sub_img = img[y: y + h, x: x + w]
            images.append(sub_img)
            position.append((x,y))
        return images, position

    """
    Given an image path returns a numpy matrix (4 x n) containing
    all boxes found in that image. Every box is represented by a 
    row containing 4 int number that are: x, y, width, height 
    """
    def generate_boxes(self, image_path):
        # Read image
        im = cv2.imread(image_path)

        if(im.shape[0] > self.chunk_height and im.shape[1] > self.chunk_width):
            images, position = self.generate_sub_images(im)

            n_cores = cpu_count()
            pool = Pool(processes=n_cores)
            rects_list = pool.map(RegionProposal.compute_chunk, zip(images, position))
            pool.close()
            pool.join()
            pool.terminate()
            rects = np.vstack(rects_list)
            return rects
        else:
            rects = RegionProposal.compute_chunk((im, (0,0)))
        return rects

    """
    Generate the boxes for a given image chunk. The imput is a pair containing
    the image and the coordinate of the chunk. 
    """
    def compute_chunk(im_position, mode='quality'):
        im = im_position[0]
        pos = im_position[1]

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # set input image on which we will run segmentation
        ss.setBaseImage(im)

        # configure the selection search
        if mode == 'quality':
            ss.switchToSelectiveSearchQuality()
        elif mode == 'fast':
            ss.switchToSelectiveSearchFast()

        rects = ss.process()
        rects[:, 0] += pos[0]
        rects[:, 1] += pos[1]
        return rects

    """
    Given a matrix containing info about boxes makes every box a square
    centered in the same position of the original box.
    """
    def to_square(self, rects, edge_length):
        squares = rects.copy()
        for i, rect in enumerate(rects):
            x, y, w, h = rect
            x1 = int(x + w / 2 - edge_length / 2)
            y1 = int(y + h / 2 - edge_length / 2)
            square = np.array([x1, y1, edge_length, edge_length])
            squares[i, :] = square
        return squares

    """
    Display all the boxes given an image and a numpy matrix that contains
    the information about the boxex. It display just the first n boxes,
    to show the next n boxes press m, to show the previus n boxex pree l,
    to quit press q.
    """
    def display(self, image_path, rects, n=10):
        im = cv2.imread(image_path)

        low_index = 0
        high_index = n
        increment = n

        while True:
            # create a copy of original image
            imOut = im.copy()

            # itereate over all the region proposals
            for rect in rects[low_index:high_index, :]:
                # draw rectangle
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

            # show output
            cv2.imshow("Output", imOut)

            # record key press
            k = cv2.waitKey(0) & 0xFF

            # m is pressed
            if k == 109:
                high_index += increment
                low_index += increment
            # l is pressed
            elif k == 108:
                high_index -= increment
                low_index -= increment
            # q is pressed
            elif k == 113:
                break

        # close image show window
        cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = TRAIN_DIR + "9999.jpg"
    boxes_path = BOXES_DIR + "9999.box"
    rp = RegionProposal()
    rects = rp.generate_boxes(image_path)
    np.savetxt(boxes_path, rects, fmt='%d')
    rp.display(image_path, rects)