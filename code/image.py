import cv2
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt

class Image(object):
	def __init__(self, real_path, dotted_path):
		self.real_path = real_path # Path to the non-dotted training image
		self.dotted_path = dotted_path # Path to the dotted training image
		self.boxes = [] # Contains Box instances

	def generate_boxes(self):
		""" 
		Generates a list of Box instances from non-dotted training image
		"""
		return []

	def generate_coordinates(self, filename):
		"""
		Generate a list of coordinate pairs from dotted training image.
		Coordinates are relative to the top left of the image.
		"""
		# read the Train and Train Dotted images
		image_1 = cv2.imread(self.real_path + filename)
		image_2 = cv2.imread(self.dotted_path + filename)
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
		print(blobs)
		res = [None] * len(blobs)

		for i, blob in enumerate(blobs):
			# get the coordinates for each blob
			y, x, s = blob
			# get the color of the pixel from Train Dotted in the center of the blob
			b, g, R = img1[int(y)][int(x)][:]
			cls = "unknown"
			# decision tree to pick the class of the blob by looking at the color in Train Dotted
			if R > 225 and b < 25 and g < 25:  # RED
				cls = "RED"
			elif R > 225 and b > 225 and g < 25:  # MAGENTA
				cls = "MAGENTA"
			elif R < 75 and b < 50 and 150 < g < 200:  # GREEN
				cls = "GREEN"
			elif R < 75 and 150 < b < 200 and g < 75:  # BLUE
				cls = "BLUE"
			elif 60 < R < 120 and b < 50 and g < 75:  # BROWN
				cls = "BROWN"

			if cls == "unknown":
				raise RuntimeError("Proposed dot could not be classified")

			res[i] = (x, y, cls)

		return res


class Box(object):
	def __init__(self, x1, y1, x2, y2, contains = False):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.contains = contains # True if box contains sea lion

