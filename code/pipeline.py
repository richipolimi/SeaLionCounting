from image import Image
import cv2
from itertools import combinations
import keras

class Pipeline(object):
	def __init__(self, classifier):
		""" Takes classifier and """
		self.classifier = classifier

	def evaluate_img(self, image_id):
		""" Generate all possible boxes for image and return sea lion count for box. """
		img = Image(image_id)
		boxes = img.get_boxes()
		positives = []
		for box in boxes:
			x, y, w, h = box
			sub_img = img[y: y + h, x: x + w]
			crop_img = cv2.resize(sub_img, shape)
			pattern = crop_img.ravel().astype(np.float64)/255
			output = self.classifier.predict(pattern)
			if output == 1:
				positives.append(box)

		blacklist = set()
		for combo in combinations(positives, 2):
			box1 = combo[0]
			box2 = combo[1]

			if (box1 in blacklist or box2 in blacklist):
				continue

			x1, y1, w1, h1 = box1
			x2, y2, w2, h2 = box2
			overlap_x = x1 + w1 - x2
			overlapy_y = y1 + h1 - y2
			overlap_area = overlap_x * overlapy_y
			if overlap_area > 0:
				overlap_norm = overlap_area/(w1*h1)
				if overlap_norm > 0.8:
					blacklist.add(box2)

		return positives

	def sea_lions_in_img(self, image_id):
		""" Counts the number of sea lions in image by looking at the dotted image. """
		pass

	def mse(self, dataset_path):
		""" Calculates mean squared error over images in dataset. """
		pass

if __name__ == "__main__":
	model = keras.models.load_model("../Model/2layers.mod")