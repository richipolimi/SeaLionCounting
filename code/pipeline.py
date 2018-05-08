class Pipeline(object):
	def __init__(self, classifier):
		""" Takes classifier and """
		self.classifier = classifier

	def evaluate_img(self, image_id):
		""" Generate all possible boxes for image and return sea lion count for box. """
		pass

	def sea_lions_in_img(self, image_id):
		""" Counts the number of sea lions in image by looking at the dotted image. """
		pass