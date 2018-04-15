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
    def generate_coordinates(self):
        """
        Generate a list of coordinate pairs from dotted training image
        """
        return []

class Box(object):
    def __init__(self, x1, y1, x2, y2, contains = False):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.contains = contains # True if box contains sea lion
