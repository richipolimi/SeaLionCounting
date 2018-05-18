import numpy as np
import cv2
from region_proposal import RegionProposal
from config import *
from image import GREEN

"""
Display all the coordinates on the image
"""


def display_coords(image_path, coords):
    img = cv2.imread(image_path)
    for cord in coords:
        x, y, _ = cord
        cv2.circle(img, (x, y), 5, (255, 0, 0), 1)
    cv2.imshow('Coordinates',img)
    cv2.waitKey(0)


"""
Remove all the boxes that are too small or too big
"""


def filter_by_size(rects, min_lenght, max_length):
        drop_list = []
        for i, rect in enumerate(rects):
            _, _, w, h = rect
            if w < min_lenght or w > max_length or h < min_lenght or h > max_length:
                drop_list.append(i)
        return np.delete(rects, drop_list, axis=0), rects[drop_list]


"""
Remove all the boxes that don't contain any point or that 
contain more that 1 points. Just one exception: if a box
contains one dot different from green (not a kid) any many
other green dot it is ok.
"""


def keep_one_dot(rects, coords, kids_allowed=False):
    delete_list = []
    for i, rect in enumerate(rects):
        count_adults = 0
        count_kids = 0
        for cord in coords:
            x, y, w, h = rect
            x0, y0, cls = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                if cls == GREEN:
                    count_kids += 1
                else:
                    count_adults += 1

        tot = count_kids + count_adults
        if kids_allowed:
            if tot == 0 or count_adults >=2 or (count_adults == 0 and count_kids >= 2):
                delete_list.append(i)
        else:
            if tot == 0 or tot >= 2:
                delete_list.append(i)
    return np.delete(rects, delete_list, axis=0), rects[delete_list]


"""
Return a list of boxes that contain exactly n dots.
"""


def keep_n_dots(rects, coords, n):
    save_list = []
    for i, rect in enumerate(rects):
        count = 0
        for cord in coords:
            x, y, w, h = rect
            x0, y0, cls = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                count += 1

        if count == n:
            save_list.append(i)
    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)


"""
For any point keep only n boxes, the ones which center is closest to
the point
"""


def keep_the_closest(rects, coords, max_dist=0.4, n=1):
    save_list = np.array([], dtype=np.uint32)
    for cord in coords:
        closest = np.array([], dtype=np.uint32)
        distance = np.array([])

        for i, rect in enumerate(rects):
            x, y, w, h = rect
            x0, y0, _ = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                center_x = x + w/2.0
                center_y = y + h/2.0
                dist = np.sqrt((center_x-x0)**2 + (center_y-y0)**2)

                if dist < max_dist * np.min([w,h]):
                    closest = np.append(closest, i)
                    distance = np.append(distance, dist)

        if closest.size > 0:
            rank = np.argsort(distance)
            save_list = np.hstack([save_list, closest[rank][0:n]])

    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)


"""
For any point keep only the n biggest boxes
"""


def keep_the_biggest(rects, coords, n=1):
    save_list = np.array([], dtype=np.uint32)
    for cord in coords:
        biggest = np.array([], dtype=np.uint32)
        size = np.array([])

        for i, rect in enumerate(rects):
            x, y, w, h = rect
            x0, y0, _ = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                biggest = np.append(biggest, i)
                size = np.append(size, w * h)
        if biggest.size > 0:
            rank = np.argsort(size)
            save_list = np.hstack([save_list, biggest[rank][-n:]])

    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)


"""
For any point keep only the n smallest boxes
"""


def keep_the_smallest(rects, coords, n=1):
    save_list = np.array([], dtype=np.uint32)
    for cord in coords:
        closest = np.array([], dtype=np.uint32)
        size = np.array([])

        for i, rect in enumerate(rects):
            x, y, w, h = rect
            x0, y0, _ = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                closest = np.append(closest, i)
                size = np.append(size, w * h)

        if closest.size > 0:
            rank = np.argsort(size)
            save_list = np.hstack([save_list, closest[rank][:n]])
    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)


"""
Keep the smaller for the kids, keep the
bigger for the adults
"""


def keep_according_color(rects, coords, n=1):
    save_list = np.array([], dtype=np.uint32)
    for cord in coords:
        boxes = np.array([], dtype=np.uint32)
        size = np.array([])

        for i, rect in enumerate(rects):
            x, y, w, h = rect
            x0, y0, cls = cord

            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                boxes = np.append(boxes, i)
                size = np.append(size, w * h)

        if boxes.size > 0:
            rank = np.argsort(size)
            if cls == GREEN:
                save_list = np.hstack([save_list, boxes[rank][:n]])
            else:
                save_list = np.hstack([save_list, boxes[rank][-n:]])

    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)



def keep_the_furthest(rects, coords, min_dist=150, stop_at=50):
    """
    Keep only the boxes that are distance from any dot.
    """

    save_list = np.array([], dtype=np.uint32)
    for i, rect in enumerate(rects):
        distance = np.array([])

        for cord in coords:
            x, y, w, h = rect
            x0, y0, _ = cord

            center_x = x + w/2.0
            center_y = y + h/2.0
            dist = np.sqrt((center_x-x0)**2 + (center_y-y0)**2)

            distance = np.append(distance, dist)

        if np.min(distance) > min_dist:
            save_list = np.append(save_list, i)

        if save_list.size >= stop_at:
            break

    save_list = np.unique(save_list)
    return rects[save_list], np.delete(rects, save_list, axis=0)

def center(rects, coords):
    """
    return a new set of rectangles that is placed in the
    center of the dot
    """

    rects = rects.copy()
    for i, rect in enumerate(rects):
        for cord in coords:
            x, y, w, h = rect
            x0, y0, _ = cord
            if x0 > x and x0 < x + w and y0 > y and y0 < y + h:
                rects[i, 0] = x0 - int(w / 2)
                rects[i, 1] = y0 - int(h / 2)
    return rects

if __name__ == '__main__':
    image_path = TRAIN_DIR + "9999.jpg"
    boxes_path = BOXES_DIR + "9999.box"
    coord_path = COORDINATES_DIR + "9999.coor"

    rects = np.loadtxt(boxes_path, dtype=int)
    coords = np.loadtxt(coord_path, dtype=int)

    rp = RegionProposal()

    #rp.display(image_path, rects)

    rects, _ = filter_by_size(rects, 20, 100)
    zero_dots, _ = keep_the_furthest(rects, coords)
    rects, _ = keep_one_dot(rects, coords, kids_allowed=True)

    #rp.display(image_path, rects, n=len(rects))
    rects, _ = keep_the_closest(rects, coords, n=4, max_dist=0.35)

    #rp.display(image_path, rects, n=len(rects))
    rects, _ = keep_according_color(rects, coords, n=1)

    #rects, _  = keep_the_smallest(rects, coords, n=1)

    rp.display(image_path, rects, n=len(rects))
    rp.display(image_path, zero_dots, n=len(rects))