from filter_box import *
from tqdm import tqdm
from image import Image
import os
import sys


def generate_sub_images(image_id, shape=(224, 224), display=False):
    """
    Given an image id generate a list containing the small images of the
    sea lions and a list containig just background. The number of images
    in each list is the same and equal to the number of sealions images
    generated.
    """
    image = Image(image_id)
    boxes = image.get_boxes()
    coords = image.get_coordinates()
    img = cv2.imread(image.real_path)

    boxes, _ = filter_by_size(boxes, 20, 100)
    #boxes_bg, _ = keep_n_dots(boxes, coords, n=0)
    boxes_bg, _ = keep_the_furthest(boxes, coords)
    boxes_sl, _ = keep_one_dot(boxes, coords, kids_allowed=True)
    boxes_sl, _ = keep_the_closest(boxes_sl, coords, n=4, max_dist=0.35)
    boxes_sl, _ = keep_according_color(boxes_sl, coords, n=1)

    sea_lion_list = []
    background_list = []

    for box in boxes_sl:
        x, y, w, h = box
        sub_img = img[y: y + h, x: x + w]
        crop_img = cv2.resize(sub_img, shape)
        sea_lion_list.append(crop_img)
        if display:
            cv2.imshow("Output", crop_img)
            cv2.waitKey(0)

    random_indices = np.random.permutation(len(boxes_bg))
    for i in random_indices[0:len(sea_lion_list)]:
        box = boxes_bg[i]
        x, y, w, h = box
        sub_img = img[y: y + h, x: x + w]
        crop_img = cv2.resize(sub_img, shape)
        background_list.append(crop_img)
        if display:
            cv2.imshow("Output", crop_img)
            cv2.waitKey(0)

    return sea_lion_list, background_list


def generate_dataset(dataset_name, list_image_ids, shape=(224, 224)):
    """
    Given a name of dataset, a list of image ids and the shape of the image
    generate a new dataset.
    """

    # generate the folder of the dataset if not present
    dataset_path = DATASET_DIR + dataset_name
    if os.path.exists(dataset_path):
        raise RuntimeError("Dataset Folder already present!")

    os.mkdir(dataset_path)

    # generate the file with the label and the matrix with the image
    file_path = dataset_path + '/' + dataset_name + '.csv'
    X_path = dataset_path + '/X'
    f = open(file_path, "w+")
    X = []

    # generate the images
    count = 0
    for image_id in tqdm(list_image_ids):
        positive_image, negative_images = generate_sub_images(image_id, shape)

        for img in positive_image:
            img_path = dataset_path + '/' + str(count) + '.jpg'
            cv2.imwrite(img_path, img)
            f.write("%d,%d\n" % (count, 1))
            X.append(img.ravel().astype(np.uint8))
            count += 1

        for img in negative_images:
            img_path = dataset_path + '/' + str(count) + '.jpg'
            cv2.imwrite(img_path, img)
            f.write("%d,%d\n" % (count, 0))
            X.append(img.ravel().astype(np.uint8))
            count += 1
    f.close()

    np.save(X_path, np.vstack(X))


def generate_boxes(image_name_list):
    """
    Given a list if integer representing the image_id generate all
    the possible boxes and store them into a file. If the file
    already exist does nothing.
    """
    for image_name in tqdm(image_name_list):
        im = Image(image_name)
        im.get_boxes()


def generate_coordinates(image_name_list):
    """
    Given a list of integers representing the image_id, generate all
    the possible coordinates and store them into a file. If the file
    already exists: do nothing.
    """

    for image_name in tqdm(image_name_list):
        im = Image(image_name)
        im.get_coordinates()


def image2file(dataset_name, list_image_ids, shape):
    """
    Given a dataset name generate a 1D vector for each image
    than store the matrix of vector in a file.
    """
    dataset_path = DATASET_DIR + dataset_name

    # generate the file with the label
    file_path = dataset_path + '/X'
    X = np.zeros(((len(list_image_ids)), shape[0] * shape[1] * 3),
        dtype=np.uint8)

    for i, image_id in tqdm(enumerate(list_image_ids)):
        img_path = dataset_path + '/' + str(image_id) + '.jpg'
        img = cv2.imread(img_path)
        X[i, :] = img.ravel()

    np.save(file_path, X)


def create_test_image(list_image_ids):
    """
    Copy the images in list_image_ids into the TEST folder
    """
    for img_id in list_image_ids:
        im = cv2.imread(Image(img_id).real_path)
        cv2.imwrite(TEST_DIR + str(img_id) + ".jpg", im)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Expected an argument: -c, -b, -i2f or -d")
    else:

        if sys.argv[1] == '-c':
            generate_coordinates(Image.get_dataset())

        elif sys.argv[1] == '-b':
            generate_boxes(Image.get_dataset())

        elif sys.argv[1] == '-i2f':
            if len(sys.argv) < 5:
                print("Expected 3 arguments: dataset_name, num_images, size")
            else:
                name = sys.argv[2]
                num_images = int(sys.argv[3])
                size = int(sys.argv[4])
                image2file(name, range(num_images), (size, size))

        elif sys.argv[1] == '-d':
            if len(sys.argv) < 5:
                print("Expected 3 arguments: dataset_name, num_images, size")
            else:
                name = sys.argv[2]
                num_images = int(sys.argv[3])
                size = int(sys.argv[4])
                generate_dataset(name, Image.get_dataset()[0:num_images], (size, size))
