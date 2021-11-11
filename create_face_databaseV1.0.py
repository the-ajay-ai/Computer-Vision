import os
import cv2 as cv
import pathlib

# import matplotlib.pyplot as plt

print("Please Wait...")
print("Loading.....")
path_ = input("Enter the image folder path:\n")
print("Face Database is preparing...")


def get_photo_dir(dir_path):
    image_dir = list()
    images_name = os.listdir(dir_path)
    for i in images_name:
        if i.endswith(".jpg") or i.endswith(".jpeg") or i.endswith(".png"):
            image_dir.append("".join([dir_path + "\\" + i]))
    return images_name, image_dir


def face_detect(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    return faces


def create_dataset(dir_path):
    images_name, image_dir = get_photo_dir(dir_path)
    home = pathlib.Path.home()
    i = 0
    for pic, name in zip(image_dir, images_name):
        name = "{}\Downloads\\face_dataset\\face_".format(home)
        img = cv.imread(pic)
        img_gray = cv.imread(pic, 0)  # Gray Scale Images
        faces = face_detect(img_gray)
        # print(len(faces))
        for x, y, w, h in faces:
            # cv.rectangle(img, (x, y), (x + w, y + h), [0, 0, 0], 2)
            i += 1
            crop_img = img[y:y + h, x:x + w]
            location = name + str(i) + '.jpg'
            if 'face_dataset' in os.listdir("{}\Downloads".format(home)):
                cv.imwrite(location, crop_img)
            else:
                os.mkdir("{}\Downloads\\face_dataset".format(home))
                cv.imwrite(location, crop_img)
        # plt.imshow(img, cmap='gray')
        # plt.show()
    print("Your DataBase is Ready!...")
    print("Your Database directory name is face_dataset")
    print("Please check into your Downloads Folder")


create_dataset(path_)
