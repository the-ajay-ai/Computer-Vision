import cv2
import numpy as np
import pytesseract
path = r"Test/car_in/image/HR26CL8605.jpg"
read = ""

def extract_plate(path):
    global read
    # the function detects and perfors blurring on the number plate.
    # plate_img = cv2.resize(cv2.imread(path, 0),(512,512))
    img = cv2.imread(path)
    # print(plate_img)
    # Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_licence_plate_rus_16stages.xml')

    # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    # plate_rect = plate_cascade.detectMultiScale(plate_img, 1.4, 3)
    # print(plate_rect)
    # cv2.imshow('img', plate_img)
    # cv2.waitKey(0)

    plate = None
    for (x, y, w, h) in plate_rect:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))  # parameter tuning
        plate = img[y + a:y + h, x + b:x + w, :]

        #IMAGE PROCESSING
        kernel = np.ones((1, 1), np.uint8)
        # plate = cv2.dilate(plate, kernel, iterations=1)
        # plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (_, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        read = pytesseract.image_to_string(plate, lang='eng')
        read = ''.join(e for e in read if e.isalnum())
        # state = read[0:2]


        # finally representing the detected contours by drawing rectangles around the edges.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img, (x, y-40), (x + w, y), (0, 255, 255), 3)
        cv2.putText(img, read, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    # return plate_img, plate  # returning the processed image


extract_plate(path)
# plate_img = cv2.imread(path)
# plate_img, plate = extract_plate(path)
# print(plate_img,plate)
#
# cv2.imshow('img', plate)
# cv2.waitKey(0)
