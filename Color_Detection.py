import cv2 as cv
import numpy as np
img_path = r"/Users/aj/Desktop/WhatsApp/1.jpeg"

img = cv.imread(img_path)

def nothing(x):
    pass

cv.namedWindow("Color_Detection")
cv.createTrackbar("low_H", "Color_Detection",0,179,nothing)
cv.createTrackbar("low_S", "Color_Detection",0,255,nothing)
cv.createTrackbar("low_V", "Color_Detection",0,255,nothing)

cv.createTrackbar("hig_H", "Color_Detection",0,179,nothing)
cv.createTrackbar("hig_S", "Color_Detection",0,255,nothing)
cv.createTrackbar("hig_V", "Color_Detection",0,255,nothing)

img_hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)

while True:
    low_H, low_S, low_V  = cv.getTrackbarPos("low_H","Color_Detection"),\
                           cv.getTrackbarPos("low_S","Color_Detection"),\
                           cv.getTrackbarPos("low_V","Color_Detection")
    hig_H, hig_S, hig_V = cv.getTrackbarPos("hig_H","Color_Detection"),\
                           cv.getTrackbarPos("hig_S","Color_Detection"),\
                           cv.getTrackbarPos("hig_V","Color_Detection")
    lower = np.array([low_H, low_S, low_V])
    higher = np.array([hig_H, hig_S, hig_V])
    mask  = cv.inRange(img_hsv, lower, higher)
    final = cv.bitwise_and(img, img, mask = mask)

    cv.imshow("Original", img)
    cv.imshow("Mask", mask)
    cv.imshow("Result", final)

    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()