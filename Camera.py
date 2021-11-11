import cv2 as cv
# import numpy as np

cap = cv.VideoCapture('http://10.81.90.1:8080/video')

while cap.isOpened():
    _, frame = cap.read()
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    cv.imshow("Live", img_hsv)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()