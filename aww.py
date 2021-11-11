import cv2 as cv
import numpy as np

drawing = False
eraser = False


def paint(event, x, y, flags, param):
    global drawing, eraser
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        cv.circle(aww, (x, y), 10, [55, 155, 255], -1)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            cv.circle(aww, (x, y), 10, [55, 155, 255], -1)
        elif eraser:
            cv.circle(aww, (x, y), 12, [0, 0, 0], -1)
    elif event == cv.EVENT_LBUTTONDBLCLK:
        aww[:] = 1
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv.EVENT_RBUTTONDOWN:
        eraser = True
        cv.circle(aww, (x, y), 12, [0, 0, 0], -1)
    elif event == cv.EVENT_RBUTTONUP:
        eraser = False


aww = np.zeros((360, 360, 3), np.uint8)
cv.namedWindow("WhiteBoard")
cv.setMouseCallback("WhiteBoard", paint)

while True:
    cv.imshow("WhiteBoard", aww)
    if cv.waitKey(1) == 27:
        break
cv.destroyAllWindows()
