import cv2 as cv
import numpy as np


def nothing(x):
    pass


img = np.zeros((512, 512, 3), np.uint8)
img[:] = 255
cv.namedWindow("White_Board", 0)

cv.createTrackbar("R", "White_Board", 0, 255, nothing)
cv.createTrackbar("G", "White_Board", 0, 255, nothing)
cv.createTrackbar("B", "White_Board", 0, 255, nothing)
cv.createTrackbar("Pencil_Size", "White_Board", 1, 28, nothing)
draw = False
eraser = False
count = 0

def paint(event, x, y, flags, param):
    global draw, eraser
    if event == cv.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv.EVENT_LBUTTONUP:
        draw = False
    elif event == cv.EVENT_MOUSEMOVE:
        if draw:
            r = cv.getTrackbarPos("R", "White_Board")
            g = cv.getTrackbarPos("G", "White_Board")
            b = cv.getTrackbarPos("B", "White_Board")
            cv.circle(img, (x, y), cv.getTrackbarPos("Pencil_Size", "White_Board"), [r, g, b], -1)
        elif eraser:
            cv.circle(img, (x, y), cv.getTrackbarPos("Pencil_Size", "White_Board"), [255, 255, 255], -1)
    elif event == cv.EVENT_RBUTTONDOWN:
        eraser = True
    elif event == cv.EVENT_RBUTTONUP:
        eraser = False
    elif event == cv.EVENT_RBUTTONDBLCLK:
        img[:] = 255


cv.setMouseCallback("White_Board", paint)


while True:
    cv.imshow("White_Board", img)
    if cv.waitKey(1) == 27:
        break
    if cv.waitKey(1) == ord('s'):
        cv.imwrite(f'C:/Users/AJAY/PycharmProjects/ComputerVision/paint/Untitled{count}.png', img)
        count = count + 1
cv.destroyAllWindows()
