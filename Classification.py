import cv2
from icecream import ic
import cvzone as cv
import tensorflow

cam_status = False
ic.configureOutput(prefix="Debug | ")
cap = cv2.VideoCapture(0)
classifier = cv.Classifier("./model/model.h5", "./model/labels.txt")

"""
    cap: cap object give some properties about video that is capture.
    1. video open or not:-- cap.isOpened() -->True/False
    2. like Width,height,rate of frame and many more explore by your self :--
     cap.get() [it's take prop id as argument]
     Width:-- cap.get(3) --> width of frame
     Height:-- cap.get(4) --> height of frame 
"""


def camActivate():
    return f"[{datetime.datetime.now()}] Camera is Activate"


def camDeactivate():
    return f"[{datetime.datetime.now()}] Camera is Deactivate"


def debug():

    pass

fps_reader = cv.FPS()
while cap.isOpened():
    if cam_status == False:
        ic(camActivate())
        ic(cap.get(3), "Width")
        ic(cap.get(4), "Height")
        cam_status = True
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fps, img = fps_reader.updata(frame, pos=(50, 50))
    pred, index = classifier.getPrediction(frame)
    print(fps, pred, index)
    if ret:
        cv2.imshow("Live Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            ic(camDeactivate())
            break

cap.release()
ic("Camera-is-release")
cv2.destroyAllWindows()
ic("destroy-All-Windows")

