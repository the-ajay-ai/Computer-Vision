import cv2 as cv

# img = cv.imread(r'C:\Users\AJAY\PycharmProjects\ComputerVision\data\ajay.jpg')

cap = cv.VideoCapture('http://10.81.90.1:8080/video')

# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter("output.mp4", fourcc, 20, (640, 480))

face_feature = r'C:\Users\AJAY\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml'
face_detector = cv.CascadeClassifier(face_feature)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # print("Frame read Successful")
        # out.write(frame)
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, 1.2, 5)
        if len(faces) > 0:
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), [55, 155, 255], 2)
        cv.imshow("Live", frame)
        if cv.waitKey(1) == 27:
            break
    else:
        break
cap.release()
# out.release()
cv.destroyAllWindows()


