import streamlit as st
import urllib
import numpy as np
from datetime import datetime
import glob
import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.50):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                            (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

def img_data():
    cols = st.beta_columns(4)
    if cols[0].button("Car-in"):
        img_path = glob.glob("./Test/car_in/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[1].button("Car-out"):
        img_path = glob.glob("./Test/car_out/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[2].button("Bike-in"):
        img_path = glob.glob("./Test/bi_in/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[3].button("Bike-out"):
        img_path = glob.glob("./Test/bi_out/image/*.jpg")
        st.image(img_path, width=100, clamp=True)


logo, times = st.beta_columns(2)
st.image("./default_image/cu.jpeg")
logo.image("./default_image/cur.png")

# logo.image("./default_image/curaj1.png",)
st.sidebar.title("VI-SDT-TL Application in Security Surveillance")
st.sidebar.image("./default_image/images.jpeg", use_column_width=True)
workflow = st.sidebar.selectbox("Let's Go!", ["Welcome", "About-Project", "Project-Quick-Recap", "DataSet", "Work-Flow", "Reference", "Requirements"])


if workflow == "Welcome":
    st.balloons()
    st.image("./default_image/img.jpeg")
    img_path = [f"./default_image/{i}.jpg" for i in range(1, 3)]
    st.image(img_path, use_column_width=True)


if workflow == "About-Project":
    # FRAME_WINDOW = st.image([])
    img_path = [f"./Project_Presentation/{i}.jpg" for i in range(1, 13)]
    st.image(img_path, use_column_width=True)

if workflow == "Project-Quick-Recap":
    st.info("Coming Soon")

if workflow == "DataSet":
    img_data()

if workflow == "Reference":
    st.text("""
            1. H. Karwal and A. Girdhar, "Vehicle Number Plate Detection System for Indian Vehicles," 
            2015 IEEE International Conference on Computational Intelligence & Communication Technology, 2015, 
            pp. 8-12,DOI: 10.1109/CICT.2015.13
            
            2. Z. Jian, Z. Yonghui, Y. Yan, L. Ruonan and W. Xueyao, 
            "MobileNet-SSD with the adaptive expansion of the receptive field," 
            2020 IEEE 3rd International Conference of Safe Production and Informatization (IICSPI), 2020, 
            pp. 177-181, DOI: 10.1109/IICSPI51290.2020.9332204.
            
            3. Chengcheng Ning, Huajun Zhou, Yan Song, and Jinhui Tang, 
            "Inception Single Shot MultiBox Detector for object detection," 
            2017 IEEE International Conference on Multimedia & Expo Workshops (ICMEW), 2017, pp. 549-554,
            DOI: 10.1109/ICMEW.2017.8026312.
            
            4. Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet, and YOLOv3) 
            By: Jonathan Hui (Medium Article)

    """)

if workflow == "Requirements":
    st.info("Updated Soon")

if workflow == "Work-Flow":
    FRAME_WINDOW = st.image([])
    check = st.sidebar.radio("Select Action", ["Live Feed", "Image Feed"])
    in_time = []
    out_time = []
    times = times.empty()
    if check == "Live Feed":
        Active, Deactive = st.beta_columns(2)
        # df = pd.read_csv("")
        active = st.radio("Camera", ["OFF", "Camera-ON"])
        camera0 = cv2.VideoCapture(0)
        camera1 = r"http://10.80.70.209:8080/shot.jpg"
        if active == "Camera-ON":
            ACTIVE = Active.selectbox("Camera-ID", ["Default Cam", "Gate No 3", "Video-File", "Canteen", "MegaMess"])
            if ACTIVE == "Default Cam":
                in_time.append(str(datetime.now()))
                pTime = 0
                detector = FaceDetector()
                while True:
                    times.text("Date & Time: " + str(datetime.now()))
                    _, frame0 = camera0.read()
                    # _, frame1 = camera1.read()
                    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                    # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    img = detector.findFaces(frame0)
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(frame0, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

                    # FRAME_WINDOW.image([frame0,frame0], width = 300,use_column_width=True)
                    FRAME_WINDOW.image([frame0], width = 300, use_column_width=True)

            if ACTIVE == "Gate No 3":
                in_time.append(str(datetime.now()))
                while True:
                    imgResponse = urllib.request.urlopen(camera1)
                    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    frame1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame1, channels='RGB')

            if ACTIVE == "Video-File":
                while True:
                    uploadFile = st.file_uploader("Browser Files", type=[".mp4", ".avi"])
                    st.video(uploadFile)

        if active == "OFF":
            camera0.release()
            out_time.append(str(datetime.now()))
            print(in_time, out_time)
            # df = pd.DataFrame({"In-time": in_time, "Out-time": out_time})
            # st.table(df)
            # df.to_csv("default.csv")

    if check == "Image Feed":
        uploadFile = st.sidebar.file_uploader("Browser Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploadFile is not None:
            image = []
            for img in uploadFile:
                file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
                image.append(cv2.imdecode(file_bytes, 1))
            # st.write("Image Uploaded Successfully")
            FRAME_WINDOW.image(image, width=100, clamp=True)
        else:
            st.write("Make sure you image is in JPG/PNG Format.")





# st.title("VI-SDT-ObjectDetection")
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)
#
#
# def welcome():
#     st.balloons()
#     st.title('Image Processing using Streamlit')
#
#     st.subheader('A simple app that shows different image processing algorithms. You can choose the options'
#                  + ' from the left. I have implemented only a few to show how it works on Streamlit. ' +
#                  'You are free to add stuff to this app.')
#
#     # st.image('hackershrine.jpg', use_column_width=True)
#
#
# def video():
#     while st.checkbox("Webcam Live Feed"):
#         _, frame = camera.read()
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)
#     else:
#         camera.release()
#         st.write('Stopped')
#
#
# def photo():
#     st.header('Image Processing')
#     uploaded_file = st.file_uploader(label="Upload image", type=['jpg', 'jpeg', 'png'])
#     if uploaded_file is not None:
#         # Convert the file to an opencv image.
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         opencv_image = cv2.imdecode(file_bytes, 1)
#         # Now do something with the image! For example, let's display it:
#         st.image(opencv_image, channels="BGR")
#         st.image(cv2.resize(opencv_image,(225,225)))
#
#
# def face_detection():
#     st.header("Face Detection using haarcascade")
#
#     if st.button('See Original Image'):
#         original = Image.open('friends.jpeg')
#         st.image(original, use_column_width=True)
#
#     image2 = cv2.imread("friends.jpeg")
#
#     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     faces = face_cascade.detectMultiScale(image2)
#     print(f"{len(faces)} faces detected in the image.")
#     for x, y, width, height in faces:
#         cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
#
#     cv2.imwrite("faces.jpg", image2)
#
#     st.image(image2, use_column_width=True, clamp=True)
#
# def face_detection():
#     st.header("Face Detection using haarcascade")
#     image2 = cv2.imread("friends.jpeg")
#     face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#     faces = face_cascade.detectMultiScale(image2)
#     print(f"{len(faces)} faces detected in the image.")
#     for x, y, width, height in faces:
#         cv2.rectangle(image2, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)
#
#     cv2.imwrite("faces.jpg", image2)
#
#     st.image(image2, use_column_width=True, clamp=True)
#
#

def img_data():
    cols = st.beta_columns(4)
    if cols[0].button("Car-in"):
        img_path = glob.glob("/Users/aj/Downloads/Test/car_in/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[1].button("Car-out"):
        img_path = glob.glob("/Users/aj/Downloads/Test/car_out/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[2].button("Bike-in"):
        img_path = glob.glob("/Users/aj/Downloads/Test/bi_in/image/*.jpg")
        st.image(img_path, width=100, clamp=True)
    if cols[3].button("Bike-out"):
        img_path = glob.glob("/Users/aj/Downloads/Test/bi_out/image/*.jpg")
        st.image(img_path, width=100, clamp=True)


# def vid_data():
#     pass
#
#
# def main():
#     selected_box = st.sidebar.selectbox('Choose one of the following', (
#     'Welcome', "Image DataSet", "Video DataSet", 'Image Processing', 'Video Processing', 'Face Detection', 'Feature Detection',
#     'Object Detection'))
#
#     if selected_box == 'Welcome':
#         welcome()
#     if selected_box == "Image DataSet":
#         img_data()
#     if selected_box == "Video DataSet":
#         vid_data()
#     if selected_box == 'Image Processing':
#         photo()
#     if selected_box == 'Video Processing':
#         video()
#     if selected_box == 'Face Detection':
#         face_detection()
#     if selected_box == 'Feature Detection':
#         feature_detection()
#     if selected_box == 'Object Detection':
#         object_detection()
#
#
# if __name__ == "__main__":
#     main()
