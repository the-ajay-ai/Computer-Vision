import streamlit as st
import urllib
from datetime import datetime
import glob
import time
import pandas as pd
import tempfile
import streamlit as st
import cvlib as cv
# from cvlib.object_detection import draw_bbox
import cv2
import tensorflow as tf
import tensorflow.keras
import numpy as np
from twilio.rest import Client
from csv import DictWriter

def sendInfo(data, mode):
    if mode == "SMS":
        from_whatsapp_number = '+16179368054'
        to_whatsapp_number = '+919812627589'
    if mode == "WhatsApp":
        from_whatsapp_number = 'whatsapp:+14155238886'
        to_whatsapp_number = 'whatsapp:+919812627589'

    account_sid = 'ACf1acf121ec795acb0f3d1440a1c9de3a'
    auth_token = 'c565bcfe6cd8ec999ee3a0e8543a9c80'
    client = Client(account_sid, auth_token)
    # index = False
    message = client.messages.create(body=f"{data.to_csv(index=False)}", from_=from_whatsapp_number,to=to_whatsapp_number)
    # message = client.messages.create(body=data, from_=from_whatsapp_number,to=to_whatsapp_number)

    return message.sid


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

def draw_bbox(img, bbox, confidence, labels):
    l, t, rt = 10, 5, 1
    for label, box, conf in zip(labels, bbox, confidence):
        caption = "{} {:.2f}%".format(label, conf * 100)
        x, y, w, h = box
        x1, y1 = x + w, y + h
        # if label == face:
        #     gender, confidence = cv.detect_gender(face_f)
        #     idx = np.argmax(confidence)
        #     label = label[idx]
        #     label = "{}: {:.2f}%".format(gender, confidence[idx] * 100)
        #     cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(img, caption, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.rectangle(img, (x, y), (x1, y1),(255, 0, 255), rt)
        cv2.circle(img, (x,y), 5, (255, 0, 0), -1)
        # # Top Left  x,y
        # cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        # cv2.line(img, (x, y), (x, y+l), (255, 0, 255),    t)
        # # Top Right  x1,y
        # cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        # cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # # Bottom Left  x,y1
        # cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        # cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # # Bottom Right  x1,y1
        # cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        # cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
    return img

logo, times = st.beta_columns(2)
st.image("./default_image/cu.jpeg")
logo.image("./default_image/cur.png")

# logo.image("./default_image/curaj1.png",)
st.sidebar.title("VI-ANPR-TL Application in Security Surveillance")
st.sidebar.image("./default_image/images.jpeg", use_column_width=True)
workflow = st.sidebar.selectbox("Let's Go!",
                                ["Welcome", "About-Project", "Project-Quick-Recap", "DataSet", "Work-Flow", "Reference",
                                 "Requirements", "Thesis Report"], key="Menu")

if workflow == "Welcome":
    st.image("./default_image/img.jpeg")
    img_path = [f"./default_image/{i}.jpg" for i in range(1, 3)]
    st.image(img_path, use_column_width=True)

if workflow == "About-Project":
    # FRAME_WINDOW = st.image([])
    img_path = [f"./Project_Presentation/{i}.jpg" for i in range(1, 17)]
    st.image(img_path, use_column_width=True)
if workflow == "Thesis Report":
    # FRAME_WINDOW = st.image([])
    img_path = [f"./Project_Presentation/{i}.jpg" for i in range(1, 13)]
    st.image(img_path, use_column_width=True)

if workflow == "Project-Quick-Recap":
    st.image("./default_image/img.jpeg")
    img_path = [f"./Getting to the Core/{i}.jpg" for i in range(1, 3)]
    st.image(img_path, use_column_width=True)

if workflow == "DataSet":
    img_data()

if workflow == "Reference":
    st.text("""
            1. H. Karwal and A. Girdhar, "Vehicle Number Plate Detection System for Indian Vehicles," 
            2015 IEEE International Conference on Computational Intelligence & Communication Technology,
            2015,pp. 8-12,DOI: 10.1109/CICT.2015.13

            3. Chengcheng Ning, Huajun Zhou, Yan Song, and Jinhui Tang,"Inception Single Shot MultiBox 
            Detector for object detection,"2017 IEEE International Conference on Multimedia & Expo Workshops 
            (ICMEW), 2017, pp. 549-554,DOI: 10.1109/ICMEW.2017.8026312.

            4. Object detection: speed and accuracy comparison (Faster R-CNN, R-FCN, SSD, FPN, RetinaNet,
             and YOLOv3)  By: Jonathan Hui (Medium Article)

    """)

if workflow == "Requirements":
    st.info("Updated Soon")

if workflow == "Work-Flow":
    FRAME_WINDOW = st.image([])
    IN_TIME = list()
    IN_OB = list()
    OUT_TIME = list()
    OUT_OB = list()
    # FRAME_DATA_IN, FRAME_DATA_OUT, COUNT_IN, COUNT_OUT = st.dataframe([]), st.dataframe([]), st.text(""), st.text("")
    # FRAME_DATA_IN, FRAME_DATA_OUT, COUNT_IN, COUNT_OUT = st.empty(), st.empty(), st.empty(), st.empty()
    FRAME_DATA_IN, FRAME_DATA_OUT = st.beta_columns(2)
    COUNT_IN, COUNT_OUT = st.beta_columns(2)
    FRAME_DATA_IN = FRAME_DATA_IN.dataframe([])
    FRAME_DATA_OUT = FRAME_DATA_OUT.dataframe([])
    COUNT_IN = COUNT_IN.text(" ")
    COUNT_OUT = COUNT_OUT.text(" ")
    # EMERGENCY, MODE,SEND = st.beta_columns(3)
    DF_IN = pd.DataFrame({"Object": ["None"], "In_time": ["None"]})
    DF_OUT = pd.DataFrame({"Object": ["None"], "Out_time": ["None"]})
    # FRAME_DATA = st.dataframe([])
    # TEXT = st.text("")
    check = st.sidebar.radio("Select Action", ["Live Feed", "Image Feed"])
    MODE = st.sidebar.selectbox("Select MODE", ["Select", "SMS", "WhatsApp"])

    in_time = list()
    out_time = list()
    times = times.empty()
    if check == "Live Feed":
        STATUS, SEND = st.sidebar.empty(), st.sidebar.button("send")
        Active, Deactive = st.beta_columns(2)
        # df = pd.read_csv("")
        active = st.radio("Camera", ["Camera-ON", "OFF"])
        #keras_model for checking in and out
        np.set_printoptions(suppress=True)
        model = tensorflow.keras.models.load_model('./model/model.h5')
        TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        if active == "Camera-ON":
            camera0 = cv2.VideoCapture(0)
            camera1 = r"http://10.80.70.209:8080/shot.jpg"
            ACTIVE = Active.selectbox("Camera-ID", ["Default Cam", "Gate No 3", "Video-File", "Canteen", "MegaMess"])
            while ACTIVE == "Default Cam":
                pTime = 0
                while True:
                    track = "bike_in", "bike_out", "car_in", 'car_out', "background", "project_owner"
                    times.text("Date & Time: " + str(datetime.now()).split(".")[0])
                    _, frame = camera0.read()
                    # _, frame1 = camera1.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    # frame = cv2.resize(frame, (224, 224))
                    image_array = np.asarray(cv2.resize(frame, (224, 224)))
                    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                    TM_DATA[0] = normalized_image_array
                    output = model.predict(TM_DATA)
                    prediction = list(model.predict(TM_DATA)[0])
                    ind = prediction.index(max(prediction))
                    face, p_conf = cv.detect_face(frame.copy())
                    bbox, label, o_conf = cv.detect_common_objects(frame.copy(), confidence=0.20, model='yolov3-tiny')
                    face_f = draw_bbox(img=frame, bbox=face, confidence=p_conf, labels=["person"])
                    obj_f = draw_bbox(img=frame, bbox=bbox, confidence=o_conf, labels=label)
                    if ind <= 5:
                        if track[ind] in ["bike_in", "car_in", "project_owner"]:
                            # if (track[ind] != DF_IN.Object.iloc[-1]) or (DF_IN.Object.iloc[-1] == "None"):
                            IN_OB.append(track[ind])
                            IN_TIME.append(str(datetime.now()).split(".")[0])
                        if track[ind] in ["bike_out", 'car_out']:
                            # if (track[ind] != DF_OUT.Object.iloc[-1]) or (DF_OUT.Object.iloc[-1] == "None"):
                            OUT_OB.append(track[ind])
                            OUT_TIME.append(str(datetime.now()).split(".")[0])

                    DF_IN = pd.DataFrame({"Object": IN_OB, "In_time": IN_TIME})
                    DF_OUT = pd.DataFrame({"Object": OUT_OB, "Out_time": OUT_TIME})
                    # DF_IN.to_csv("default_IN.csv")
                    # DF_OUT.to_csv("default_Out.csv")
                    cTime = time.time()
                    fps = 1 / (cTime - pTime)
                    pTime = cTime
                    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

                    # FRAME_WINDOW.image([frame0,frame0], width = 300,use_column_width=True)
                    FRAME_WINDOW.image([obj_f], use_column_width=True)
                    if active == "Camera-ON":
                        FRAME_DATA_IN.dataframe(DF_IN)
                        FRAME_DATA_OUT.dataframe(DF_OUT)
                        COUNT_IN.text("IN_Count: "+str(len(DF_IN)))
                        COUNT_OUT.text("OUT_Count: "+str(len(DF_OUT)))
            field_names = ["Object", "In_time"]
            for x, y in zip(IN_OB, IN_TIME):
                with open('Incoming.csv', 'a') as f:
                    writer_object = DictWriter(f, fieldnames=field_names)
                    writer_object.writerow({"Object": x, "In_time": y})
                    f.close()
            # DF_IN.to_csv("Incoming.csv", index=False)
            DF_OUT = pd.DataFrame({"Object": OUT_OB, "Out_time": OUT_TIME})
            field_names = ["Object", "OUT_TIME"]
            for x, y in zip(OUT_OB, OUT_TIME):
                with open('Incoming.csv', 'a') as f:
                    writer_object = DictWriter(f, fieldnames=field_names)
                    writer_object.writerow({"Object": x, "Out_time": y})
                    f.close()
            STATUS = st.sidebar.empty()
            if (MODE == "SMS") and STATUS.info("SMS Preparing....."):
                df = pd.read_csv("Incoming.csv")
                # print(DF_IN)
                if st.sidebar.button("send", key="live_sms"):
                    ID = sendInfo(df, "SMS")
                    # DF_IN =
                    if ID:
                        STATUS.success("Succesfully Send")
            if (MODE == "WhatsApp") and STATUS.info("WhatsApp Massage Preparing....."):
                df = pd.read_csv("Incoming.csv")
                if st.sidebar.button("send", key="live_app"):
                    ID = sendInfo(df, "WhatsApp")
                    if ID:
                        STATUS.success("Succesfully Send")
            # if (MODE == "SMS") and STATUS.info("SMS Preparing....."):
            #     # print(DF_IN)
            #     if SEND == "send":
            #         ID = sendInfo(DF_IN, "SMS")
            #         # DF_IN =
            #         if ID:
            #             STATUS.success("Succesfully")
            # if (MODE == "WhatsApp") and STATUS.info("WhatsApp Massage Preparing....."):
            #     if SEND == "send":
            #         ID = sendInfo(DF_IN, "WhatsApp")
            #         if ID:
            #             STATUS.success("Succesfully")

            if ACTIVE == "Gate No 3":
                in_time.append(str(datetime.now()))
                while True:
                    imgResponse = urllib.request.urlopen(camera1)
                    imgNp = np.array(bytearray(imgResponse.read()), dtype=np.uint8)
                    img = cv2.imdecode(imgNp, -1)
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame, channels='RGB')

            if ACTIVE == "Video-File":
                uploaded_file = st.file_uploader("Browser Files", type=[".mp4", ".avi"])
                if uploaded_file is not None:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(uploaded_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    while True:
                        track = "bike_in", "bike_out", "car_in", 'car_out', "background", "project_owner"
                        # times.text("Date & Time: " + str(datetime.now()).split(".")[0])
                        _, frame = cap.read()
                        # _, frame1 = camera1.read()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                        # frame = cv2.resize(frame, (224, 224))
                        image_array = np.asarray(cv2.resize(frame, (224, 224)))
                        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                        TM_DATA[0] = normalized_image_array
                        output = model.predict(TM_DATA)
                        prediction = list(model.predict(TM_DATA)[0])
                        ind = prediction.index(max(prediction))
                        face, p_conf = cv.detect_face(frame.copy())
                        bbox, label, o_conf = cv.detect_common_objects(frame.copy(), confidence=0.20,
                                                                       model='yolov3-tiny')
                        face_f = draw_bbox(img=frame, bbox=face, confidence=p_conf, labels=["person"])
                        obj_f = draw_bbox(img=frame, bbox=bbox, confidence=o_conf, labels=label)
                        if ind <= 5:
                            if track[ind] in ["bike_in", "car_in", "project_owner"]:
                                # if (track[ind] != DF_IN.Object.iloc[-1]) or (DF_IN.Object.iloc[-1] == "None"):
                                IN_OB.append(track[ind])
                                IN_TIME.append(str(datetime.now()).split(".")[0])
                            if track[ind] in ["bike_out", 'car_out']:
                                # if (track[ind] != DF_OUT.Object.iloc[-1]) or (DF_OUT.Object.iloc[-1] == "None"):
                                OUT_OB.append(track[ind])
                                OUT_TIME.append(str(datetime.now()).split(".")[0])

                        DF_IN = pd.DataFrame({"Object": IN_OB, "In_time": IN_TIME})
                        DF_OUT = pd.DataFrame({"Object": OUT_OB, "Out_time": OUT_TIME})
                        # DF_IN.to_csv("default_IN.csv")
                        # DF_OUT.to_csv("default_Out.csv")
                        # cTime = time.time()
                        # fps = 1 / (cTime - pTime)
                        # pTime = cTime
                        # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

                        # FRAME_WINDOW.image([frame0,frame0], width = 300,use_column_width=True)
                        FRAME_WINDOW.image([obj_f], use_column_width=True)
                        if active == "Camera-ON":
                            FRAME_DATA_IN.dataframe(DF_IN)
                            FRAME_DATA_OUT.dataframe(DF_OUT)
                            COUNT_IN.text("IN_Count: " + str(len(DF_IN)))
                            COUNT_OUT.text("OUT_Count: " + str(len(DF_OUT)))

                    # while True:
                    #     times.text("Date & Time: " + str(datetime.now()))
                    #     _, frame = cap.read()
                    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #     face, conf = cv.detect_face(frame)
                    #     face_f = draw_bbox(img=frame, bbox=face, confidence=conf, labels="Person")
                    #     bbox, label, conf = cv.detect_common_objects(frame.copy(), confidence=0.20, model='yolov3-tiny')
                    #     obj_f = draw_bbox(img=frame, bbox=bbox, confidence=conf, labels=label)
                    #     cTime = time.time()
                    #     # fps = 1 / (cTime - pTime)
                    #     # pTime = cTime
                    #     # cv2.putText(frame, f'FPS:{fps}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                    #     # cv2.waitKey(20)
                    #     FRAME_WINDOW.image([frame], width=300, use_column_width=True)
        # FRAME_DATA_IN.dataframe(DF_IN)
        # FRAME_DATA_OUT.dataframe(DF_OUT)
        # COUNT_IN.text(len(DF_IN))
        # COUNT_OUT.text(len(DF_OUT))
        # print(DF_IN)
        # df.to_csv("default.csv")

    if check == "Image Feed":
        # uploaded_file = st.sidebar.file_uploader("Browser Files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        label = "bike_in", "bike_out", "car_in", 'car_out', "background", "project_owner"
        uploaded_file = st.file_uploader("Browser Files", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            frame = cv2.imread(tfile.name)
            frame = cv2.resize(frame, (512, 512))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('./model/model.h5')
            TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            image_array = np.asarray(cv2.resize(frame, (224, 224)))
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            TM_DATA[0] = normalized_image_array
            output = model.predict(TM_DATA)
            prediction = list(model.predict(TM_DATA)[0])
            ind = prediction.index(max(prediction))
            # file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            # frame = cv2.imdecode(file_bytes, 1)
            # face, conf = cv.detect_face(frame)
            # face_f = draw_bbox(img=frame, bbox=face, confidence=conf, labels="Person")
            # bbox, label, conf = cv.detect_common_objects(frame.copy(), confidence=0.20, model='yolov3-tiny')
            # obj_f = draw_bbox(img=frame, bbox=bbox, confidence=conf, labels=label)
            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            frame = cv2.putText(frame, label[ind], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            if ind <= 5:
                if label[ind] in ["bike_in", "car_in"]:
                    # if (track[ind] != DF_IN.Object.iloc[-1]) or (DF_IN.Object.iloc[-1] == "None"):
                    IN_OB.append(label[ind])
                    IN_TIME.append(str(datetime.now()).split(".")[0])
                if label[ind] in ["bike_out", 'car_out']:
                    # if (track[ind] != DF_OUT.Object.iloc[-1]) or (DF_OUT.Object.iloc[-1] == "None"):
                    OUT_OB.append(label[ind])
                    OUT_TIME.append(str(datetime.now()).split(".")[0])

            DF_IN = pd.DataFrame({"Object": IN_OB, "In_time": IN_TIME})
            field_names = ["Object", "In_time"]
            for x, y in zip(IN_OB, IN_TIME):
                with open('Incoming.csv', 'a') as f:
                    writer_object = DictWriter(f, fieldnames=field_names)
                    writer_object.writerow({"Object": x, "In_time": y})
                    f.close()
            # DF_IN.to_csv("Incoming.csv", index=False)
            DF_OUT = pd.DataFrame({"Object": OUT_OB, "Out_time": OUT_TIME})
            field_names = ["Object", "OUT_TIME"]
            for x, y in zip(OUT_OB, OUT_TIME):
                with open('Incoming.csv', 'a') as f:
                    writer_object = DictWriter(f, fieldnames=field_names)
                    writer_object.writerow({"Object": x, "Out_time": y})
                    f.close()
            # DF_OUT.to_csv("Outgoing.csv")
            FRAME_WINDOW.image([frame], clamp=True)
            FRAME_DATA_IN.dataframe(DF_IN)
            FRAME_DATA_OUT.dataframe(DF_OUT)
            COUNT_IN.text("IN_Count: " + str(len(DF_IN)))
            COUNT_OUT.text("OUT_Count: " + str(len(DF_OUT)))
        # st.write("Image Uploaded Successfully")
        STATUS = st.sidebar.empty()
        if (MODE == "SMS") and STATUS.info("SMS Preparing....."):
            df = pd.read_csv("Incoming.csv")
            # print(DF_IN)
            if st.sidebar.button("send", key="img_sms"):
                ID = sendInfo(df, "SMS")
                # DF_IN =
                if ID:
                    STATUS.success("Succesfully Send")
        if (MODE == "WhatsApp") and STATUS.info("WhatsApp Massage Preparing....."):
            df = pd.read_csv("Incoming.csv")
            if st.sidebar.button("send", key="img_app"):
                ID = sendInfo(df, "WhatsApp")
                if ID:
                    STATUS.success("Succesfully Send")

        else:
            st.write("Make sure you image is in JPG/PNG Format.")
