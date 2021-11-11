import os
import glob
import cv2 as cv
import numpy as np
import pickle
import dlib
def face_detection(img):
  img = cv.imread(img)
  # print(img)
  img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
  # face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
  face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
  faces = face_cascade.detectMultiScale(img,scaleFactor=1.2, minNeighbors=5)
  # print("This image having {} person.".format(len(faces)))
  #ractangle box on faces
  crop_img = None
  if len(faces)>0:
    crop_img = []
    for x,y,w,h in faces:
      img = cv.rectangle(img,(x,y),(x+w,y+h+30),(0,0,255),3)
      crop_face = cv.resize(img[y:y + h, x:x + w],(160,160))
      crop_img.append(crop_face)
  # cv2_imshow(img)
  # cv2_imshow(crop_img[0])
  return crop_img[0]

def create_dataset(img):
  image = cv.imread(img)
  crop_img = face_detection(image)
  # path =
  # label =
  if crop_img is not None:
    location = img.replace("person","Person")
    i +=1
    # if i == 100:
    #   break
    if label in os.listdir("/content"):
      cv.imwrite(location, crop_img)
    else:
      os.mkdir("{}".format(label))
      cv.imwrite(location, crop_img)
    else:
      continue

for i in glob.glob("opencv/dataset/person_1/*.jpg"):
  print(i)