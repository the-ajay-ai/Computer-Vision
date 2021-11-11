import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
label = "bike_in", "bike_out", "car_in", 'car_out', "none", "project_owner"
# Load the model
model = tensorflow.keras.models.load_model('./model/model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open('Test/bi_in/image/IMG20210412164354.jpg')

#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)

# display the resized image
# image.show()

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = list(model.predict(data)[0])
ind = prediction.index(max(prediction))
print(label[ind])

#
# label = bike_in, bike_out, car_in, car_out, none, project_owner
# import tensorflow.keras
# from PIL import Image, ImageOps
# import numpy as np
# import cv2 as cv #Please install with PIP: pip install cv2
#
# TM_DATA = None
# model = None
# cap = None
# ret = None
# frame = None
# output = None
# key = None
#
#
# print('START')
# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)
#
# # Load the model
# model = tensorflow.keras.models.load_model('./model/model.h5')
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1.
# TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# cap = cv.VideoCapture(0)
# while True:
#   ret , frame = cap.read()
#   cv.imshow('Window',frame)
#   frame = cv.resize(frame, (224, 224))
#   image_array = np.asarray(frame)
#   # Normalize the image
#   normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#   # Load the image into the array
#   TM_DATA[0] = normalized_image_array
#   output = model.predict(TM_DATA)
#   key = cv.waitKey(100)
#   print('Prediction')
#   print(output)
#   if key == (ord('q')):
#     break
# cv.destroyAllWindows()
# cap.release()
# print('TNE END')
