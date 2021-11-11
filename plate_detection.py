STATE={"AN":"Andaman and Nicobar", "AP":"Andhra Pradesh", "AR":"Arunachal Pradesh","AS":"Assam", "BR":"Bihar", "CG":"Chhattisgarh",
"CH":"Chandigarh", "DD":"Dadra and Nagar Haveli and Daman and Diu", "DL":"Delhi", "GA":"Goa", "GJ":"Gujarat",
"HP":"Himachal Pradesh", "HR":"Haryana", "JH":"Jharkhand", "JK":"Jammu and Kashmir", "KA":"Karnataka", "KL":"Kerala",
"LA":"Ladakh","LD":"Lakshadweep", "MH":"Maharashtra", "ML":"Meghalaya", "MN":"Manipur", "MP":"Madhya Pradesh", "MZ":"Mizoram",
"NL":"Nagaland", "OD":"Odisha", "PB":"Punjab","PY":"Puducherry", "RJ":"Rajasthan", "SK":"Sikkim", "TN":"Tamil Nadu",
"TR":"Tripura", "TS":"Telangana", "UK":"Uttarakhand", "UP":"Uttar Pradesh", "WB":"West Bengal"}

# Loading the required python modules
import pytesseract # this is tesseract module
# import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os
import numpy as np
# specify path to the license plate images folder as shown below
path_for_license_plates = "./number_plate/*.jpg"
# path_for_license_plates = "AS9527XZ.png"

list_license_plates = []
predicted_license_plates = []
state = []
for path_to_license_plate in glob.glob(path_for_license_plates, recursive=True):
    license_plate_file = path_to_license_plate.split("/")[-1]
    license_plate, _ = os.path.splitext(license_plate_file)
    list_license_plates.append(license_plate)
    plate = cv2.imread(path_to_license_plate)

    kernal = np.ones((1, 1), np.uint8)
    plate = cv2.dilate(plate, kernal, iterations=3)
    plate = cv2.erode(plate, kernal, iterations=3)
    # plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    (thros, plate) = cv2.threshold(plate, 150, 255, cv2.THRESH_BINARY)



    '''
    We then pass each license plate image file
    to the Tesseract OCR engine using the Python library
    wrapper for it. We get back predicted_result for
    license plate. We append the predicted_result in a
    list and compare it with the original the license plate
    '''
    # conf = '--oem 3 --psm 6 tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    predicted_result = pytesseract.image_to_string(plate, lang='eng')
    predicted_result = ''.join(e for e in predicted_result if e.isalnum() and (e.isupper() or e.isdigit()))
    state.append(predicted_result[0:2])

    # filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_license_plates.append(predicted_result)
print(" THRESH:" ,thros)
print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
print("--------------------", " ", "-----------------------",  "\t", "--------")
# print(list_license_plates)

def calculate_predicted_accuracy(actual_list, predicted_list):
    for actual_plate, predict_plate, city in zip(actual_list, predicted_list, state):
        accuracy = "0 %"
        num_matches = 0
        if actual_plate == predict_plate:
            accuracy = "100 %"
        else:
            if len(actual_plate) == len(predict_plate):
                for a, p in zip(actual_plate, predict_plate):
                    if a == p:
                        num_matches += 1
                accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
                accuracy += "%"
        print("	 ", actual_plate, "\t", predict_plate, "\t\t", accuracy)


calculate_predicted_accuracy(list_license_plates, predicted_license_plates)

# Read the license plate file and display it
# path = r"Test/car_in/image/DL1CV3683.jpg"
#
# img = cv2.imread(path)
# # resize_test_license_plate = cv2.resize(test_license_plate, None, fx = 2, fy = 2,interpolation = cv2.INTER_CUBIC)
# grayscale_resize_test_license_plate = cv2.cvtColor(test_license_plate, cv2.COLOR_BGR2GRAY)
# gaussian_blur_license_plate = cv2.GaussianBlur(grayscale_resize_test_license_plate, (5, 5), 0)
# conf = '--oem 3 -l eng --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
# # conf = r'--oem 3 --psm 6 outputbase digits'
# new_predicted = pytesseract.image_to_string(gaussian_blur_license_plate)
# filter_new_predicted = "".join(new_predicted.split()).replace(":", "").replace("-", "")
# print(new_predicted)

# boxes = pytesseract.image_to_data(img)
# for a,b in enumerate(boxes.splitlines()):
#         print(b)
#         if a!=0:
#             b = b.split()
#             if len(b)==12:
#                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
#                 cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
#                 cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 2)

# plate_img = cv2.imread(path)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)
