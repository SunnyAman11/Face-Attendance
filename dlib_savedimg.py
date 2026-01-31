import face_recognition
import cv2
import numpy as np
import os
import time

import pandas as pd

dataImg = 'Computer_Batch_2022'
register = 'CS2026_frame_dlib'


input_img = 'CS2026_frame.JPG'

enc_vec = {}

Known_inference ={}

for img in os.listdir(dataImg):

    img_path = os.path.join(dataImg, img)

    if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_read = face_recognition.load_image_file(img_path)

    t11 = time.time()
    face_encs = face_recognition.face_encodings(img_read)
    t12 = time.time()

    t1= t12 - t11
    Known_inference[img] = t1

    if len(face_encs) == 0:
        continue

    enc_vec[img] = face_encs[0]





input_enc = []
count = 0

img_ = face_recognition.load_image_file(input_img)


rgb_img = img_.copy()

t21 = time.time()
face_locations = face_recognition.face_locations(img_)
t22 = time.time()

#print(face_locations)
t2 = t22 - t21
tf = t2

input_enc = face_recognition.face_encodings(img_, face_locations)


bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


attendance = []
count = 0

name = []

inf_frame = []

names = []

detection_time_total = tf

for encoding, loc in zip(input_enc, face_locations):

    distance_dict = {}

    for name, known_encoding in enc_vec.items():
        match_ = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.65)
        dist = face_recognition.face_distance([known_encoding], encoding)[0]
        print(match_, name)
        distance_dict[name] = dist

    best_name = min(distance_dict, key=distance_dict.get)
    best_distance = distance_dict[best_name]



    if (best_distance < 0.65):
        attendance.append(best_name)
    else:
        attendance.append("unknown")
    print('/n')

    top, right, bottom, left = loc
    # Face ROI
    img_roi = rgb_img[top:bottom, left:right]

    t3 = time.time()
    encoding = face_recognition.face_encodings(rgb_img, [loc])[0]
    t4 = time.time()
    encoding_time = t4 - t3

    # file_name = f"{best_name}_{count}.jpg"
    # count += 1
    """
    inf_frame.append({
        "face_id": f"{best_name}_{count}",
        "encoding_time_sec": encoding_time,
        "name": best_name
    })
    """
    img_roi = img_[top:bottom, left:right]

    img_roi_bgr = cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR)

    """
    file_name = f"{best_name}_{count}.jpg"
    save_path = os.path.join(register, file_name)
    count +=1
    cv2.imwrite(save_path, img_roi_bgr)
    """
print(attendance)

for (top, right, bottom, left) in face_locations:
    cv2.rectangle(
        bgr_img,
        (left, top),
        (right, bottom),
        (0, 255, 0),
        2
    )


"""
for (top, right, bottom, left) in face_locations:

    img_roi = img_[top:bottom, left:right]

    if img_roi.size == 0:
        continue

    img_roi_bgr = cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR)

    img_roi_bgr = cv2.resize(img_roi_bgr,(300,300))
    cv2.imshow("photo frame", img_roi_bgr)
    cv2.waitKey(0)

cv2.destroyAllWindows()
"""

bgr_img = cv2.resize(bgr_img,(800,600))
cv2.imshow("Labeled Faces", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




"""
print("detection_time_sec: ", detection_time_total)  # approx per face


df = pd.DataFrame(inf_frame)
df.to_csv("face_recognition_inference_times_frame.csv", index=False)

print("Attendance:", attendance)
"""