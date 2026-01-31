from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
import face_recognition
import pandas as pd

import time

embedder = FaceNet()
detector = MTCNN()

known_encodings = {}
data_path = "Computer_Batch_2022"

storage = "CS2026_frame"

inference_time_known_enc= {}

count =0


for img_name in os.listdir(data_path):
    count +=1
    img_path = os.path.join(data_path, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t11 = time.time()

    faces = detector.detect_faces(img)
    t12 = time.time()

    t1 = t12 - t11

    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)

    face_img = img[y:y+h, x:x+w]

    t21 = time.time()
    embedding = embedder.embeddings([face_img])[0]
    t22 = time.time()

    t2 = t22 - t21

    inf_tm = t2 + t1

    inference_time_known_enc[img_name]= inf_tm

    known_encodings[img_name] = embedding

print("Total faces stored:", len(known_encodings))
#print(known_encodings)

inference_frame = {}

input_img = 'CSFrame2.jpeg'
input_img = cv2.imread(input_img)
rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

tf1 = time.time()
input_faces = detector.detect_faces(rgb_img)
tf2 = time.time()


time_frame_detect = tf2 - tf1

print("total time for faces dtect: ", time_frame_detect)


attendance = []

count =0

inf_time_frame={}


for face in input_faces:
    x, y, w, h = face['box']

    input_face = rgb_img[y:y + h, x:x + w]

    total_tf = 0

    #tf1 = time.time()
    input_embedding = embedder.embeddings([input_face])[0]
    #tf2 = time.time()
    compare_faces = {}
    #
    #total_tf = tf2 - tf1
    #
    #
    for name,encoding in known_encodings.items():
        dist = face_recognition.face_distance([encoding], input_embedding)
        compare_faces[name]=dist
    #
    matched_face= min(compare_faces, key=compare_faces.get)
    attendance.append(matched_face)

    #inf_time_frame[f"{matched_face}_{count}"] =

    #img_roi_bgr = cv2.cvtColor(input_face, cv2.COLOR_RGB2BGR)

    #file_name = f"{matched_face}_{count}.jpg"
    #
    #inference_frame[f"{matched_face}_{count}"] = total_tf
    #
    #save_path = os.path.join(storage, file_name)
    count += 1
    #cv2.imwrite(save_path, img_roi_bgr)





for face in input_faces:
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)

    cv2.rectangle(
        rgb_img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

rgb_img = cv2.resize(rgb_img, (800, 600))


bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

cv2.imshow("Face Attendance", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
row1 = []

for name,tm in inference_time_known_enc.items():
    print(name, 'time : ',tm)
    row1.append([name, tm])

df_known = pd.DataFrame(
    row1,
    columns=["Name", "Inference Time (seconds)"]
)

df_known.to_csv("inference_times_known.csv", index=False)

"""

"""
row2=[]

for name,tm in inference_frame.items():
    print(name,tm)
    row2.append([ name, tm])

df_frame = pd.DataFrame(
    row2,
    columns=["Frame","Inference Time (seconds)"]
)

df_frame.to_csv("inference_times_frame_Poster.csv", index=False)

print("time to detect the faces in frame : ", time_frame_detect)
"""