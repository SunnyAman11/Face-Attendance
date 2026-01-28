from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
import face_recognition

embedder = FaceNet()
detector = MTCNN()

known_encodings = {}
data_path = "players"

storage = 'facenet_register'

for img_name in os.listdir(data_path):
    img_path = os.path.join(data_path, img_name)

    image = cv2.imread(img_path)
    if image is None:
        continue

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img)
    if len(faces) == 0:
        continue

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)

    face_img = img[y:y+h, x:x+w]

    embedding = embedder.embeddings([face_img])[0]

    known_encodings[img_name] = embedding

print("Total faces stored:", len(known_encodings))
#print(known_encodings)

input_img = 'indianCricket.jpg'
input_img = cv2.imread(input_img)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_faces = detector.detect_faces(input_img)

attendance = []

count =0

for face in input_faces:
    x, y, w, h = face['box']

    input_face = input_img[y:y + h, x:x + w]

    input_embedding = embedder.embeddings([input_face])[0]

    compare_faces = {}

    for name,encoding in known_encodings.items():
        dist = face_recognition.face_distance([encoding], input_embedding)
        compare_faces[name]=dist

    matched_face= min(compare_faces, key=compare_faces.get)
    attendance.append(matched_face)


    img_roi_bgr = cv2.cvtColor(input_face, cv2.COLOR_RGB2BGR)

    file_name = f"{matched_face}_{count}.jpg"
    save_path = os.path.join(storage, file_name)
    count += 1
    cv2.imwrite(save_path, img_roi_bgr)
