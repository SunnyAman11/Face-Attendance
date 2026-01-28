import face_recognition
import cv2
import numpy as np
import os

dataImg = 'database_photos'
saved_ = 'captured_images'

os.makedirs(saved_, exist_ok=True)

enc_vec = {}

for img in os.listdir(dataImg):

    img_path = os.path.join(dataImg, img)

    if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_read = face_recognition.load_image_file(img_path)

    face_encs = face_recognition.face_encodings(img_read)

    if len(face_encs) == 0:
        continue

    enc_vec[img] = face_encs[0]

#print(f"Loaded {len(enc_vec)} known faces")

def consine_sim(emb1,emb2):
    j=0



input_enc = []
cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        input_enc.append(enc)
    """
    

        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue

        img_path = os.path.join(saved_, f"face_{count}.jpg")
        cv2.imwrite(img_path, face_img)
        count += 1

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    """
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

attendance = []

for captured_encoding in input_enc:

    distance_dict = {}

    for name, known_encoding in enc_vec.items():
        dist = face_recognition.face_distance(
            [known_encoding], captured_encoding
        )[0]

        distance_dict[name] = dist

    best_name = min(distance_dict, key=distance_dict.get)
    best_distance = distance_dict[best_name]

    if best_distance < 0.45:
        attendance.append(best_name)
    else:
        attendance.append("Unknown")

print(attendance)
