import face_recognition
import cv2
import numpy as np
import os

dataImg = 'players'
saved_ = 'captured_images'
register = 'dlib_register'

os.makedirs(saved_, exist_ok=True)

input_img = 'indianCricket.jpg'

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





input_enc = []
count = 0

img_ = face_recognition.load_image_file(input_img)

rgb_img = img_.copy()


face_locations = face_recognition.face_locations(img_)
#print(face_locations)
input_enc = face_recognition.face_encodings(img_, face_locations)

bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


attendance = []
count = 0

for encoding, (top, right, bottom, left) in zip(input_enc, face_locations):

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



    img_roi = img_[top:bottom, left:right]

    img_roi_bgr = cv2.cvtColor(img_roi, cv2.COLOR_RGB2BGR)

    file_name = f"{best_name}_{count}.jpg"
    save_path = os.path.join(register, file_name)
    count +=1
    cv2.imwrite(save_path, img_roi_bgr)

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


cv2.imshow("Labeled Faces", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
