import face_recognition
import cv2
import numpy as np
import os
import time

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
    print("input", face_encs)
    if len(face_encs) == 0:
        continue

    enc_vec[img] = face_encs[0]


# print(f"Loaded {len(enc_vec)} known faces")

def cosine_sim(emb1, emb2):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

"""
input_enc = []
cam = cv2.VideoCapture(0)
count = 0
last_time= time.time()
while True:
    current_time=time.time()
    time_diff= current_time -last_time
    if time_diff > 5:
        ret, frame = cam.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        last_time= current_time


        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print(face_encodings)
        for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
            input_enc.append(enc)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        face_img = frame[top:bottom, left:right]
        if face_img.size == 0:
            continue

        img_path = os.path.join(saved_, f"face_{count}.jpg")
        cv2.imwrite(img_path, face_img)
        count += 1

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    



cam.release()
cv2.destroyAllWindows()

attendance = []
all_similarities = []   # store everything if needed later

for i, captured_encoding in enumerate(input_enc):

    print(f"\nCaptured Face {i}")
    distance_dict = {}
    cosine_dict = {}

    for name, known_encoding in enc_vec.items():

        # Euclidean distance (L2)
        dist = face_recognition.face_distance(
            [known_encoding], captured_encoding
        )[0]

        # Cosine similarity
        cos_sim = cosine_sim(captured_encoding, known_encoding)

        distance_dict[name] = dist
        cosine_dict[name] = cos_sim

        print(f"  {name} → distance: {dist:.4f}, cosine: {cos_sim:.4f}")

    # Best match by distance
    best_name = min(distance_dict, key=distance_dict.get)
    best_distance = distance_dict[best_name]
    best_cosine = cosine_dict[best_name]

    # Decision rule (you can tune this)
    if best_distance < 0.45 and best_cosine > 0.85:
        attendance.append(best_name)
    else:
        attendance.append("Unknown")

    all_similarities.append({
        "captured_face": i,
        "distances": distance_dict,
        "cosine_similarities": cosine_dict
    })

print("\nFinal Attendance Result:")
print(attendance)
print(all_similarities)
"""

