from retinaface import RetinaFace
import cv2
from insightface.app import FaceAnalysis


import time

# Image path
img_path = "CSFrame2.jpeg"

# Load image for OpenCV
img = cv2.imread(img_path)
#print(img)
# Detect faces

t1 = time.time()
faces = RetinaFace.detect_faces(img_path)
t2 = time.time()

tm = t2 - t1

print("total time to detect faces : ", tm)
# Loop over detected faces
for face_key in faces:
    identity = faces[face_key]

    facial_area = identity["facial_area"]   # [x1, y1, x2, y2]
    landmarks = identity["landmarks"]

    x1, y1, x2, y2 = facial_area

    # Draw bounding box
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        5
    )

    # Draw landmarks
    for point in landmarks.values():
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

# Show result
img = cv2.resize(img, (800,600))
cv2.imshow("RetinaFace Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
