
import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = "C:\\Users\\aymen\\OneDrive\\Documents\\datasets\\faces"

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    print(f"Processing images for emotion: {emotion}")
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load image at path: {image_path}")
            continue

        face_landmarks = get_face_landmarks(image)

        if face_landmarks is None:
            print(f"Failed to get face landmarks for image at path: {image_path}")
            continue

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))
print("Script completed. Check data.txt for output.")