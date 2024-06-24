# scripts/load_data.py

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import preprocess_input

def load_images(data_dir):
    images = []
    labels = []
    person_names = os.listdir(data_dir)
    detector = MTCNN()

    for person_name in person_names:
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                results = detector.detect_faces(image)
                if results:
                    x, y, width, height = results[0]['box']
                    face = image[y:y+height, x:x+width]
                    face = cv2.resize(face, (224, 224))
                    face = preprocess_input(np.array(face, dtype="float32"))
                    images.append(face)
                    labels.append(person_name)
    
    if not labels:
        raise ValueError("No labels found in the data directory.")

    images = np.array(images)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)
    
    return images, labels, le.classes_

if __name__ == "__main__":
    data_dir = "known_faces"
    try:
        images, labels, class_names = load_images(data_dir)
        np.savez_compressed('../data.npz', images=images, labels=labels, class_names=class_names)
        print("Data loaded and saved successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")
