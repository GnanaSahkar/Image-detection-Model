# scripts/recognize.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_frame(frame, detector):
    results = detector.detect_faces(frame)
    if results:
        x, y, width, height = results[0]['box']
        face = frame[y:y+height, x:x+width]
        face = cv2.resize(face, (224, 224))
        face = preprocess_input(np.array(face, dtype="float32"))
        face = np.expand_dims(face, axis=0)
        return face
    return None

# Load model and class names
model = load_model('../models/best_model.keras')
data = np.load('../data.npz')
class_names = data['class_names']
detector = MTCNN()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = preprocess_frame(frame, detector)
    if processed_frame is not None:
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions[0])
        predicted_name = class_names[predicted_class]

        # Display the result
        cv2.putText(frame, predicted_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Tracking", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
