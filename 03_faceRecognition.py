import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import sqlite3
from datetime import datetime

# Load your trained model
model = load_model('face_recognition_model.h5')  # Update the path as needed

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Retrieve the class names
class_names = label_encoder.classes_
print("Class names:", class_names)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a confidence threshold
confidence_threshold = 0.95  # Adjust as needed

# Set up the SQLite database connection
conn = sqlite3.connect("recognition_log.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS recognition_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
''')
conn.commit()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy of the frame for output
    output_frame = frame.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Draw a bounding box around the detected face
        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess the face for prediction
        face_roi = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (128, 128))
        face_preprocessed = face_resized.astype('float32') / 255.0
        face_preprocessed = np.expand_dims(face_preprocessed, axis=0)

        # Get prediction and confidence
        predictions = model.predict(face_preprocessed)
        max_confidence = np.max(predictions)
        label_index = np.argmax(predictions)
        if max_confidence >= confidence_threshold:
            label_text = label_encoder.inverse_transform([label_index])[0]
        else:
            label_text = "Unknown"

        # Prepare label background for readability
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(output_frame, (x, y - label_size[1] - 10), (x + label_size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(output_frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Log recognized user with timestamp if not "Unknown"
        if label_text != "Unknown":
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute("INSERT INTO recognition_log (user, timestamp) VALUES (?, ?)", (label_text, timestamp))
            conn.commit()

    # Display the resulting frame
    cv2.imshow('Face Recognition', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
conn.close()
