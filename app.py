from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Global configuration
IMG_SIZE = (128, 128)
CONFIDENCE_THRESHOLD = 0.90  # Lowered threshold for better detection
LOGGING_INTERVAL = 5  # Log recognized users every 5 seconds

# Load model and label encoder
model = load_model('face_recognition_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# SQLite database setup
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

# Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Global variables for logging and recognition
last_logged = {}  # Dictionary to track last log time for each user
lock = threading.Lock()  # Lock for thread safety

def process_frame(frame):
    """Processes a single frame for face detection and recognition."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    recognized_faces = []

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMG_SIZE)
        face_preprocessed = face_resized.astype('float32') / 255.0
        face_preprocessed = np.expand_dims(face_preprocessed, axis=0)

        predictions = model.predict(face_preprocessed)
        max_confidence = np.max(predictions)
        label_index = np.argmax(predictions)

        if max_confidence >= CONFIDENCE_THRESHOLD:
            label_text = label_encoder.inverse_transform([label_index])[0]
        else:
            label_text = "Unknown"

        recognized_faces.append({
            'label': label_text,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })

    return recognized_faces

def log_recognition(label_text):
    """Logs recognized user with timestamp, respecting the logging interval."""
    with lock:
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        if label_text != "Unknown":
            last_log_time = last_logged.get(label_text)
            if last_log_time is None or (now - last_log_time).seconds >= LOGGING_INTERVAL:
                cursor.execute("INSERT INTO recognition_log (user, timestamp) VALUES (?, ?)",
                               (label_text, timestamp))
                conn.commit()
                last_logged[label_text] = now

def gen_frames():
    """Video streaming generator with optimized processing and logging."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        recognized_faces = process_frame(frame)

        for face_data in recognized_faces:
            x, y, w, h = face_data['x'], face_data['y'], face_data['w'], face_data['h']
            label_text = face_data['label']

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            log_recognition(label_text)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    """Renders the interactive HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    """Returns recognition logs as JSON."""
    cursor.execute("SELECT user, timestamp FROM recognition_log ORDER BY timestamp DESC LIMIT 50")
    logs = cursor.fetchall()
    return jsonify([{'user': log[0], 'timestamp': log[1]} for log in logs])

if __name__ == '__main__':
    app.run(debug=True)