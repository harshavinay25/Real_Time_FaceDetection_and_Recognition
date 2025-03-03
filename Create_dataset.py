import cv2
import os
import numpy as np

# Global constant for image size
IMG_SIZE = (128, 128)  # <<-- Global config for consistency

def detect_faces(frame, face_cascade):
    """Helper function to detect and sort faces by size."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    return faces

def refined_create_folder_images(name, directory, total_images, img_size=IMG_SIZE, color_mode='rgb'):  # <<-- Renamed tn to total_images and use IMG_SIZE
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    # Initialize the Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Start the video capture
    cam = cv2.VideoCapture(0)
    captured_count = 0

    print("Starting face capture. Press 'q' to quit early.")
    
    while captured_count < total_images:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        
        # Use the helper function for face detection
        faces = detect_faces(frame, face_cascade)
        
        if len(faces) > 0:
            # Draw bounding boxes for all detected faces (green)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Highlight the largest face with a red bounding box
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            # Extract the face region from the frame
            face_img = frame[y:y+h, x:x+w]
            
            # Convert color if needed (for training consistency)
            if color_mode.lower() == 'rgb':
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Resize to match the training image size
            face_img = cv2.resize(face_img, img_size, interpolation=cv2.INTER_AREA)
            
            # Save the captured face image; convert back to BGR if necessary
            filename = os.path.join(directory, f"{name}_{captured_count+1}.jpg")
            if color_mode.lower() == 'rgb':
                save_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            else:
                save_img = face_img
            cv2.imwrite(filename, save_img)
            captured_count += 1
            print(f"Captured image {captured_count}/{total_images}")
        
        # Overlay the capture progress on the frame
        cv2.putText(frame, f"Captured: {captured_count}/{total_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the current frame
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting capture by user command.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Capture completed. Total images captured:", captured_count)


if __name__ == "__main__":
    name = input("Enter your name: ")
    directory = os.path.join("dataset", name)
    total_images = int(input("Enter number of images to be captured: "))  # <<-- Changed variable name
    refined_create_folder_images(name, directory, total_images, img_size=IMG_SIZE, color_mode='rgb')
