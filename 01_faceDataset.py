import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Global constant for image size (imported from the config in Step 1)
IMG_SIZE = (128, 128)

def load_images_from_folder(folder_path, img_size=IMG_SIZE):
    images = []       # To store the processed images
    labels = []       # To store corresponding labels (names)

    # Iterate through each subfolder (each representing a person)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)  # Using IMG_SIZE for consistency
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_and_save(folder_path, output_path='data.pkl', img_size=IMG_SIZE):
    images, labels = load_images_from_folder(folder_path, img_size)
    images = images / 255.0  # Normalize images

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    with open(output_path, 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test, le), f)
    
    print("Data preprocessed and saved to", output_path)

def check_labels_and_counts(folder_path):
    label_counts = {}
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            num_images = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))])
            label_counts[label] = num_images
    return label_counts

dataset_folder_path = 'dataset'
preprocess_and_save(dataset_folder_path)

label_counts = check_labels_and_counts(dataset_folder_path)
print("Labels and their image counts:")
for label, count in label_counts.items():
    print(f"{label}: {count} images")
