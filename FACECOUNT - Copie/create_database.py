import os
import cv2
import pickle
import torch
import numpy as np
from facecount_recognition import FaceRecognitionSystem

# --- Configuration ---
ATTENDANCE_DIR = 'attendance'
DATABASE_FILE = 'face_database.pkl'

def initialize_database():
    """
    Initializes or updates the face database by processing images from the attendance directory.
    """
    # The device is set inside the FaceRecognitionSystem class
    print(f"Initializing Face Recognition System...")

    # Initialize the recognition system
    try:
        recognition_system = FaceRecognitionSystem()
        print(f"Running on device: {recognition_system.device}")
    except Exception as e:
        print(f"Error initializing FaceRecognitionSystem: {e}")
        return

    face_database = {}
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'rb') as f:
                face_database = pickle.load(f)
            print(f"Loaded existing database with {len(face_database)} person(s).")
        except (pickle.UnpicklingError, EOFError):
            print("Could not read existing database file. Starting fresh.")
            face_database = {}

    if not os.path.exists(ATTENDANCE_DIR):
        print(f"Error: Attendance directory '{ATTENDANCE_DIR}' not found.")
        return

    print(f"Scanning '{ATTENDANCE_DIR}' for new faces...")
    image_files = [f for f in os.listdir(ATTENDANCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No images found in the attendance directory.")
        return

    for filename in image_files:
        try:
            # Extract person's name from filename (e.g., 'person_name_123.jpg' -> 'person_name')
            person_name = '_'.join(filename.split('_')[:-1])
            if not person_name:
                print(f"Could not extract name from '{filename}'. Skipping.")
                continue

            print(f"Processing {filename} for person: {person_name}")
            image_path = os.path.join(ATTENDANCE_DIR, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {filename}")
                continue

            # Detect faces
            faces = recognition_system.detect_faces(image)
            if not faces:
                print(f"No faces detected in {filename}.")
                continue

            # For simplicity, we use the largest face found in the image
            main_face = max(faces, key=lambda face: (face[2] - face[0]) * (face[3] - face[1]))
            face_box = [int(c) for c in main_face[:4]]

            # Extract embedding
            embedding = recognition_system.extract_face_embedding(image, face_box)

            if embedding is not None:
                if person_name not in face_database:
                    face_database[person_name] = []
                face_database[person_name].append(embedding)
                print(f"  Successfully added embedding for {person_name}.")
            else:
                print(f"  Could not extract embedding for a face in {filename}.")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

    # Save the updated database
    if face_database:
        try:
            with open(DATABASE_FILE, 'wb') as f:
                pickle.dump(face_database, f)
            print(f"\nDatabase saved successfully to {DATABASE_FILE}")
            print(f"Total persons in database: {len(face_database)}")
            for name, embeddings in face_database.items():
                print(f"  - {name}: {len(embeddings)} embedding(s)")
        except Exception as e:
            print(f"Error saving database: {e}")
    else:
        print("\nDatabase is empty. No faces were added.")

if __name__ == "__main__":
    initialize_database()
