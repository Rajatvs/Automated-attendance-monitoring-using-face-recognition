import cv2
import os
import numpy as np

# Set the student's name or ID for the folder
student_name = "RAJATSHIMPI"  # Replace with actual name or ID
dataset_path = f"C:\\Users\\rajat\\anaconda3\\Lib\\site-packages\\pythonwin\\pywin\\selfATTENDANCE\\{student_name}"

# Create a folder for face data if it doesn't exist
os.makedirs(dataset_path, exist_ok=True)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize a list to hold face data and labels
face_samples = []
ids = []

# A numeric ID for the student (you can use any integer, not just the name)
student_id = 1  # Assign a unique integer ID for the student

# Loop through each image in the dataset folder
for image_name in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, image_name)
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face detected, add the image and label to the data list
    for (x, y, w, h) in faces:
        face_samples.append(gray[y:y + h, x:x + w])  # Crop the face from the image
        ids.append(student_id)  # Use the student's numeric ID as the label

# Train the recognizer with the face data and labels
recognizer.train(face_samples, np.array(ids))

# Save the trained model
recognizer.save(f"{dataset_path}\\trainer.yml")

print("Training complete and model saved.")
