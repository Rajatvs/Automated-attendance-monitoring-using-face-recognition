import cv2
import numpy as np
import os
from datetime import datetime

# Load the trained model
student_name = "RAJATSHIMPI"  # Replace with the student name or ID
dataset_path = f"C:\\Users\\rajat\\anaconda3\\Lib\\site-packages\\pythonwin\\pywin\\selfATTENDANCE\\{student_name}"

# Load the trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(f"{dataset_path}\\trainer.yml")

# Load the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to mark attendance
def mark_attendance(name):
    with open(f"{dataset_path}\\attendance.csv", "a") as f:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{current_time}\n")

# Start recognizing faces
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face
        face_region = gray[y:y + h, x:x + w]

        # Predict the ID of the recognized face
        id_, confidence = recognizer.predict(face_region)

        if confidence < 100:
            # Add the name based on the ID (you can use a mapping of IDs to names if you have multiple students)
            name = "RAJATSHIMPI"  # Use your student's name here
            confidence_text = f"  {round(100 - confidence)}%"
            # Mark attendance
            mark_attendance(name)
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"

        # Draw rectangle around the face and add name and confidence
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {confidence_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
