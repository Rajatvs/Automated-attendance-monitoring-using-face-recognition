import cv2
import os

# Set the student's name or ID for the folder name
student_name = "RAJATSHIMPI"  # Use the actual name or ID
save_path = f"C:\\Users\\rajat\\anaconda3\\Lib\\site-packages\\pythonwin\\pywin\\selfATTENDANCE\\{student_name}"
os.makedirs(save_path, exist_ok=True)  # Create the folder if it doesn't exist

cap = cv2.VideoCapture(0)  # Open the default camera
count = 0  # Initialize a counter for image filenames

while True:
    ret, frame = cap.read()  # Capture a frame
    if not ret:
        break

    cv2.imshow("Frame", frame)  # Display the frame

    # Press 's' to save the current frame as an image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_name = os.path.join(save_path, f"{student_name}_{count}.jpg")
        cv2.imwrite(img_name, frame)  # Save the frame as an image file
        print(f"Image {img_name} saved!")
        count += 1

    # Press 'q' to quit the program
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
