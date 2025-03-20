import cv2
import os

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a directory to store captured faces
path = "faces"
os.makedirs(path, exist_ok=True)

while True:
    # Take input for the person's name
    name = input("\n📸 Enter name of person (or press 'ESC' to Quit): ")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Initialize image counter
    image_count = 1

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display the entered name on the frame
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display on-screen instructions
        cv2.putText(frame, "Press 'C' to Capture", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'ESC' to Quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Press 'CTRL + S' to Save", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the output
        cv2.imshow(f"{name}", frame)

        key = cv2.waitKey(1) & 0xFF

        # If 'c' key is pressed, capture and save the image
        if key == ord('c'):
            filename = os.path.join(path, f"{name}_{image_count}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Image {image_count} Saved: {filename}")
            image_count += 1
            break  # Restart name input after capturing

        # If 'ESC' key is pressed, exit
        elif key == 27:  # ASCII value of ESC key
            cap.release()
            cv2.destroyAllWindows()
            print("\n🔴 Exiting Program...")
            exit()

    # Release camera after capturing image
    cap.release()
    cv2.destroyAllWindows()
