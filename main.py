import torch
import cv2
import numpy as np
import mediapipe as mp
import concurrent.futures

# Load YOLOv5 model using torch.hub from the correct directory
model = torch.hub.load('D:/ObjectAndHandRecognition/ObjDectection/yolov5', 'custom', path='yolov5x.pt', source='local')

# MediaPipe initialization for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam using DirectShow
print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
else:
    print("Webcam opened successfully.")

cv2.namedWindow("Hand, Face, and Object Detection - Pranav v2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand, Face, and Object Detection - Pranav v2", 640, 480)

frame_skip = 2  # Process every 2nd frame
frame_count = 0

def capture_frame():
    ret, frame = cap.read()
    return ret, frame

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    return results

def process_frame(frame, results):
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]
        confidence = float(conf)
        # Drawing rectangle around the object
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        # Drawing the label and confidence on the frame
        text = f'{label}: {confidence:.2f}'
        font_scale = 1.5
        font_thickness = 3
        text_color = (34, 139, 34)  # Dark Green color
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_w, text_h = text_size
        # Background rectangle for the text
        cv2.rectangle(frame, (int(x1), int(y1) - text_h - 10), (int(x1) + text_w, int(y1)), (0, 0, 0), -1)
        # Putting the text on the frame
        cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    return frame

def main_loop():
    global frame_count
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            future_frame = executor.submit(capture_frame)
            ret, frame = future_frame.result()

            if not ret:
                print("Failed to grab frame")
                break

            height, width, channels = frame.shape
            frame_count += 1

            if frame_count % frame_skip == 0:
                future_results = executor.submit(detect_objects, frame)
                results = future_results.result()
                frame = process_frame(frame, results)

            # Process the frame for hand detection
            rgb_frame = frame[:, :, ::-1]
            results_hands = hands.process(rgb_frame)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3),
                                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3))

            # Process the frame for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

            # Display the frame with all detections
            cv2.imshow('Hand, Face, and Object Detection - Pranav v2', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main_loop()
    cap.release()
    cv2.destroyAllWindows()
