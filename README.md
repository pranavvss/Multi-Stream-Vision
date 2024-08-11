# Multi-Stream-Vision-Real-Time-Object-Face-and-Hand-Detection- (NO HARDWARE REQUIRED), (NOT FOR BEGINNERS)
 An advanced real-time detection system capable of recognizing multiple objects, hands, and faces within a video stream. (Could be also done on a Pre-recorded video)

---------------------------------------------------------------------------

## Video Example-

https://github.com/user-attachments/assets/1d4c14ac-b2b7-43b4-8250-7fa2d30a7297

---------------------------------------------------------------------------

## Project Info- 

This project involves developing an advanced real-time detection system capable of recognizing multiple objects, hands, and faces within a video stream. Leveraging the YOLOv5 model for object detection, MediaPipe for hand tracking, and OpenCV Haar Cascade for face detection, the system efficiently processes video feed from a webcam. This project demonstrates the integration of powerful computer vision tools and techniques to create a versatile and responsive detection system. The project also uses CUDA and cuDNN for GPU acceleration, ensuring the system operates with minimal lag, providing a smoother and faster user experience.

---------------------------------------------------------------------------

## Project Requirements

### 1. Programming Language: 
Python (Somewhere between 3.11 to 3.12) [Download Python](https://www.python.org/downloads/)

---------------------------------------------------------------------------

### 2. Libraries and Frameworks i have used:
   
2.1 OpenCV
Used for video capture, image processing, and Haar Cascade for face detection. [Download](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
  
2.2 Torch 
(PyTorch): A deep learning framework used to load and run the YOLOv5 model. [Download](https://pytorch.org/docs/stable/index.html)
  
2.3 YOLOv5
A state-of-the-art object detection model, known for its speed and accuracy. Not suggested to use any other version of Yolo. The YOLOv5 model can be easily integrated using PyTorch’s torch.hub.load() method [torch.hub Documentation](https://pytorch.org/docs/stable/hub.html). YOLOv5 is known for its balance of accuracy and speed, making it suitable for real-time applications. Ensure the model weights are properly downloaded and stored in the correct directory.
[All About Yolo5](https://docs.ultralytics.com/yolov5/)
[Git clone/Download yolo5](https://github.com/ultralytics/yolov5). Make sure Yolo5 is in the same directory where you will be saving the Pyhton script.
  
2.4 MediaPipe
Utilized for real-time hand detection and tracking.  [Download](https://ai.google.dev/edge/mediapipe/solutions/guide)
  
2.5 GStreamer 
An advanced pipeline-based multimedia framework used to ensure high-quality video input and output.
[Download](https://gstreamer.freedesktop.org/download/#windows),
Note: Download both Runtime installer and Development installer, Let the program download the files in default directory.
Note 2: Add GStreamer to the system PATH 
``` 
(Go to System environement variables > Under system variables > Click on Path > Edit > New > Paste the path of bin folder of GStream)
``` 
to enable its use within Python and OpenCV. It is essential for enhancing video quality by using efficient codecs and pipelines.
  
2.6 CUDA and cuDNN: 
Libraries that enable GPU acceleration for deep learning tasks, improving the performance and responsiveness of the system. CUDA/cuDNN is also crucial for handling the computational demands of YOLOv5 and other deep learning tasks efficiently.
If your Gpu supports CUDA CuDNN (/Nvidia Gpu does) Make sure to research and install CUDA that is perfect fit for your GPU, and Make sure to research and install cuDNN that is a perfect fit for your CUDA version. 

---------------------------------------------------------------------------

### 3. Hardware Requirements:
- GPU: NVIDIA GeForce RTX, with CUDA support for optimal performance.
- CPU: A powerful processor to handle non-GPU accelerated tasks and manage multi-threading efficiently.
- Memory: A minimum of 16GB of RAM is recommended to handle the intensive memory requirements of the model and video processing.

---------------------------------------------------------------------------

### 4. Prerequisites:

- Intermediate Python Programming,
- Basic Knowledge of Machine Learning(neural networks, models like YOLO, and experience with deep learning frameworks such as PyTorch),
- Must know Computer Vision: Experience with OpenCV, image processing, and video capture techniques.
- Understanding of GPU Acceleration: Basic knowledge of how to use GPU resources for deep learning tasks using CUDA and cuDNN.

---------------------------------------------------------------------------

Note:
Make Sure you Directory Structure look like this- 
just use the integrated terminal for your directory and clone yolo5 repo there it self. Just to cross check make sure the yolo5 folder has (hubconf.py) , and inside yolo5 there will be a segment folder and inside that make sure you have (__init__.py).

![image](https://github.com/user-attachments/assets/789fd95c-0d39-4198-ad29-d16bdeadad36)


---------------------------------------------------------------------------
# LETS DIG IN
---------------------------------------------------------------------------
Step 1. Import libraries (Do read the documents i mentioned above if you are not familiar with these libraries).
``` 
import torch
import cv2
import numpy as np
import mediapipe as mp
```
or
```
pip install torch opencv-python-headless numpy mediapipe
```
Quick Explanation- torch is use for loading the YOLOv5 model, which is used for our object detection. The torch library is part of PyTorch, a popular machine learning library. cv2 is a part of the OpenCV library, which is required for video capturing, image processing, and displaying the video feed. Numpy is a fundamental package for scientific computing in Python. It is used here for image and video data manipulation. Mediapipe is a library that was developed by Google for machine learning solutions like face and hand detection.
 
---------------------------------------------------------------------------

Step 2. Loading the YOLOv5 Model
```
model = torch.hub.load('D:/ObjectAndHandRecognition/ObjDectection/yolov5', 'custom', path='yolov5x.pt', source='local')
```
Explanation- YOLO (You Only Look Once) is a popular object detection model. We are loading a custom-trained YOLOv5 model from a local path. (torch.hub.load) This function allows us to load the YOLOv5 model using Torch's Hub. The model is loaded from the specified directory and is ready for inference.

Note 1 : Ensure the YOLOv5 model (yolov5x.pt) is placed in the specified directory as I mentioned earlier. Make sure that PyTorch is installed with CUDA support if you have a compatible GPU
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

```
Note 2 : To use GPU acceleration, ensure that CUDA and cuDNN are installed on your machine. You can check CUDA installation using
```
nvidia-smi #in CMD
```

---------------------------------------------------------------------------

Step 3. MediaPipe Initialization
```
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
```
Explanation- MediaPipe is initialized for hand detection. (mp_hands) handles the detection, and (mp_drawing) helps in drawing the landmarks on the detected hands and its finger. (Yes works for your legs as well xD).

---------------------------------------------------------------------------

Step 4. Loading the Haar Cascade for Face Detection
```
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```
Explanation- I am using a pre-trained Haar Cascade model provided by OpenCV for detecting faces. The (haarcascade_frontalface_default.xml) file is a model that can detect frontal faces in an image.

---------------------------------------------------------------------------

Step 5. Opening the Webcam
Note: If you have a webcam its good, but I used Droid cam as a webcam, If you don't know, Droid cam helps you to use cam service with using your phones camera as a web cam, You just have to download droid cam in both your device(phone and laptop) and just connect them through same wifi. [Droid Cam Download Link](https://www.dev47apps.com/)
```
print("Opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
else:
    print("Webcam opened successfully.")
```
Explanation- cv2.VideoCapture(0, cv2.CAP_DSHOW): This line initializes the webcam using DirectShow on Windows. The webcam will be used as the video source. The code checks if the webcam is successfully opened and shows us an error message if it isn't.

Notes: Ensure that your webcam is connected and working. DirectShow (cv2.CAP_DSHOW) is used here for Windows systems. If you encounter any issues with the webcam, ensure your camera drivers are updated.

---------------------------------------------------------------------------

Step 6. Displaying the Video Feed
```
cv2.namedWindow("Hand, Face, and Object Detection - Pranav v2", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand, Face, and Object Detection - Pranav v2", 640, 480)
```
Explanation- (cv2.namedWindow) Creates a pop up window and c(v2.resizeWindow) Sets the initial size to 640x480 you can resize if you want.

---------------------------------------------------------------------------

Step 7. Main Loop for Video Processing

```
def main_loop():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        height, width, channels = frame.shape
        frame_count += 1

        if frame_count % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(rgb_frame)
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
```
Explanation-

main_loop() is the loop that captures video frames, processes them for object, face, and hand detection, and displays the output. 
cap.read() Captures the current frame from the webcam. 
process_frame(frame, results) this function processes the frame to draw bounding boxes around detected objects.
mp_drawing.draw_landmarks Draws landmarks on detected hands.
face_cascade.detectMultiScale Detects faces in the frame.
cv2.imshow Displays the video with all the detections.
cv2.waitKey(1) Allows the user to exit the loop by pressing the 'q' key change the key according to your liking.
cap.release() and cv2.destroyAllWindows() Ensures that the video capture is released and all windows are closed when the program exits.

---------------------------------------------------------------------------
                             WE ARE DONE
---------------------------------------------------------------------------
    CONSIDER FOLLOWING MY GITHUB PROFILE FOR MORE AWSOME PROJECTS
---------------------------------------------------------------------------
