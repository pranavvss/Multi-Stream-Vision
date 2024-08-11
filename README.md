# Multi-Stream-Vision-Real-Time-Object-Face-and-Hand-Detection- (NOT FOR BEGINNERS)
 An advanced real-time detection system capable of recognizing multiple objects, hands, and faces within a video stream. (Could be also done on a Pre-recorded video)

### Video Example-

https://github.com/user-attachments/assets/1d4c14ac-b2b7-43b4-8250-7fa2d30a7297

### Project Info- 

This project involves developing an advanced real-time detection system capable of recognizing multiple objects, hands, and faces within a video stream. Leveraging the YOLOv5 model for object detection, MediaPipe for hand tracking, and OpenCV Haar Cascade for face detection, the system efficiently processes video feed from a webcam. This project demonstrates the integration of powerful computer vision tools and techniques to create a versatile and responsive detection system. The project also uses CUDA and cuDNN for GPU acceleration, ensuring the system operates with minimal lag, providing a smoother and faster user experience.

### Project Requirements

1. Programming Language: Python (Somewhere between 3.11 to 3.12) [Download Pythin](https://www.python.org/downloads/)
2. Libraries and Frameworks i have used:
- OpenCV: Used for video capture, image processing, and Haar Cascade for face detection. [Download](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

- Torch (PyTorch): A deep learning framework used to load and run the YOLOv5 model. [Download](https://pytorch.org/docs/stable/index.html)

- YOLOv5: A state-of-the-art object detection model, renowned for its speed and accuracy.
  [All About Yolo5](https://docs.ultralytics.com/yolov5/)
  [Git clone/Download yolo5](https://github.com/ultralytics/yolov5), Note: Make sure Yolo5 is in the same directory where you will be saving the Pyhton script.
  
- MediaPipe: Utilized for real-time hand detection and tracking.  [Download](https://ai.google.dev/edge/mediapipe/solutions/guide)
- 
- GStreamer: An advanced pipeline-based multimedia framework used to ensure high-quality video input and output.
  [Download](https://gstreamer.freedesktop.org/download/#windows), Note: Download both Runtime installer and Development installer, Let the program download the files in default directory.
  
- CUDA and cuDNN: Libraries that enable GPU acceleration for deep learning tasks, improving the performance and responsiveness of the system.
  If you Gpu supports CUDA CuDNN (/Nvidia Gpu does) Make sure to research and install CUDA that is perfect fit for you GPU, and Make sure to research and install cuDNN that is a perfect fit for your CUDA version.
