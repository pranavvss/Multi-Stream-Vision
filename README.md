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
i just use the integrated terminal for your directory and clone yolo5 repo there it self. Just to cross check make sure the yolo5 folder has (hubconf.py) , and inside yolo5 there will be a segment folder and inside that make sure you have (__init__.py).

![image](https://github.com/user-attachments/assets/789fd95c-0d39-4198-ad29-d16bdeadad36)


