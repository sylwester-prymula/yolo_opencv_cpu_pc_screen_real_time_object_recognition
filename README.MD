# yolo_opencv_cpu_pc_screen_real_time_object_recognition

Introduction
--------------------------
Real-time object recognition from the captured computer screen. \
The program uses the CPU.

On the right a source e.g. with a browser window, on the left a window with captured data and with recognized objects.

![YOLO Real-time object recognition from the captured computer screen - example 1](https://github.com/sylwester-prymula/yolo_opencv_cpu_pc_screen_real_time_object_recognition/blob/master/example/example.png)

Video example 1: https://github.com/sylwester-prymula/yolo_opencv_cpu_pc_screen_real_time_object_recognition/blob/master/example/1.mp4

Video example 2: https://github.com/sylwester-prymula/yolo_opencv_cpu_pc_screen_real_time_object_recognition/blob/master/example/2.mp4

My code was based on: \
https://pysource.com/2019/07/08/yolo-real-time-detection-on-cpu/

YOLO (You Only Look Once) is a deep learning algorythm. \
More information: https://pjreddie.com/darknet/yolo/

Installation and requirements
--------------------------
1. https://www.python.org/ - Python - Install the `AMD64/EM64T/x64` version
2. `pip install opencv-contrib-python` - OpenCV packages for Python
3. `pip install Pillow` - Python Imaging Library (Fork)
4. - yolov3.weights - https://pjreddie.com/media/files/yolov3.weights
    - yolov3.cfg - https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
    - coco.names - https://github.com/pjreddie/darknet/blob/master/data/coco.names
