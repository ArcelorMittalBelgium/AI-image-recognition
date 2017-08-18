---
title: "Image Operations"
sequence: 4
---

OpenCV is the most widely used Computer Vision Library out there. OpenCV is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.It is very powerful and offers many features suiting the needs of anyone requiring Image Processing.

# Installation

You can install OpenCV in your linux system in two ways: from pre-built binaries available in linux repositories. 

 `sudo apt-get install libopencv-dev`
 `sudo apt-get install python-opencv`

Or compile it from the source [link](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).

More information on the installation procedure on Windows can be found in the following [link](http://docs.opencv.org/3.2.0/d5/de5/tutorial_py_setup_in_windows.html).

# Online tutorials 

There are many online tutorials available on how to use the OpenCV framework. 
The following link describe a set of basic [OpenCV python tutorials](http://docs.opencv.org/3.2.0/d6/d00/tutorial_py_root.html)


# Code blocks
Before you get started it is important to load to correct python packages.

    import cv2
    import numpy as np
 
### Accessing and Modifying pixel values

Image properties include number of rows, columns and channels, type of image data, number of pixels etc.
The img is a numpy array where you can access the single pixel values or channels. 

    img = cv2.imread('image_sample.jpg')
    cv2.imshow(img)
 
### colorspace
For color conversion, we use the function cv2.cvtColor(input_image, flag) where flag determines the type of conversion.

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow(img)
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

 
 

### geometric transformations
    # rotation
    # translation
    # cropping
    # scaling

# image histograms

# image thresholding

# image smoothing/ blurring

# morphological operations

# edge detection + contour detection
