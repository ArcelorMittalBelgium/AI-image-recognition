---
title: "Image Operations"
sequence: 4
---

OpenCV is the most widely used Computer Vision Library out there. OpenCV is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.It is very powerful and offers many features suiting the needs of anyone requiring Image Processing.

# Installation

You can install OpenCV in your linux system in two ways: from pre-built binaries available in linux repositories. 

> `sudo apt-get install libopencv-dev`
> `sudo apt-get install python-opencv`

Or compile it from the source [link](http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).

More information on the installation procedure on Windows can be found in the following [link](http://docs.opencv.org/3.2.0/d5/de5/tutorial_py_setup_in_windows.html).

# Online tutorials 

There are many online tutorials available on how to use the OpenCV framework. 
The following link describe a set of basic [OpenCV python tutorials](http://docs.opencv.org/3.2.0/d6/d00/tutorial_py_root.html)


#Code blocks
Before you get started it is important to load to correct python packages.

    import cv2
    import numpy as np
 
##Accessing and Modifying pixel values

Image properties include number of rows, columns and channels, type of image data, number of pixels etc.
The img is a numpy array where you can access the single pixel values or channels. 

    img = cv2.imread('image_sample.jpg')
    cv2.imshow('image_sample',img)
    cv2.waitKey(0)
 
##Geometric transformations
There are five main geometric transformations that can be performed on an image (i.e.,scaling, translation,rotation,cropping, affine transformation). 

###Scaling
Scaling is just resizing of the image. OpenCV comes with a function cv2.resize() for this purpose. The size of the image can be specified manually, or you can specify the scaling factor. Different interpolation methods are used. Preferable interpolation methods are cv2.INTER_AREA for shrinking and cv2.INTER_CUBIC (slow) or cv2.INTER_LINEAR for zooming. By default, interpolation method used is cv2.INTER_LINEAR for all resizing purposes.  
     
     scaled_img = cv2.resize(img,None,fx=x_scaling_factor, fy=y_scaling_factor, interpolation = cv2.INTER_CUBIC)
     
     scaled_img = cv2.resize(img,(new_width, new_height), interpolation = cv2.INTER_CUBIC
     
###Translation 
Translation is the shifting of object’s location. The translation in the x-direction is taken as t_x whereas the translation in the y-direction is defined by t_y.
   
    rows,cols = img.shape
    M = np.float32([[1,0,t_x],[0,1,t_y]])
    translated_img = cv2.warpAffine(img,M,(cols,rows))

###Rotation
OpenCV provides scaled rotation with adjustable center of rotation so that you can rotate at any location you prefer.To find this transformation matrix, OpenCV provides a function, cv2.getRotationMatrix2D where you define the rotation center and the rotation angle. 

    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((rotationcenter_x,rotationcenter_y),angle,1)
    rotated_img = cv2.warpAffine(img,M,(cols,rows))

###Cropping 
Cropping or selecting regions of interest can be easily done with slicing arrays, this due the numpy array format of the image.

    cropped_img = img[startpoint_x:endpoint_x, startpoint_y:endpoint_y]

###Affine Transformation 
In affine transformation, all parallel lines in the original image will still be parallel in the output image. To find the transformation matrix, we need three points from input image and their corresponding locations in output image. Then cv2.getAffineTransform will create a 2x3 matrix which is to be passed to cv2.warpAffine.

    pts_old_image = np.float32([[pt1_x,pt1_y],[pt2_x,pt2_y],[pt3_x,pt3_y]])
    pts_new_image = np.float32([[pt1_new_x,pt1_new_y],[pt2_new_x,pt2_new_y],[pt3_new_x,pt3_new_y]])
    M = cv2.getAffineTransform(pts_old_image,pts_new_image)
    affine_img = cv2.warpAffine(img,M,(cols,rows))
    
##Image color analysis
###Colorspace conversion
For color conversion, we use the function cv2.cvtColor(input_image, flag) where flag determines the type of conversion.
The following example show the BGR (Blue, Green, Red) conversion to HSV(Hue, Saturation, Value)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

##Image histogram
You can consider histogram as a graph or plot, which gives you an overall idea about the intensity distribution of an image. It is a plot with pixel values (ranging from 0 to 255, not always) in X-axis and corresponding number of pixels in the image on Y-axis.

    from matplotlib import pyplot as plt
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
###Histogram equalization 
Consider an image whose pixel values are confined to some specific range of values only. For eg, brighter image will have all pixels confined to high values. But a good image will have pixels from all regions of the image. So you need to stretch this histogram to either ends (as given in below image, from wikipedia) and that is what Histogram Equalization does (in simple words). This normally improves the contrast of the image.OpenCV has a function to do this, cv2.equalizeHist(). Its input is just grayscale image and output is our histogram equalized image. 
![](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/histogram_equalization.png)
    
    equalized_img = cv2.equalizeHist(img)

##Image thresholding
Different thresholding mechanisms and properties are available in Opencv, the following image gives an overview of the most standard techniques. More detailed information can be found in the following [link](http://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html)
![](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/threshold.jpg)

    ret,thresholded_img = cv2.threshold(img,lower_threshold,upper_threshold,cv2.THRESH_BINARY)
    
##Gamma correction 
Gamma correction, or often simply gamma, is the name of a nonlinear operation used to encode and decode luminance or tristimulus values in video or still image systems.
[Gamme correction](http://www.codepool.biz/image-processing-opencv-gamma-correction.html)

#Image smoothing/ blurring
Image blurring is achieved by convolving the image with a low-pass filter kernel. It is useful for removing noise. It actually removes high frequency content (e.g: noise, edges) from the image resulting in edges being blurred when this is filter is applied. (Well, there are blurring techniques which do not blur edges). OpenCV provides mainly four types of blurring techniques. Averaging is the most general blur technique, Gaussian filtering is highly effective in removing gaussian noise from the image, Median filtering is highly effective against salt-and-pepper noise in the images and bilateral filtering is highly effective in noise removal while keeping edges sharp.
   
    average_blur_img = cv2.blur(img,(kernelsize_x,kernelsize_y))
    gaussian_blur_img = cv2.GaussianBlur(img,(kernelsize_x,kernelsize_y),0)
    median_blur_img = cv2.medianBlur(img,filtersize)
    bilateral_blur_img = cv2.bilateralFilter(img,filter_diameter,sigmaColor,sigmaSpace )


#Morphological operations
Morphological operations apply a structuring element to an input image and generate an output image.The most basic morphological operations are: Erosion and Dilation. 
The applications of this operation are as follows:
- Removing noise
- Isolation of individual elements and joining disparate elements in an image.
- Finding of intensity bumps or holes in an image

##Erosion 
The basic idea is based on soil erosion , it erodes away the boundaries of foreground object.

    kernel = np.ones((kernelsize_x,kernelsize_y),np.uint8)
    eroded_img = cv2.erode(img,kernel,iterations = 1)

##Dilation
Normally, in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. 
So we dilate it. Since noise is gone, they won’t come back, but our object area increases. It is also useful in joining broken parts of an object.

    kernel = np.ones((kernelsize_x,kernelsize_y),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)

# edge detection + contour detection
