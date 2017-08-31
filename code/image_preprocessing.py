
import cv2
import numpy as np

img = cv2.imread('test_sample.jpg')
cv2.imshow('image_sample',img)
cv2.waitKey(0)

scaled_img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('image_sample',scaled_img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image_sample',gray_img)
cv2.waitKey(0)

rows,cols = gray_img.shape
M = cv2.getRotationMatrix2D((50,100),90,1)
rotated_img = cv2.warpAffine(gray_img,M,(cols,rows))
cv2.imshow('image_sample',rotated_img)
cv2.waitKey(0)

cropped_img = gray_img[20:500, 10:250]
cv2.imshow('image_sample',cropped_img)
cv2.waitKey(0)

from matplotlib import pyplot as plt

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

equalized_img = cv2.equalizeHist(gray_img)
cv2.imshow('image_sample',equalized_img)
cv2.waitKey(0)

ret,thresholded_img = cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY)
cv2.imshow('image_sample',thresholded_img)
cv2.waitKey(0)

average_blur_img = cv2.blur(img,(5,5))
cv2.imshow('image_sample',average_blur_img)
cv2.waitKey(0)

gaussian_blur_img = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('image_sample',gaussian_blur_img)
cv2.waitKey(0)

median_blur_img = cv2.medianBlur(img,5)
cv2.imshow('image_sample',median_blur_img)
cv2.waitKey(0)

bilateral_blur_img = cv2.bilateralFilter(img,5,100,100 )
cv2.imshow('image_sample',bilateral_blur_img)
cv2.waitKey(0)

kernel = np.ones((10,10),np.uint8)
eroded_img = cv2.erode(img,kernel,iterations = 1)
cv2.imshow('image_sample',eroded_img)
cv2.waitKey(0)

kernel = np.ones((10,10),np.uint8)
dilated_img = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow('image_sample',dilated_img)
cv2.waitKey(0)

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('image_sample',hsv_img)
cv2.waitKey(0)

edges = cv2.Canny(img,100,150)
cv2.imshow('image_sample',edges)
cv2.waitKey(0)
