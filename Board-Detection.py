import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


image = cv2.imread(r"images/image2.jpg")
#opencv reads images as BGR not RGB !

#grayscale
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#RGB
rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#Gaussian BLUR
gaussian_blur = cv2.GaussianBlur(gray_image,(3,3),0) # (5,5) is kernel size and 0 is std deviation, changed to (3,3) for smoothness

#OTSU THRESHHOLD
ret,otsu_binary = cv2.threshold(gaussian_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # it returns the threshhold used (which is ret ) and the binary image.

#Canny Edge detection
canny = cv2.Canny(gaussian_blur, 50,150) #started with 20, 255 on otsu but didnt really look good, trying it on gaussian to see if we get better results, better results on gaussian

# First Dilation
kernel = np.ones((7,7), np.uint8)
img_dilation = cv2.dilate(canny,kernel, iterations=1)

# the letters of each square are weird so im attempting to remove them,
#this gives us statistics about each connected component in the image with background which is alwasy at index 0
num_labels, labels , stats, centroids = cv2.connectedComponentsWithStats(img_dilation,4,cv2.CV_32S)
#This grabs the area of every component other then  background at 0
areas  = stats[1:,cv2.CC_STAT_AREA]
#FInd the index of the largest area
largest_component= np.argmax(areas) + 1
#Create a black image with the same size as our original image
cleaned_img = np.zeros(image.shape,dtype = np.uint8)
#Turn the pixels in the biggest component to white
cleaned_img[labels == largest_component] = 255

# morphological didnt work with either image

#hough lines
hough_iamge = cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2GRAY)

lines = cv2.HoughLinesP(hough_iamge,1,np.pi/180,threshold=200, minLineLength = 100, maxLineGap = 50)

if lines is not None:
    for i, line in enumerate(lines):
        x1,y1,x2,y2 = line[0]

        cv2.line(hough_iamge,(x1,y1),(x2,y2),(255,255,255),2)

#running dilation again

kernel = np.ones((3,3),np.uint8)

img_dilation2 = cv2.dilate(hough_iamge,kernel,iterations=1)






plt.subplot(3,3,1)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

plt.subplot(3,3,2)
plt.imshow(gray_image, cmap= 'gray') # have to put cmap = gray else output is wrong
plt.title("GrayScale")
plt.axis("off")

plt.subplot(3,3,3)
plt.imshow(gaussian_blur, cmap = "gray")
plt.title("Gaussian")
plt.axis("Off")

plt.subplot(3,3,4)
plt.imshow(otsu_binary, cmap="gray")
plt.title("OTSU")
plt.axis("off")



plt.subplot(3,3,5)
plt.imshow(canny, cmap="gray")
plt.title("canny")
plt.axis("off")

plt.subplot(3,3,6)
plt.imshow(img_dilation, cmap="gray")
plt.title("dilation")
plt.axis("off")

plt.subplot(3,3,7)
plt.imshow(cleaned_img, cmap="gray")
plt.title("cleaned")
plt.axis("off")

plt.subplot(3,3,8)
plt.imshow(hough_iamge, cmap="gray")
plt.title("hough")
plt.axis("off")

plt.subplot(3,3,9)
plt.imshow(img_dilation2, cmap="gray")
plt.title("dilation2")
plt.axis("off")



plt.show()

