import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


image = cv2.imread(r"images/image2.jpg")
#opencv reads images as BGR not RGB !

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

gaussian_blur = cv2.GaussianBlur(gray_image,(3,3),0) # (5,5) is kernel size and 0 is std deviation, changed to (3,3) for smoothness


ret,otsu_binary = cv2.threshold(gaussian_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # it returns the threshhold used (which is ret ) and the binary image.


plt.subplot(2,2,1)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(gray_image, cmap= 'gray') # have to put cmap = gray else output is wrong
plt.title("GrayScale")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(gaussian_blur, cmap = "gray")
plt.title("Gaussian")
plt.axis("Off")

plt.subplot(2,2,4)
plt.imshow(otsu_binary, cmap="gray")
plt.title("OTSU")
plt.axis("off")



plt.show()

