import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


image = cv2.imread(r"images/image1.jpg")
#opencv reads images as BGR not RGB !

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

plt.subplot(1,2,1)
plt.imshow(rgb_image)
plt.title("RGB")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gray_image, cmap= 'gray') # have to put cmap = gray else output is wrong
plt.title("GrayScale")
plt.show()