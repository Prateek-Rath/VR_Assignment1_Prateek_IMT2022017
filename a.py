# Python program to explain cv2.blur() method 

# importing cv2 
import cv2 
import matplotlib.pyplot as plt

# path 
path =  './coins2.jpeg'

# Reading an image in default mode 
image = cv2.imread(path)

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ksize 
ksize = (10, 10) 

# Using cv2.blur() method 
blur_image = cv2.blur(image, ksize) 
blur_gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)

# Displaying the image

# cv2.imshow("light", image)
plt.imshow(blur_gray_image, cmap='gray') # we need to exlicitly tell matplotib about the cmap
plt.axis('off')
plt.title('blurred grayscale coins')
plt.show()
plt.close()


edge = cv2.Canny(gray_image, 200, 250)
plt.axis('off')
# plt.title('original')
# plt.imshow(blur_gray_image,  cmap='gray') 
plt.title('edge')
plt.imshow(edge, cmap='gray') 

plt.show()
plt.close()


