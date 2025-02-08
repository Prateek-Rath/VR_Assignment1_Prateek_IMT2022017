# Python program to explain cv2.blur() method 

# importing cv2 
import cv2 
import matplotlib.pyplot as plt
from skimage import exposure, filters, color
import numpy as np

# path 
path =  './coins2.jpeg'

# Reading an image in default mode 
image = cv2.imread(path)

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# the order is supposed to be:
# grayscale
# histogram equalization -- skip as it introduces artifacts in the image!!
# gaussian blur

ksize = (5, 5) 
# gray_image = cv2.equalizeHist(gray_image)
blur_gray_image = cv2.blur(gray_image, ksize)
# blur_gray_image = cv2.equalizeHist(blur_gray_image) 
# blur_gray_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)



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
plt.title('canny edges')
plt.imshow(edge, cmap='gray') 
plt.show()
plt.close()


# first we'll do a thresholding based segmentation
# we'll first visualize the intensity histogram of the grayscale image
# plt.hist(blur_gray_image, 20)
# plt.title('histogram of blur gray intensities')
# plt.show()
# plt.close()

#let's try various thresholds
th=245
_, mask = cv2.threshold(blur_gray_image, th, 255, cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title(f'th = {th}')
plt.show()
plt.close()


# threshold on the normal gray image not blurred
th=245
_, mask = cv2.threshold(gray_image, th, 255, cv2.THRESH_BINARY)
plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
plt.title(f'th = {th}')
plt.show()
plt.close()


# the above thing is better because there is no 'noise' in the image

#  Unlike thresholding, which uses intensity or color values to separate objects from the background, 
# region-based segmentation considers the relationships between pixels to group them into coherent regions.
# so we'll use region based segmentation and see what we get

# first we'll use the watershed algorithm:
# source: https://medium.com/softplus-publication/image-segmentation-part-1-7adcdab5b375

# The key steps of the watershed algorithm are as follows:

#1. Gradient Computation: Calculate the gradient magnitude of the image, representing the local variations in intensity.
# we'll use a sobel operator to calculate gradients

scale = 1
delta = 0
ddepth = cv2.CV_16S # The "depth" of an image refers to the number of bits used to represent each pixel in the image.
grad_x = cv2.Sobel(blur_gray_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(blur_gray_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
    

    
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
plt.imshow(cv2.cvtColor(grad, cv2.COLOR_BGR2RGB))
plt.title('weighted gradient') 
plt.axis('off')
plt.show()
plt.close()


# to get the markers for the next step we need to view the histogram i.e. count of pixels vs pixel value
plt.hist(blur_gray_image)
plt.show()
plt.close()

#2. Marker Generation: Identify markers, which are seeds or starting points for the segmentation. 
# These markers can be user-defined or automatically generated based on certain criteria.
markers = np.zeros_like(grad)
markers[blur_gray_image < 200] = 1
markers[blur_gray_image > 240] = 2 
# these are pretty unclear but I'm assuming that 200 and 230 work well for image2


# print(type(markers))
# print(markers)
# plt.imshow(markers, cmap=plt.cm.nipy_spectral)
plt.imshow(markers)
plt.axis('off')
plt.title(markers)
plt.show()
plt.close()





#3. Flood Filling: Starting from the markers, perform a “flood filling” process that assigns each pixel to the nearest marker.
print(type(image))
print(type(markers))
markers = np.int32(markers)
seg = cv2.watershed(image, markers)
plt.imshow(seg, cmap='gray')
plt.axis('off')
plt.title('watershed segmentation')
plt.show()
plt.close()

#4. Segmentation: The flood-filled regions are then merged together to form the final segmented regions.

# clearly watershed gives us shit results as compared to canny 


# now clustering

