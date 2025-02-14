import cv2
import matplotlib.pyplot as plt


image = cv2.imread('coins2_cropped.jpeg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ksize = (5, 5)
blur_gray_image = cv2.blur(gray_image, ksize)

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
# plt.hist(blur_gray_image)
# plt.show()
# plt.close()

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
# print(type(image))
# print(type(markers))
markers = np.int32(markers)
seg = cv2.watershed(image, markers)
plt.imshow(seg, cmap='gray')
plt.axis('off')
plt.title('watershed segmentation')
plt.show()
plt.close()

#4. Segmentation: The flood-filled regions are then merged together to form the final segmented regions.

# clearly watershed gives us bad results as compared to canny 
