# Python program to explain cv2.blur() method 

# importing cv2 
import cv2 
import matplotlib.pyplot as plt
from skimage import exposure, filters, color
import numpy as np

# path 
path =  './coins2_cropped.jpeg'

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


edges = cv2.Canny(gray_image, 200, 250)
plt.axis('off')
# plt.title('original')
# plt.imshow(blur_gray_image,  cmap='gray') 
plt.title('canny edges')
plt.imshow(edges, cmap='gray') 
plt.show()
plt.close()


contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# plt.axis('off')
# plt.title('contours')
# plt.imshow(contours)
# plt.show()
# plt.close()

# filter contours
min_area = 2
valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
number_of_coins = len(valid_contours)
print(number_of_coins)


# draw the contours on the image
output_image = image.copy()
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 1)  # the last two parameters are color and thickness

plt.title('Contours around Coins')
plt.imshow(output_image)
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


# region based segmentation

# now we'll try clustering
k= 10
img_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image into a 2D array of pixels and 3 color values (RGB)
print('the image shape is', img_convert.shape)
vectorized = img_convert.reshape((-1,3)) # this is 2 * 2
print('the new shape is', vectorized.shape)
vectorized = np.float32(vectorized)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
# the criteria is to use a change of less than epsilon and a max number of iterations



# let's try and find the optimal k using elbow method
# arr = []
# k_range = []
# for k in range(1, 10):
#     ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     result_image = res.reshape((img_convert.shape))
#     arr.append(ret)
#     k_range.append(k)

# plt.plot(k_range, arr, marker='o')
# plt.title('Elbow method for optimal k')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('ret value from cv2.kmeans')
# plt.show()

# we choose k = 2 to separate the foreground from the background
k= 2

img_convert = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # opencv uses bgr but we want rgb

# Reshape the image into a 2D array of pixels and 3 color values (RGB)
vectorized = image.reshape((-1,3)) # this is 2 * 2
vectorized = np.float32(vectorized)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply KMeans
ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
print(center)
result_image = res.reshape((img_convert.shape))

plt.imshow(result_image, cmap='viridis')
plt.axis('off')
plt.title('k means result')
plt.show()
plt.close()


# all the above stuff we do is fine, but we need to 