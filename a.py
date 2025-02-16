# Python program to explain cv2.blur() method 

# importing cv2 
import cv2 
import matplotlib.pyplot as plt
from skimage import exposure, filters, color
import numpy as np

# path 
path =  './coins.jpeg'

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

ret, bin_img = cv2.threshold(blur_gray_image,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

edges = cv2.Canny(bin_img, 50, 250)
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
min_area = 4
valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
number_of_coins = len(valid_contours)
print('acc to canny contours, number of coins is', number_of_coins)


# draw the contours on the image
output_image = image.copy()
cv2.drawContours(output_image, valid_contours, -1, (0, 255, 0), 3)  # the last two parameters are color and thickness

plt.title('Contours around Coins')
plt.imshow(output_image)
plt.show()
plt.close()



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


# let's try and find the optimal k using elbow method --no use as we only want to extort the foreground
arr = []
k_range = []
explore_k = 1
for k in range(1, explore_k):
    ret, label, center = cv2.kmeans(vectorized, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img_convert.shape))
    arr.append(ret)
    k_range.append(k)

plt.plot(k_range, arr, marker='o')
plt.title('Elbow method for optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('ret value from cv2.kmeans')
plt.show()

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


# all the above stuff we do is fine, but we need to get each coin separately for which we use the watershed method
# turns out it is especially useful when the coins are touching as in the image

# Read the image
img = cv2.imread('./coins.jpeg')

# Convert from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, bin_img = cv2.threshold(gray,
                             0, 255, 
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plt.imshow(bin_img, cmap='gray')
plt.axis('off')
plt.title('binary image')
plt.show()
plt.close()


# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # all 1s in 3*3
bin_img = cv2.morphologyEx(bin_img, 
                           cv2.MORPH_OPEN,
                           kernel,
                           iterations=2)


plt.imshow(bin_img, cmap='gray')
plt.axis('off')
plt.title('Binary Image with noise removed')
plt.show()
plt.close()



sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
plt.imshow(bin_img, cmap='gray')
plt.axis('off')
plt.title('Sure background')
plt.show()
plt.close()

dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
plt.imshow(dist, cmap='gray')
plt.axis('off')
plt.title('Distance Transform')
plt.show()
plt.close()

ret, sure_fg = cv2.threshold(dist, 0.45 * dist.max(), 250, cv2.THRESH_BINARY) # 0.45 works for this particular image
sure_fg = sure_fg.astype(np.uint8)  
plt.imshow(sure_fg, cmap='gray')
plt.axis('off')
plt.title('Sure Foreground')
plt.show()
plt.close()


unknown = cv2.subtract(sure_bg, sure_fg)
plt.imshow(unknown, cmap='gray')
plt.axis('off')
plt.title('Unknown area')
plt.show()
plt.close()


ret, markers = cv2.connectedComponents(sure_fg) # find connected components in foreground
print('ret is', ret)
print('type of markers is', type(markers))
print('shape of markers is', markers.shape)
 
# Add one to all labels so that background is not 0, but 1
markers += 1
# mark the region of unknown with zero
markers[unknown == 255] = 0
 
plt.imshow(markers, cmap='tab20b')
plt.axis('off')
plt.title('Markers')
plt.show()
plt.close()




# Watershed Algorithm
markers = cv2.watershed(img, markers)
print('markers looks like', markers) 

plt.imshow(markers, cmap='tab20b')
plt.axis('off')
plt.title('Markers after watershed')
plt.show()
plt.close()
 
 
labels = np.unique(markers)
 
coins = []
for label in labels[2:]:  
 
# Create a binary image in which only the area of the label is in the foreground 
#and the rest of the image is in the background   
    target = np.where(markers == label, 255, 0).astype(np.uint8)
   
  # Perform contour extraction on the created binary image
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coins.append(contours[0])
 
# Draw the outline
img = cv2.drawContours(img, coins, -1, color=(0, 23, 255), thickness=3)
plt.imshow(img, cmap='tab20b')
plt.axis('off')
plt.title('Contours formed')
plt.show()
plt.close()

# the class or region of each pixel can be found be seeing the markers array


# counting number of coins in the image
print('number of coins according to watershed is', len(coins))




