import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('./easy.jpeg')

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

ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
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
img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)


plt.imshow(img, cmap='tab20b')
plt.axis('off')
plt.title('Contours formed')
plt.show()
plt.close()


