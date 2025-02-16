import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('coins2_cropped.jpeg')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to remove noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply thresholding
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_area = 500
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw bounding rectangles around each coin
segmented_image = image.copy()

for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the segmented coins
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Extract individual coins
coin_images = []

for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    coin = image[y:y+h, x:x+w]
    coin_images.append(coin)

# Display the individual coins
for idx, coin in enumerate(coin_images):
    plt.subplot(1, len(coin_images), idx+1)
    plt.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    plt.axis('off')
plt.show()
