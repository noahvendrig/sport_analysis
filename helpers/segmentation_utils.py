import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans, MeanShift

def get_segmentation_coordinates(contours): # input is desired contours
    coordinates = []
    for contour in contours:
        for point in contour:
            coordinates.append(tuple(point[0]))

    return coordinates


def segment_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's method for thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color range for segmentation (adjust these values as needed)
    lower_color = np.array([30, 40, 40])
    upper_color = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)


    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw the desired contours
    inner_region_mask = np.zeros_like(mask)

    # Draw the contours onto the blank mask
    cv2.drawContours(inner_region_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Subtract the inner region mask from the original mask to get the desired region
    desired_region_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_region_mask))

    # Find contours in the desired region mask
    desired_contours, _ = cv2.findContours(desired_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours of the desired region on the original image for visualization
    outlined_img = img.copy()
    cv2.drawContours(outlined_img, desired_contours, -1, (0, 255, 0), cv2.FILLED)

    return desired_contours

def k_cluster(img, n_clusters=2):
    img_2d = np.array(img).reshape(-1, 3)
    k_means = KMeans(n_clusters=n_clusters, random_state=0).fit(img_2d)

    return k_means

def mean_shift(img, bandwidth=2):
    img_2d = img.reshape(-1, 3)
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(img_2d)
    mean_shift_labels = mean_shift.labels_
    mean_shift_clustered_img = mean_shift_labels.reshape(img.shape[0], img.shape[1])

    return mean_shift_clustered_img, mean_shift_labels

def crop_frame(frame, bounding_box):
    return frame[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

def get_img_weighted_avg(img):
    height, width = img.shape
    # Create a weight matrix
    weights = np.zeros((height, width))

    # Assign higher weights to the sides, excluding the top
    for y in range(height):
        for x in range(width):
            if y > 0:  # Exclude the top row
                # Linear weighting: increase weight linearly towards the sides
                weight = max(x / (width / 2), (width - x - 1) / (width / 2))
                weights[y, x] = weight

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Compute the weighted average for the single-channel image
    weighted_avg = np.sum(img * weights)

    return weighted_avg