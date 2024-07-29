import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_segmentation_coordinates(contours): # input is desired contours
    coordinates = []
    for contour in contours:
        for point in contour:
            coordinates.append(tuple(point[0]))

    return coordinates


def segment_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's method for thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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
    outlined_frame = frame.copy()
    cv2.drawContours(outlined_frame, desired_contours, -1, (0, 255, 0), cv2.FILLED)

    return desired_contours

