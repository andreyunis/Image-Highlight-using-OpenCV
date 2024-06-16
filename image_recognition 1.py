import cv2
import numpy as np

# Read the image
image_path = 'Image\input noisy.jpg'
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    print(f"Error: Unable to load image from '{image_path}'")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Find lines in the edge-detected image using the probabilistic Hough transform
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20, minLineLength=50, maxLineGap=10)

# Create a mask to store segmented lines
line_mask = np.zeros_like(edges)

# Define thresholds for horizontal highlighting
min_horizontal_length = 80 # Minimum length of a horizontal line to be highlighted
max_horizontal_length = 200  # Maximum length of a horizontal line to be highlighted

# Filter and draw non-vertical and non-horizontal lines, handling overlapping segments
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the absolute slope of the line
        slope = abs((y2 - y1) / (x2 - x1 + 1e-6))  # Adding a small value to avoid division by zero
        
        # Skip lines close to horizontal or vertical
        if slope < 0.1 or slope > 10:
            length = abs(x2 - x1)
            # Highlight horizontal lines within specified length range
            if min_horizontal_length < length < max_horizontal_length:  
                # Check for overlapping segments
                if np.any(line_mask[y1:y2, x1:x2] != 0):
                    continue
                # Draw the line on the mask
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            continue
        
        # Check for overlapping segments
        if np.any(line_mask[y1:y2, x1:x2] != 0):
            continue
        
        # Draw the line on the mask
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

# Draw the detected lines on a copy of the original image
lines_image = np.copy(image)
lines_image[line_mask != 0] = [0, 255, 0]

# Display the result
cv2.imshow("Highlighted Line Segments", lines_image)
cv2.waitKey(0)
cv2.destroyAllWindows()