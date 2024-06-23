import cv2
import numpy as np

# Read the image
image_path = r'Image\input.jpg'
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    print(f"Error: Unable to load image from '{image_path}'")
    exit()

#Calculate the image height, width and channel
height, width, channel = image.shape[:3]

# Calculate the image size
size = image.size

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find lines in the edge-detected image using the probabilistic Hough transform depending the size of the image
if size < 100210:
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=2)
else:
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=50, maxLineGap=12)

# Create a mask to store segmented lines
line_mask = np.zeros_like(edges)

# Define minimum thresholds for horizontal highlighting depending the size of the image
if size < 100210:
    min_horizontal_length = 30 # Minimum length of a horizontal line to be highlighted when the size is small
else:
    min_horizontal_length = 80 # Minimum length of a horizontal line to be highlighted when the size is big

# Define maximum thresholds for horizontal highlighting 
max_horizontal_length = 150  # Maximum length of a horizontal line to be highlighted

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


# Function to rotate a point (x, y) around a center by a given angle
def rotate_point(point, center, angle):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([[point[0]], [point[1]], [1]]))
    return int(rotated_point[0][0]), int(rotated_point[1][0])

# Generate 5 random angles for rotation
random_angles = np.random.randint(-180, 180, size=5)

# Perform the same highlighting operation on the original and rotated images
highlighted_images = []
for angle in [0] + list(random_angles):
    # Rotate the image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    # Rotate the lines and highlight them on the rotated image
    highlighted_image = rotated_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Rotate the endpoints of the line segment
            rotated_x1, rotated_y1 = rotate_point((x1, y1), center, angle)
            rotated_x2, rotated_y2 = rotate_point((x2, y2), center, angle)
            # Calculate the length of the rotated line segment
            rotated_length = np.sqrt((rotated_x2 - rotated_x1) ** 2 + (rotated_y2 - rotated_y1) ** 2)
            # Calculate the angle of the line segment with respect to the horizontal axis in the original image
            angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Check if the line segment is not nearly horizontal or if it is a long horizontal line
            if abs(rotated_length <= max_horizontal_length) and not x1 == x2:
                # Calculate the slope of the line detected
                slope = abs((y2 - y1) / (x2 - x1 + 1e-6))
                # Eliminate lines vertical or horizontal smaller than the minimum and highlight the rest
                if not( slope < 0.1 or slope > 10) or (rotated_length >= min_horizontal_length):
                   cv2.line(highlighted_image, (rotated_x1, rotated_y1), (rotated_x2, rotated_y2), (0, 255, 0), 2)
    
    # Store the highlighted image
    highlighted_images.append(highlighted_image)

# Draw the detected lines on a copy of the original image
lines_image = np.copy(image)
lines_image[line_mask != 0] = [0, 255, 0]

# Display the result and the Original images
cv2.imshow("Highlighted Line Segments", lines_image)
cv2.imshow('Original Image', image)

# Save Highlighted image an the original in Image Result folder
cv2.imwrite("Image Result/Highlighted Line Segments.jpg", lines_image)
cv2.imwrite(r'Image Result\Original Image.jpg', image)

# Iterate each rotated image generated, show it and save it in Image Result folder
for i, (angle, highlighted_image) in enumerate(zip([0] + list(random_angles), highlighted_images)):
    cv2.imshow(f'Rotated Image {i} (Angle: {angle} degrees)', highlighted_image)
    print(cv2.imwrite(f'Image/Rotated Image {i}.jpg', highlighted_image))

# Press any key to quit
cv2.waitKey(0)
cv2.destroyAllWindows()
