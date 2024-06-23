import cv2
import numpy as np

# Read the image
image_path = 'Image/input noisy dashed.jpg'
image = cv2.imread(image_path)

# Check if image is loaded successfully
if image is None:
    print(f"Error: Unable to load image from '{image_path}'")
    exit()

#Calculated size image/Calcula el tamaño de la imagen
height, width, channel = image.shape[:3]
size = image.size
print(size)
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Find lines in the edge-detected image using the probabilistic Hough transform
# Se agrego una condicion en base al tamaño detectado
if size < 100000:
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=11)
else:
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=10, minLineLength=50, maxLineGap=10)



# Create a mask to store segmented lines
line_mask = np.zeros_like(edges)

# Define thresholds for horizontal highlighting
# Se bajo el tamaño minimo para ser compatible con ambas imagenes
min_horizontal_length = 50
 # Minimum length of a horizontal line to be highlighted
max_horizontal_length = 300  # Maximum length of a horizontal line to be highlighted

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


# Function to rotate a point (x, y) around a center by a given angle a
def rotate_point(point, center, angle):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([[point[0]], [point[1]], [1]]))
    return int(rotated_point[0]), int(rotated_point[1])

# Generate 5 random angles for rotation a
random_angles = np.random.randint(-180, 180, size=5)

# Perform the same highlighting operation on the original and rotated images a
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
            # if abs(angle_deg) > 10 or rotated_length > min_horizontal_length:
            #     cv2.line(highlighted_image, (rotated_x1, rotated_y1), (rotated_x2, rotated_y2), (0, 255, 0), 2)
            if abs(rotated_length > min_horizontal_length and rotated_length < max_horizontal_length) and not x1 == x2:
                cv2.line(highlighted_image, (rotated_x1, rotated_y1), (rotated_x2, rotated_y2), (0, 255, 0), 2)
    
    # Store the highlighted image
    highlighted_images.append(highlighted_image)

# Draw the detected lines on a copy of the original image
lines_image = np.copy(image)
lines_image[line_mask != 0] = [0, 255, 0]

# Display the result
cv2.imshow("Lines", lines_image)
cv2.imshow('Original Image', image)
cv2.imshow('gray', gray)
cv2.imshow('edges', edges)

for i, (angle, highlighted_image) in enumerate(zip([0] + list(random_angles), highlighted_images)):
    cv2.imshow(f'Rotated Image {i} (Angle: {angle} degrees)', highlighted_image)
    
cv2.waitKey(0)
cv2.destroyAllWindows()
