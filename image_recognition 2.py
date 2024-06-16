import cv2
import numpy as np

# Read the image
image_path = 'Image\input.jpg'
original_image = cv2.imread(image_path)
# Convert the original image to grayscale
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to draw filtered contours
mask = np.zeros_like(edges)

# Filter contours based on area and aspect ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)
    # Filter based on aspect ratio and area
    if 0.2 < aspect_ratio < 5 and area > 100:
        cv2.drawContours(mask, [contour], -1, 255, -1)

# Combine mask with edges
filtered_edges = cv2.bitwise_and(edges, edges, mask=mask)

# Find lines (segments) in the filtered edge image
lines = cv2.HoughLinesP(filtered_edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=10, maxLineGap=10)

# Function to rotate a point (x, y) around a center by a given angle
def rotate_point(point, center, angle):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_point = np.dot(rotation_matrix, np.array([[point[0]], [point[1]], [1]]))
    return int(rotated_point[0]), int(rotated_point[1])

# Generate 5 random angles for rotation
random_angles = np.random.randint(-180, 180, size=5)

# Minimum length for horizontal lines to be highlighted
min_horizontal_length = 50

# Perform the same highlighting operation on the original and rotated images
highlighted_images = []
for angle in [0] + list(random_angles):
    # Rotate the image
    center = (original_image.shape[1] // 2, original_image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(original_image, rot_mat, original_image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
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
            if abs(angle_deg) > 10 or rotated_length > min_horizontal_length:
                cv2.line(highlighted_image, (rotated_x1, rotated_y1), (rotated_x2, rotated_y2), (0, 255, 0), 2)
    
    # Store the highlighted image
    highlighted_images.append(highlighted_image)

# Display the original image and the rotated images along with the highlighted line segments
cv2.imshow('Original Image', original_image)
for i, (angle, highlighted_image) in enumerate(zip([0] + list(random_angles), highlighted_images)):
    cv2.imshow(f'Rotated Image {i} (Angle: {angle} degrees)', highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()