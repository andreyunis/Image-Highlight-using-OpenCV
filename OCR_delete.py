import cv2
import pytesseract

# Function to recognize characters in an image using Tesseract OCR
def recognize_characters(image_path):
    # Load the image using OpenCV
    image1 = cv2.imread(image_path)
    image = resize_image(image1)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image (if needed)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use pytesseract to recognize characters from the image
    text = pytesseract.image_to_string(thresh,config='-c tessedit_char_whitelist=' + allowlist)
    
    return text

# Function to delete characters from an image based on recognized text
def delete_characters(image_path):
    # Load the image using OpenCV
    image1 = cv2.imread(image_path)
    image = resize_image(image1)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image (if needed)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Use pytesseract to recognize characters from the image
    text = pytesseract.image_to_string(thresh)

    # Get bounding boxes for recognized characters
    boxes = pytesseract.image_to_boxes(gray)
    # Iterate over recognized characters and replace them with white pixels
    for b in boxes.splitlines():
        b = b.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        image[y:h, x:w] = 0 # Set region to white (erase characters)
    
    # Display the modified image
    cv2.imshow('Modified Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(image):
    height, width, channel = image.shape[:3]
    desired_width = 313
    aspect_ratio = desired_width/width
    desired_height = int(aspect_ratio*height)
    dim =(desired_width, desired_height)

    #Resize Image
    image = cv2.resize(image, dsize=dim, interpolation=cv2.INTER_AREA)
    return image


# Example usage:
if __name__ == "__main__":
    image_path = r'Image\input.jpg'
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890.'
    recognized_text = recognize_characters(image_path)
    print("Recognized Text:")
    print(recognized_text)
    # Uncomment the line below to delete characters from the image
    delete_characters(image_path)