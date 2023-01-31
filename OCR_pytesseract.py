import cv2
import pytesseract

# Load image
image = cv2.imread("text.jpg")

# Get text from image using pytesseract
text = pytesseract.image_to_string(image)

print(text)
