import cv2
import easyocr

# Set the path to your image
image_path = r"C:\Users\Lathika\OneDrive\Desktop\YOLOv5\plate_test.webp"

# Load the image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Image not loaded. Check if the file path is correct and the file exists.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur and detect edges
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 100, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plate_img = None

# Loop through contours to find license plate-like shapes
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv2.contourArea(contour)

    if 2 < aspect_ratio < 6 and 1000 < area < 25000:
        plate_img = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        break

# OCR with EasyOCR
if plate_img is not None:
    reader = easyocr.Reader(['en'])
    results = reader.readtext(plate_img)

    for (bbox, text, prob) in results:
        print(f"Detected Plate Number: {text} (Confidence: {prob:.2f})")

    # Save cropped plate
    cv2.imwrite(r"C:\Users\Lathika\OneDrive\Desktop\YOLOv5\cropped_plate.jpg", plate_img)
else:
    print("Warning: No plate-like region detected.")

# Save the result with bounding box
cv2.imwrite(r"C:\Users\Lathika\OneDrive\Desktop\YOLOv5\plate_boxed.jpg", image)
