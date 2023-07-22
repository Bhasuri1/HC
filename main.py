import os
from PIL import Image
import pytesseract
from ultralytics import YOLO

# Define the path to Tesseract OCR executable (adjust based on your system)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Load the image for detection
image_path = "c:/Users/bhasu/Downloads/runs/detect/train5/PR_curve.png"
image = Image.open(image_path)

# Perform object detection using YOLO
yolo = YOLO("c:/Users/bhasu/Downloads/runs/detect/train/weights/best.pt")
yolo.conf = 0.25  # Set the detection confidence threshold
results = yolo(image)
#print(results)

# Extract detected text using OCR
detected_texts = []
if results.boxes is not None:
    for det in results.boxes[0]:
        *coords, _, conf, cls = det.tolist()
        if yolo.names[int(cls)] == "text" and conf >= yolo.conf:
            x1, y1, x2, y2 = coords
            cropped_img = image.crop((x1, y1, x2, y2))
            text = pytesseract.image_to_string(cropped_img)
            detected_texts.append(text.strip())

# Define the file path to save the detected texts
output_file_path = "c:/Users/bhasu/Downloads/detected_texts.txt"

# Write the detected texts to a file
with open(output_file_path, "w") as output_file:
    for text in detected_texts:
        output_file.write(text + "\n")

print("Detected texts saved to:", output_file_path)
