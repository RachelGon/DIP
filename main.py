import os
import cv2
import pytesseract
import numpy as np
import easyocr

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load pre-trained data on license plates (Haar Cascade)
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Function to get the Region of Interest (ROI) where the license plate is located
def get_ROI(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    plates = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  # Detect license plates

    for (x, y, w, h) in plates:
        return image[y:y + h, x:x + w]  # Return the area of the first detected plate

    # Fallback method using contours if no plates are detected initially
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y + h, x:x + w]  # Return the area of the largest contour found

# Function to enhance the quality of the image for better OCR results
def enhance_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cv2.imshow('Grayscale Image', gray)  # Show the grayscale image

    clahe = cv2.createCLAHE(clipLimit=2.9, tileGridSize=(11, 11))  # Apply CLAHE for contrast enhancement
    contrast_enhanced = clahe.apply(gray)
    cv2.imshow('CLAHE Enhanced Image', contrast_enhanced)  # Show the enhanced image

    gaussian_blur = cv2.GaussianBlur(contrast_enhanced, (3, 3), 0)  # Apply Gaussian Blur
    cv2.imshow('Gaussian Blur Image', gaussian_blur)  # Show the blurred image

    unsharp_image = cv2.addWeighted(contrast_enhanced, 1.2, gaussian_blur, -0.2, 0)  # Apply Unsharp Mask
    cv2.imshow('Unsharp Mask Image', unsharp_image)  # Show the unsharp mask image

    adaptive_thresh = cv2.adaptiveThreshold(unsharp_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('Adaptive Threshold Image', adaptive_thresh)  # Show the final preprocessed image

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return adaptive_thresh  # Return the processed image for OCR

# Function to extract the license plate using selected OCR engine
def extract_license_plate(image, use_easyocr):
    roi_image = get_ROI(image)  # Get the region of interest (ROI)
    if roi_image is None:
        print("No ROI found.")
        return ""

    final_image = enhance_image_quality(roi_image)  # Enhance image quality

    if use_easyocr:
        reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
        result = reader.readtext(final_image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')  # OCR with EasyOCR
        license_plate_text = " ".join([res[1] for res in result])  # Extract text from results
    else:
        config = '--psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'  # Tesseract config
        license_plate_text = pytesseract.image_to_string(final_image, config=config)  # OCR with Tesseract

    return license_plate_text.strip()  # Return the cleaned text

# Main function to iterate through images and apply OCR
def main():
    print("Select OCR Engine:")
    print("1: EasyOCR")
    print("2: Tesseract")
    choice = input("Enter choice (1 or 2): ")  # User selects OCR engine
    use_easyocr = (choice == '1')  # Boolean to determine which OCR engine to use

    directory = 'images/SmallAmountImg/'  # Directory containing images
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Process only image files
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)  # Load the image
            license_plate_text = extract_license_plate(image, use_easyocr)  # Extract text from the image
            print(f"Image: {filename}")
            print(f"Extracted License Plate Text: {license_plate_text}\n")  # Print results

            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
