import os

import cv2
import pytesseract

# Path to Tesseract executable (change this based on your system)
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def get_ROI(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and find the largest contour
    contour_areas = [cv2.contourArea(c) for c in contours]
    max_contour_index = contour_areas.index(max(contour_areas))
    largest_contour = contours[max_contour_index]

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Extract the region of interest (ROI) containing the license plate
    license_plate_roi = image[y:y + h, x:x + w]

    return license_plate_roi

def extract_license_plate(image):

    # Preprocess the license plate image
    roi_image = get_ROI(image)

    cv2.imshow('ROI', roi_image)

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    kernel_size = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilate_image = cv2.dilate(gray, kernel, iterations=1)



    cv2.imshow('ROI - roi_image', dilate_image)

    # Use Pytesseract to extract text from the license plate image
    license_plate_text = pytesseract.image_to_string(dilate_image, config='--psm 8')


    return license_plate_text.strip()

def main():
    # Path to the directory containing the images
    directory = 'images/SmallAmountImg/'

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            # Extract license plate text
            license_plate_text = extract_license_plate(image)

            # Print the filename and extracted text
            print("Image:", filename)
            print("Extracted License Plate Text:", license_plate_text)
            print()

            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

    cv2.waitKey(0)
