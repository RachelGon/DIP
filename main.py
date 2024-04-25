import os
import imutils

import cv2
import numpy as np
import pytesseract

# Path to Tesseract executable (change this based on your system)
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")


def get_ROI(image):
    # Load the pre-trained car detection model

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect plates in the image
    plates = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over detected cars
    for (x, y, w, h) in plates:
        # Define ROI for license plate
        roi = image[y:y + h, x:x + w]

        # Perform OCR on the license plate ROI
        license_plate_text = pytesseract.image_to_string(roi, config='--psm 8')

        # If license plate text is detected, return the license plate ROI
        if license_plate_text:
            return roi
    else:
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
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

    
    cv2.imshow('ROI', roi_image)

    # # gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    # # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)

    # thresh = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # eroded = cv2.erode(thresh, None, iterations=1)
    # dilated = cv2.dilate(eroded, None, iterations=1)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = cv2.adaptiveThreshold(blur, 252, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 46)
    thr2 = cv2.adaptiveThreshold(blur, 252, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 66)
    bnt = cv2.bitwise_not(thr)
    bnt2 = cv2.bitwise_not(thr2)

    bigkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    smallkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    eroded = cv2.erode(bnt, smallkernel, iterations=1)
    dilate = cv2.dilate(eroded, smallkernel, iterations=2)
    erodeded = cv2.erode(dilate, smallkernel, iterations=1)


    # dilate = cv2.dilate(eroded, dilatekernel, iterations=2)
    # dilate = cv2.dilate(dilate, None, iterations=1)
    

    cv2.imshow('1ROI - bnt {image}', bnt)
    cv2.imshow('1.1ROI - bnt {image}', bnt2)
    cv2.imshow('2ROI - eroded {image}', eroded)
    cv2.imshow('3ROI - dilate {image}', dilate)
    cv2.imshow('4ROI - erodeded {image}', erodeded)
    license_plate_text = pytesseract.image_to_string(bnt2, lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


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
