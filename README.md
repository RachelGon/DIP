# License Plate Recognition System

This repository contains a Python application designed to recognize license plates from images using two different Optical Character Recognition (OCR) technologies: EasyOCR and Tesseract OCR. Users can choose the preferred OCR engine at runtime, making this a versatile tool for license plate detection in various environments.

## Features

- **Dual OCR Support**: Utilizes EasyOCR and Tesseract OCR for license plate recognition.
- **Region of Interest (ROI) Detection**: Automatically detects the area of the license plate within the image.
- **Image Enhancement**: Applies multiple preprocessing techniques to improve OCR accuracy.
- **Flexible OCR Selection**: Users can select their preferred OCR engine via command line input.

## Prerequisites

Before you can run this project, you need to install the following software:

- Python 3.6 or higher
- OpenCV
- Pytesseract
- EasyOCR
- NumPy

You also need to ensure that Tesseract-OCR is correctly installed on your system. For Windows, you can install it from [here](https://github.com/UB-Mannheim/tesseract/wiki) and make sure the path to the executable is correctly set in the script:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
