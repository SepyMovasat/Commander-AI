"""
Screen module: Capture screen, OCR, and visual context.
Requires: Tesseract OCR (install with `sudo apt install tesseract-ocr`)
"""
import pytesseract
from PIL import ImageGrab

def capture_screen():
    # Capture the screen (cross-platform)
    img = ImageGrab.grab()
    return img

def ocr_screen(img=None):
    if img is None:
        img = capture_screen()
    text = pytesseract.image_to_string(img)
    return text
