import docx2txt
import pytesseract
from PIL import Image
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = 'D:/Program Files/tesseract/tesseract.exe'

"""
TODO
https://github.com/UB-Mannheim/tesseract/wiki
"""

from PIL import Image
im = Image.open(r'C:\Users\<user>\Downloads\dashboard-test.jpeg')
print(im)
print(image_to_string(im))





