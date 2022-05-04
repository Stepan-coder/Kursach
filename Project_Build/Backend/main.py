import docx2txt
import pytesseract
from NER import *
from OCR import *
from PIL import Image
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

"""
TODO
https://github.com/UB-Mannheim/tesseract/wiki
https://docs.wand-py.org/en/latest/guide/install.html#install-imagemagick-on-windows
"""

class Models:
    def __init__(self, model_path: str, is_bert: bool, download: bool = False):
        self.__rus_text_recognition = RussianHandwritingRecognition(model_path=model_path)
        self.__ner_text_marckup = TextMarkUp(is_bert=is_bert, download=False)


models = Models(model_path="checkpoint_epoch_8.pt", is_bert=False)
print(models.__rus_text_recognition.predict(images={"со слов": Image.open('test12.png')}))


# from PIL import Image
# im = Image.open(r'Снимок.PNG')
# print(im)
# print(image_to_string(im, lang='rus+en'))