import os
import cv2
import docx2txt
import pytesseract
import numpy as np
import pytesseract
from NER import *
from OCR import *
from SIG import *
from PIL import Image
from typing import Dict
from pdf2image import convert_from_path
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class Models:
    def __init__(self, hand_model_path: str, signature_model_path: str, is_bert: bool, download: bool = False):
        self.__rus_text_recognition = RussianHandwritingRecognition(path_to_model=hand_model_path)
        self.__YOLO_signature_detector = YoloSignatureDetector(path_to_model=signature_model_path)
        self.__ner_text_markup = TextMarkUp(is_bert=is_bert, download=False)

    def get_NER_markup(self, text: str) -> List[MarkUp]:
        return self.__ner_text_markup.encode(text_markup=self.__ner_text_markup.get_markup(text=text))

    def handwritten_OCR(self, image: np.ndarray or Image.Image) -> str:
        print("Getting text with ResNet model...")
        sequence = self.__detect_handwritten_words_sequence(image=image)
        expression = self.__crop_texts(image=image, sequence=sequence)
        prediction = self.__rus_text_recognition.predict(images=expression)
        words = dict(sorted(prediction.items(), key=lambda x: int(x[0]))).values()
        print("ResNet model done!")
        return " ".join(words).replace("  ", " ")

    def tesseract_OCR(self, image: np.ndarray or Image.Image) -> str:
        print("Getting text with TesseractOCR...")
        text = image_to_string(image, lang='rus+en')
        print("TesseractOCR done!")
        return image_to_string(image, lang='rus+en')

    def signature_OCR(self, image: np.ndarray or Image.Image):
        print("Getting signatures with YOLOv5x...")
        signatures = self.__YOLO_signature_detector.predict(images=[image])
        print("YOLOv5x done!")
        return signatures

    @staticmethod
    def __detect_handwritten_words_sequence(image: np.ndarray or Image.Image) -> Dict[str, tuple]:
        if isinstance(image, Image.Image):
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        if isinstance(image, np.ndarray):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        coords, xes, yes, sequence_points = [], [], [], []
        sequence = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            coords.append((x, y, x + w, y + h, int(x + w / 2.0), int(y + h / 2.0)))
            xes.append(int(x + w / 2.0))
            yes.append(int(y + h / 2.0))
        xes.sort()
        yes.sort()
        counter = 0
        for h in yes:
            for w in xes:
                for coord in coords:
                    if coord[0] <= w <= coord[2] and coord[1] <= h <= coord[3] and (coord[4], coord[5]) \
                            not in sequence_points:
                        sequence[str(counter)] = coord
                        sequence_points.append((coord[4], coord[5]))
                        counter += 1
        return sequence

    @staticmethod
    def __crop_texts(image: np.ndarray or Image.Image, sequence: Dict[str, tuple]) -> Dict[str, Image.Image]:
        expression = {}
        for expr in sequence:
            x, y, x1, y1, cx, cy = sequence[expr]
            if isinstance(image, np.ndarray):
                expression[expr] = Image.fromarray(image[y:y1, x:x1])
            if isinstance(image, Image.Image):
                expression[expr] = image.crop((x, y, x1, y1))
        return expression