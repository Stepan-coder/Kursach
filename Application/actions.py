import os
import cv2
import pytesseract
import pandas as pd
from load_models import *
from typing import List
from pdf2image import convert_from_path
from pytesseract import image_to_string
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def clean_string(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("  ", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def read_docs_file(path: str) -> str:
    return clean_string(text=docx2txt.process(path))


def read_from_xlsx(filename: str, sheet_name: str) -> pd.DataFrame:
    """
    This method reads the dataset from a .csv file
    :param filename: The name of the .csv file
    :return: The dataframe read from the file
    """
    for engine in ["xlrd", "openpyxl", "odf", "pyxlsb"]:
        try:
            return pd.read_excel(filename, sheet_name=sheet_name, engine=engine)
        except:
            pass


def convert_PDF_to_image(path: str) -> List[str]:
    pages = convert_from_path(path,  500, poppler_path=os.path.join(os.getcwd(), "Library", "bin"))
    name = ".".join(os.path.basename(path).split(".")[:-1])
    dir = os.path.dirname(path)
    if os.path.exists(os.path.join(dir, name)):
        for file in os.listdir(os.path.join(dir, name)):
            os.remove(os.path.join(dir, name, file))
        os.rmdir(os.path.join(dir, name))
    os.mkdir(os.path.join(dir, name))
    images_path = []
    counter = 0
    for page in pages:
        page.save(os.path.join(dir, name, f"{name}_{counter}.png"), 'JPEG')
        images_path.append(os.path.join(dir, name, f"{name}_{counter}.png"))
        counter += 1
    return images_path


def convert_markup_to_json(markup: List[MarkUp]) -> Dict[int, Any]:
    result = {}
    counter = 0
    for mp in markup:
        if not mp.is_empty():
            result[counter] = {"type": mp.get_type().name, "text": mp.get_text()}
            counter += 1
    return result


def build_str_markup(json_markup: Dict[int, Any], border: str = "") -> str:
    output = ""
    for index in list(json_markup.keys()):
        output += f"{border}{json_markup[index]['type']}: {json_markup[index]['text']}\n"
    return output


def markup_docx(path: str, models: Models) -> Dict[int, Any]:
    original_text = read_docs_file(path=path)
    markup = models.get_NER_markup(text=original_text)
    return convert_markup_to_json(markup=markup)


def markup_image(path: str, models: Models) -> Dict[int, Any]:
    image = Image.open(path)
    tesseract_text = models.tesseract_OCR(image=image.copy())
    handwritten_text = models.handwritten_OCR(image=image.copy())
    signatures = len(models.signature_OCR(image=image.copy())[0])
    markup = models.get_NER_markup(text=f"{tesseract_text} {handwritten_text}")
    json_markup = convert_markup_to_json(markup=markup)
    json_markup[len(json_markup) + 1] = {'type': f'SIGNATURES', 'text': f'{signatures}'}
    return json_markup


def markup_PDF(path: str, models: Models) -> Dict[int, Any]:
    images = convert_PDF_to_image(path=path)
    dir = os.path.dirname(path)
    page_counter = 1
    counter = 0
    markup = {}
    for image in images:
        markup[counter] = {'type': f'PAGE', 'text': f'{page_counter}'}
        counter += 1
        page_counter += 1
        image_markup = markup_image(path=os.path.join(dir, image), models=models)
        for imm in image_markup:
            markup[counter] = image_markup[imm]
            counter += 1
    return markup



