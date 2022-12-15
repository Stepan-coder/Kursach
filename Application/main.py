#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import wx
import eel
import cv2
import json
import pytesseract
import pandas as pd
from database import *
from typing import List
from load_models import *
from pdf2image import convert_from_path
from pytesseract import image_to_string
from jinja2 import Environment, FileSystemLoader
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


env = Environment(loader=FileSystemLoader('web/templates'))
eel.init("web")
DocumentPreview = env.get_template('DocumentPreview.html')
FilePreview = env.get_template('/pattern/FilePreview.html')
LeftBarFile = env.get_template('/pattern/LeftBarFile.html')
FileData = env.get_template('/pattern/FileData.html')
count = 0


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


@eel.expose
def login(email, password):
    if db.get_table("users").get_from_cell(key=str(0), column_name="email") != email and \
       password != db.get_table("users").get_from_cell(key=str(0), column_name="password"):
        quit()
    files = []
    data = []
    js = json.loads(db.get_table("users").get_from_cell(key=str(0), column_name="files"))
    for i in range(len(js)):
        files.append({'type': js[str(i)]['type'], 'name': js[str(i)]['name']})
        data.append(js[str(i)]['data'])
    # print(files)
    # print(data)
    global count
    count = len(files)
    # files = [{'type': 'docx', 'name': 'efef.ef'}]
    # data = ["""123"""]

    return DocumentPreview.render(files=files, data=data)


@eel.expose
def pythonFunction(wildcard="*"):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()

    print(path)
    # Тут происходит твоя магия
    file_type = os.path.basename(path).split(".")[-1]
    basename = os.path.basename(path)
    if file_type == 'docx':
        file_type = 'pdf'
        data = build_str_markup(json_markup=markup_docx(path=path, models=models))
    elif file_type == 'png' or file_type == "jpg":
        file_type = 'image'
        data = build_str_markup(json_markup=markup_image(path=path, models=models))
    elif file_type == 'pdf':
        data = build_str_markup(json_markup=markup_PDF(path=path, models=models))
    global count
    print(data)
    files = json.loads(db.get_table("users").get_from_cell(key=str(0), column_name="files"))
    files[str(count)] = {"type": file_type, "name": basename, "data": data}
    db.get_table("users").set_to_cell(key=str(0), column_name="files", new_value=json.dumps(files))
    count += 1

    return {'preview': FilePreview.render(file={'type': file_type, 'name': basename, 'len': count}),
            'file': {'type': file_type, 'html': LeftBarFile.render(file={'type': file_type, 'name': basename})},
            'get_data': FileData.render(file={'data': data, 'len': count})}


db = DataBase(path=os.path.join(os.getcwd(), "database"), filename="DigiDoc.db")
db.create_table(name="users",
                labels={"id": DBType.TEXT,
                        "email": DBType.TEXT,
                        "password": DBType.TEXT,
                        "files": DBType.TEXT},
                primary_key="id")
models = Models(hand_model_path="models/checkpoint_epoch_8.pt",
                signature_model_path="models/model.pt",
                is_bert=True)
eel.start('templates/index.html', jinja_templates='templates', port=8000)

