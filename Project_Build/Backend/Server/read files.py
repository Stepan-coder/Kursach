import docx2txt
import pandas as pd
from pdf2image import convert_from_path


def read_docs_file(path: str) -> str:
    return docx2txt.process(path)


def convert_pdf_to_image(path: str):
    images = convert_from_path(path, 500)
    print(images)
    
    
def read_from_csv(filename: str, delimiter: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    This method reads the dataset from a .csv file
    :param filename: The name of the .csv file
    :param delimiter: Symbol-split in a .csv file
    :param encoding: Explicit indication of the .csv file encoding
    :return: The dataframe read from the file
    """
    return pd.read_csv(filename, encoding=encoding, delimiter=delimiter)


def read_from_xlsx(filename: str, sheet_name: str) -> pd.DataFrame:
    """
    This method reads the dataset from a .csv file
    :param filename: The name of the .csv file
    :return: The dataframe read from the file
    """
    try:
        return pd.read_excel(filename, sheet_name=sheet_name, engine="xlrd")
    except:
        pass
    try:
        return pd.read_excel(filename, sheet_name=sheet_name, engine="openpyxl")
    except:
        pass
    try:
        return pd.read_excel(filename, sheet_name=sheet_name, engine="odf")
    except:
        pass
    try:
        return pd.read_excel(filename, sheet_name=sheet_name, engine="pyxlsb")
    except:
        pass