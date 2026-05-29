# Built-in modules
import os
import re
from contextlib import contextmanager
from typing import *
import difflib

# Project modules
import src.global_variables as gv

TESSERACT_DIR = os.path.join(gv.PROJECT_DIR, "Tesseract-OCR")
os.environ["PATH"] = str(TESSERACT_DIR) + os.pathsep + os.environ["PATH"]
os.environ["TESSDATA_PREFIX"] = str(os.path.join(TESSERACT_DIR, "tessdata"))
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

# External modules
from PIL import Image as PIL_Image
import tesserocr
import cv2
import numpy as np
import polars as pl
from bs4 import BeautifulSoup
from img2table.ocr.data import OCRDataframe
from img2table.ocr.base import OCRInstance

TARGET_DPI = 288.0
"""
+-----+-------+-------------------+
| DPI | Scale | A4 page size (px) |
+-----+-------+-------------------+
|  72 |  1x   |    595  x  842    |
| 144 |  2x   |   1190  x 1684    |
| 216 |  3x   |   1785  x 2526    |
| 288 |  4x   |   2380  x 3368    |
| 360 |  5x   |   2975  x 4210    |
| 432 |  6x   |   3570  x 5052    |
| 504 |  7x   |   4165  x 5894    |
| 576 |  8x   |   4760  x 6736    |
| 648 |  9x   |   5355  x 7578    |
| 720 |  10x  |   5950  x 8420    |
+-----+-------+-------------------+
"""

OCR_CORRECTIONS = str.maketrans({
    '$': 'S', ']': 'J', '[': 'J', '|': 'I',
    '{': 'C', '}': 'C', '(': 'C', ')': 'J', '@': '0',
    'А': 'A', 'а': 'A', 'В': 'B', 'в': 'B',
    'С': 'C', 'с': 'C', 'Е': 'E', 'е': 'E',
    'Н': 'H', 'н': 'H', 'К': 'K', 'к': 'K',
    'М': 'M', 'м': 'M', 'О': 'O', 'о': 'O',
    'Р': 'P', 'р': 'P', 'Т': 'T', 'т': 'T',
    'Х': 'X', 'х': 'X', 'У': 'Y', 'у': 'Y',
    'v': 'V', 'h': 'H', 's': 'S', 'j': 'J', 'b': 'B',
    'p': 'P', 'g': 'G', 'c': 'C', 'f': 'F', 'w': 'W', 'l': 'L'
})
TAG_PATTERN = re.compile(r'\b[A-Za-zА-Яа-я0-9$\]\[|@()]{2,8}(?:\s*-\s*[A-Za-zА-Яа-я0-9$\]\[|@()]{1,8}){2,7}\b')

def fix_tags(text: str) -> str:
    def replacer(match):
        raw_candidate = match.group(0)
        cleaned_candidate = raw_candidate.replace(" ", "").translate(OCR_CORRECTIONS).upper()
        if cleaned_candidate in gv.ALL_TAGS:
            return cleaned_candidate
        closest_matches = difflib.get_close_matches(cleaned_candidate, list(gv.ALL_TAGS), n=1, cutoff=0.7)
        if len(closest_matches) == 1:
            return closest_matches[0]
        return cleaned_candidate
    return TAG_PATTERN.sub(replacer, text)


class OptimizedTesseractOCR(OCRInstance):
    def __init__(self, n_threads: int = 1, lang: str = 'eng', psm: int = 3):
        global TARGET_DPI
        self.lang = lang
        self.psm = psm
        self.api = tesserocr.PyTessBaseAPI()
        self.api.SetVariable("user_defined_dpi", f"{int(TARGET_DPI)}")
        self.api.SetVariable("load_system_dawg", "0")
        self.api.SetVariable("load_freq_dawg", "0")
        self.api.Init(lang=lang, psm=psm, oem=tesserocr.OEM.LSTM_ONLY)
        # self.api.SetVariable("hocr_font_info", "1")
        self.n_threads = int(n_threads)
        self.cached_df: pl.DataFrame = None
        self.hocr_content = None

    def get_remaining_text(self, exclude_bboxes: list[tuple[int, int, int, int]]) -> str:
        if self.cached_df.is_empty():
            return ""
        df = self._exclude_bboxes_from_df(self.cached_df, exclude_bboxes)
        filtered = df.filter(pl.col("value").is_not_null())
        return " ".join(filtered["value"].to_list())

    def process_page_once(self, image: Union[np.ndarray, PIL_Image.Image]):
        if not isinstance(image, PIL_Image.Image):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                pil_img = PIL_Image.fromarray(gray)
            else:
                pil_img = PIL_Image.fromarray(image)
        else:
            pil_img = image.copy()
        with suppress_stdout_stderr():
            self.api.SetImage(pil_img)
            self.api.SetPageSegMode(self.psm)
            # self.api.Recognize()
            self.hocr_content = self.api.GetHOCRText(0)
            ocr_df = self.to_ocr_dataframe([self.hocr_content])
            if ocr_df and not ocr_df.df.is_empty():
                self.cached_df = ocr_df.df
            else:
                self.cached_df = pl.DataFrame(schema=self.pl_schema)
        return pil_img

    def get_text_in_rect(self, x1: int, y1: int, x2: int, y2: int) -> str:
        if self.cached_df.is_empty():
            return ""
        filtered = self.cached_df.filter(
            (pl.col("x1") >= x1) & (pl.col("y1") >= y1) &
            (pl.col("x2") <= x2) & (pl.col("y2") <= y2) &
            pl.col("value").is_not_null()
        )
        return " ".join(filtered["value"].to_list())

    def to_ocr_dataframe(self, content: list[str]) -> OCRDataframe:
        list_dfs = []
        for page, hocr in enumerate(content):
            soup = BeautifulSoup(hocr, features='xml')
            list_elements = []
            for element in soup.find_all(attrs={"class": "ocrx_word"}):
                title_str = element.get("title", "")
                d_el = {
                    "page": page,
                    "class": "ocrx_word",
                    "id": element.get("id", ""),
                    "parent": element.parent.get('id') if element.parent else "",
                    "value": re.sub(r"^(\s|\||L|_|;|\*)*$", '', element.get_text()).strip(),
                }
                str_conf = re.search(r"x_wconf (\d+)", title_str)
                d_el["confidence"] = int(str_conf.group(1)) if str_conf else None
                bbox_match = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title_str)
                if bbox_match:
                    d_el["x1"], d_el["y1"], d_el["x2"], d_el["y2"] = map(int, bbox_match.groups())
                else:
                    d_el["x1"] = d_el["y1"] = d_el["x2"] = d_el["y2"] = 0
                list_elements.append(d_el)

            if list_elements:
                list_dfs.append(pl.DataFrame(data=list_elements, schema=self.pl_schema))
        return OCRDataframe(df=pl.concat(list_dfs)) if list_dfs else None

    @staticmethod
    def _exclude_bboxes_from_df(df: pl.DataFrame,
                                exclude_bboxes: list[tuple[int, int, int, int]]) -> pl.DataFrame:
        for (bx1, by1, bx2, by2) in exclude_bboxes:
            df = df.filter(
                ~((((pl.col("x1") + pl.col("x2")) / 2) >= bx1) &
                  (((pl.col("y1") + pl.col("y2")) / 2) >= by1) &
                  (((pl.col("x1") + pl.col("x2")) / 2) <= bx2) &
                  (((pl.col("y1") + pl.col("y2")) / 2) <= by2))
            )
        return df

    def of(self, document) -> OCRDataframe:
        return OCRDataframe(df=self.cached_df)

    def hocr(self, image: np.ndarray) -> str:
        return self.hocr_content
