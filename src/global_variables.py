import spacy
import openpyxl as xl
import requests

import os
import sqlite3
from datetime import datetime
import builtins
import re
import json
original_print = builtins.print

def print_with_time(*args, **kwargs):
    original_print(datetime.now().strftime('%d.%m.%Y %H:%M:%S'), *args, **kwargs)
builtins.print = print_with_time

PROJECT_DIR = os.path.dirname(__file__)
if "storage" not in os.listdir(PROJECT_DIR):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
# Database
conn = sqlite3.connect(os.path.join(PROJECT_DIR, "storage", "file_bank.db"))
cur = conn.cursor()
ALL_TAGS = set(open(os.path.join(PROJECT_DIR, "storage", "tags.csv"), encoding='utf-8-sig').read().splitlines())

# Excel
TARGET_COLUMNS = ["Tag Number", "Property Name", "Property Name Rus", "Property Value", "Property Description Rus", "UoM"]
TAG_COL, NAME_COL, NAME_RU_COL, VALUE_COL, DESC_RU_COL, UM_COL = TARGET_COLUMNS
MODIFIED = False
xls_file: xl.Workbook = None

import atexit
def exit_handler():
    global MODIFIED, conn, ALL_TAGS
    if MODIFIED and xls_file is not None:
        xls_file.save(f"../xls/Test_output.xlsx")
    conn.commit()
    conn.close()
    with open(os.path.join(PROJECT_DIR, "storage", "tags.csv"), "w") as f:
        f.write("\n".join(ALL_TAGS))
atexit.register(exit_handler)

# General
LOGS = True
if PROJECT_DIR.endswith("src"):
    PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TIKA_ADDRESS = os.environ.get('TIKA_ADDRESS')
LLM_ADDRESS, LLM_PORT = os.environ.get('LLM_ADDRESS'), os.environ.get('LLM_PORT')
# LLM_URL = f"{LLM_ADDRESS}:{LLM_PORT}/llm/request/"
LLM_URL = "http://localhost:11434/v1/chat/completions"
MAX_RETRIES, LLM_FAILURE_FLAG = 1, "NOANS"
CONTEXT_OVERLAP = 500
CURRENT_TABLE_REPR = "html"
CHUNK_SIZE = 100_000

# Text parsing
nlp_ru = None
nlp_en = None
ru_lemmatizer = None
TAG_REGULAR_EXP = r"(?:E-)?((?:[A-Z0-9]+-)+(?:[A-Z0-9]+))"
RU_LETTERS = [chr(i) for i in range(1072, 1104)] + ["ё"]
EOF_FLAG, EOP_FLAG = "(!@#$)", "[!@#$%]"
header_pattern = re.compile(r"(\d+\.*)+\s+([A-ZА-Я]{2,})")
EXCLUDE_POSES = ["ADP", "CCONJ", "DET", "INTJ", "NUM", "PART", "PRON", "PUNCT", "SCONJ", "SYM"]
WINDOW_SIZE_MULTIPLIER = 4

# Profiling
REQUESTS_TIME_SPENT = 0
METHODS_TIME_SPENT = 0

def validate_result(result, possible_values: list[str] = None) -> bool:
    global LLM_FAILURE_FLAG
    if isinstance(result, str):
        if result not in ["N/A", "none", "no", "нет", "n/a", "{}", "json", "JSON", "JSON:",
                          f'{{"value": {LLM_FAILURE_FLAG}, "uom": {LLM_FAILURE_FLAG}}}', None, LLM_FAILURE_FLAG]:
            if not (result.startswith(LLM_FAILURE_FLAG) or result.endswith(LLM_FAILURE_FLAG) or result.startswith("```")):
                if possible_values:
                    if result not in possible_values:
                        return False
                return True
    return False

def smart_llm_request(system_messages, user_messages):
    global LLM_FAILURE_FLAG, CONTEXT_OVERLAP
    if isinstance(user_messages, list):
        user_messages = "\n".join(user_messages)
    if len(user_messages) < 10_000:
        return direct_llm_request(system_messages, user_messages)
    # Too big context (almost always means mistake)
    if len(user_messages) > 100_000:
        return {"value": LLM_FAILURE_FLAG, "uom": LLM_FAILURE_FLAG}
    batches = [user_messages[i-CONTEXT_OVERLAP:i + 10_000 + CONTEXT_OVERLAP] for i in range(CONTEXT_OVERLAP, len(user_messages), 10_000)]
    for batch in batches:
        some_ans = direct_llm_request(system_messages, batch)
        if some_ans != LLM_FAILURE_FLAG:
            return some_ans
    return {"value": LLM_FAILURE_FLAG, "uom": LLM_FAILURE_FLAG}

def direct_llm_request(system_messages, user_messages=None):
    global REQUESTS_TIME_SPENT, LLM_FAILURE_FLAG, LLM_URL
    if not isinstance(system_messages, list):
        system_messages = [system_messages]
    if not isinstance(user_messages, list):
        user_messages = [user_messages]
    essential_context = """"Ты - ИИ-помощник в извлечении значения свойств из текста. Ты работаешь с инженерными документами и паспортами объекта."""
    essential_context = re.sub(r"\s+", " ", essential_context)
    messages = [{"role": "system", "content": essential_context}]
    for msg in system_messages:
        messages.append({"role": "system", "content": msg})
    for msg in user_messages:
        messages.append({"role": "user", "content": msg})
    if not messages:
        return {"value": LLM_FAILURE_FLAG, "uom": LLM_FAILURE_FLAG}
    data = {
        # "model": "qwen2.5-coder:7b",
        "model": "gemma3:12b",
        "messages": messages,
        "temperature": 0.0,
        "format": "json",
        "stream": False,
        "options": {
            # "num_ctx": 16384,
            "seed": 420
        }
    }
    start = datetime.now()
    try:
        response = requests.post(LLM_URL, json=data).json()
    except requests.exceptions.ConnectionError:
        print("\033[33m{}\033[0m".format(f"ERROR! Failed to connect to LLM service"))
        exit(-1)
    if LOGS:
        with open("logs.txt", 'a') as f:
            f.write(json.dumps(messages, ensure_ascii=False, indent=4) + "\n")
            f.write(json.dumps(response, ensure_ascii=False, indent=4) + "\n")
    REQUESTS_TIME_SPENT += (datetime.now() - start).total_seconds()
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = content.strip()
        if '```' in content:
            content = re.search(r"```(?:json)*(.+\s*)+```", content).group(1)
        content = content.replace('\n', '')
        try:
            return json.loads(content)
        except json.decoder.JSONDecodeError:
            return {"value": LLM_FAILURE_FLAG, "uom": LLM_FAILURE_FLAG}
    else:
        print("\033[33m{}\033[0m".format(f"ERROR! Got wrong data format from LLM: {type(content)}"))
        exit(-1)


def has_russian_letters(string: str) -> bool:
    global RU_LETTERS
    if not isinstance(string, str):
        return False
    string = string.lower()
    for let in RU_LETTERS:
        if let in string:
            return True
    return False
def has_letters(string: str) -> bool:
    if not isinstance(string, str):
        return False
    string = string.lower()
    global RU_LETTERS
    for let in RU_LETTERS:
        if let in string:
            return True
    for i in range(ord('a'), ord('z') + 1):
        if chr(i) in string:
            return True
    return False
def init_spacy():
    global nlp_ru, nlp_en, ru_lemmatizer
    nlp_ru = spacy.load("ru_core_news_lg")
    nlp_en = spacy.load("en_core_web_md")
    ru_lemmatizer = spacy.load("ru_core_news_lg", disable=["ner", "parser", "attribute_ruler"])
