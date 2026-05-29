# Built-in modules
import re
from typing import *
import json
from functools import lru_cache
from collections import deque
import itertools

# External modules
import spacy.tokens
from img2table.tables.objects.extraction import ExtractedTable
import numpy as np
import pandas as pd
import pickle

# Project modules
from src import global_variables as gv


def validate_token(token: spacy.tokens.Token) -> bool:
    return token.is_alpha and not token.is_stop and not token.is_punct and not token.is_space

def extracted_table_repr(table: ExtractedTable):
    if isinstance(table, ExtractedTable):
        df = table.df
    else:
        df = table
    df = df.drop_duplicates()
    df = df.fillna('-')
    df = df.replace(r'\n', ' ', regex=True)
    match gv.CURRENT_TABLE_REPR:
        case "markdown":
            return df.to_markdown(index=False)
        case "html":
            return df.to_html(index=False)
        case "json":
            return df.to_json(orient='records')
        case _:
            return df.to_html(index=False)


def get_frames(source: str | List[ExtractedTable], points: List[int] | Tuple[int, int, int, str]):
    """
    :param source:
    :param points:
    :return:
    """
    result = "Отрывки:\n" if isinstance(source, str) else "Таблицы:\n"
    for point in points:
        if isinstance(source, str):
            left = max(source[:point].rfind(gv.EOF_FLAG), source[:point].rfind(gv.EOP_FLAG), 0)
            right = min(source[point:].find(gv.EOF_FLAG), source[point:].find(gv.EOP_FLAG))
            right = max(right, 0)
            result += source[left:right + 1] + '\n'
        else:
            # Cross (row + column)
            df = source[point[0]].df
            cross = pd.concat([df.loc[point[1]], df[point[2]]], axis=1)
            result += extracted_table_repr(df) + gv.EOF_FLAG + '\n'
            # Only column
            # df: pd.DataFrame = source[point[0]].df
            # result.append(df.loc[[0, point[1]]].to_html())
    return result


def search_in_tables(tables: List[ExtractedTable], prop_tokens: List[spacy.tokens.Token]) -> List[Tuple[int, Any, Any]]:
    """
    Прогоняет таблицы как текст через поиск и возвращает уникальные координаты.
    Возвращает список: [(table_idx, row_label, col_label), ...]
    """
    def _build_mapped_table_string(some_df: pd.DataFrame) -> Tuple[str, List[Tuple[Any, Any]]]:
        """
        Создает линейное строковое представление таблицы и карту соответствия:
        индекс_символа_в_строке -> (row_label, col_label).
        """
        text_builder, ch_map, default_row = [], [], some_df.index[0] if not some_df.empty else 0
        for col_label in some_df.columns:
            text_builder.append(f"{col_label} ")
            ch_map.extend([(default_row, col_label)] * len(f"{col_label} "))
        text_builder, ch_map = text_builder + ["\n"], ch_map + [None]
        for row_label in some_df.index:
            for col_label in some_df.columns:
                cell_val = "" if str(some_df.loc[row_label, col_label]) in ("nan", "None", "-") else str(
                    some_df.loc[row_label, col_label])
                text_builder.append(cell_val + ' ')
                ch_map.extend([(row_label, col_label)] * len(cell_val + ' '))
            text_builder.append("\n")
            ch_map.append(None)
        return "".join(text_builder), ch_map

    found_coordinates = []
    if not prop_tokens or not tables:
        return found_coordinates
    if isinstance(prop_tokens, str):
        prop_tokens = [t for t in gv.nlp_ru(prop_tokens)]
    for table_idx, table in enumerate(tables):
        df = table.df if hasattr(table, 'df') else table
        if df.empty:
            continue

        linear_text, char_map = _build_mapped_table_string(df)

        best_tokens, count, match_indices = get_most_specific_tokens(linear_text, prop_tokens)
        if not match_indices:
            continue
        for char_idx in match_indices:
            if 0 <= char_idx < len(char_map):
                cell_coord = char_map[char_idx]
                if cell_coord is None and char_idx > 0:
                    cell_coord = char_map[char_idx - 1]
                if cell_coord is not None:
                    r_label, c_label = cell_coord
                    coord_tuple = (table_idx, r_label, c_label)
                    if coord_tuple not in found_coordinates:
                        found_coordinates.append(coord_tuple)
    return found_coordinates


@lru_cache(maxsize=5)
def lemmatize_text(text: str) -> List[Tuple[str, int]]:

    def lemmatize(some_text: str, offset: int = 0) -> List[Tuple[str, int]]:
        res = []
        some_doc = gv.ru_lemmatizer(some_text)
        for token in some_doc:
            if validate_token(token):
                global_idx = token.idx + offset
                res.append((token.lemma_.lower(), global_idx))
        return res

    if len(text) < gv.CHUNK_SIZE:
        return lemmatize(text)
    lemmatized_data, i = [], 0
    while i < len(text):
        if text[i] in ("\n", " "):
            next_symb = re.search(r"\S", text[i:])
            if not next_symb:
                break
            i += next_symb.start()
        chunk_end = text.rfind("\n", i, i + gv.CHUNK_SIZE)
        if chunk_end == -1:
            chunk_end = text.rfind(' ', i, i + gv.CHUNK_SIZE)
            if chunk_end == -1:
                chunk_end = i + gv.CHUNK_SIZE
        chunk = text[i: chunk_end]
        lemmatized_data.extend(lemmatize(chunk, offset=i))
        i = chunk_end
    return lemmatized_data

def count_groups(sets: List[Set[int]], width: int) -> List[int]:
    """
    Ищет группы слов в пределах окна `width`.
    Возвращает список индексов (середины найденных окон).
    """
    if not sets:
        return []
    n, items = len(sets), [(v, sid) for sid, s in enumerate(sets) for v in s]
    items.sort()
    cnt, covered, window = [0] * n, 0, deque()
    match_indices = []
    for v, sid in items:
        window.append((v, sid))
        if cnt[sid] == 0:
            covered += 1
        cnt[sid] += 1

        while window and (abs(window[-1][0] - window[0][0]) >= width):
            lv, lid = window.popleft()
            cnt[lid] -= 1
            if cnt[lid] == 0:
                covered -= 1

        if covered == n:
            center_idx = (window[0][0] + window[-1][0]) // 2
            match_indices.append(center_idx)
            cnt, covered = [0] * n, 0
            window.clear()

    return match_indices

def find_all_occurs(text: str, substr: str, lemmatize: bool = False) -> List[int]:
    """
    :param text: Source text (haystack)
    :param substr: Substring (needle)
    :param lemmatize: Use lemmas
    :return: Number of occurrences
    """
    if not substr or not text:
        return []

    if not lemmatize:
        return [m.start() for m in re.finditer(f'(?={re.escape(substr)})', text, re.IGNORECASE)]

    parsed_substr = gv.ru_lemmatizer(substr)
    target_tokens, occurs = substr.split(), []
    for i in range(len(target_tokens)):
        target_lemma: str = parsed_substr[i].lemma_
        lemmatized_text = lemmatize_text(text)
        lemma_indices = {idx for lemma, idx in lemmatized_text if lemma == target_lemma}

        if not lemma_indices:
            return []

        occurs.append(lemma_indices)

    window_width = max(150, len(substr) * gv.WINDOW_SIZE_MULTIPLIER)
    matches_indices = count_groups(occurs, window_width)
    return matches_indices


def get_most_specific_tokens(src: str | ExtractedTable, tokens: List[spacy.tokens.Token] | spacy.tokens.doc.Doc,
                             validator: Callable = validate_token,
                             fixed_ners: bool = True, include_ner_types: List[str] = None) -> Tuple[List[spacy.tokens.Token], int, List[int]]:
    """
    :param src: Source text (haystack)
    :param tokens: Doc tokens list (needle)

    :param validator: Function to validate tokens (to exclude stopwords, punctuation, etc.)
    :param fixed_ners: If True, named entities would be forced to be in the answer to prevent losing critical information
    :param include_ner_types: Provide a list of NER types to be considered as NERs (e.g. ["PER", "ORG", "LOC"])
    :return: Most specific token(-s) out of given (min but not zero occurrences)
    """
    best_combination = [None, 10 ** 6, None]
    if not isinstance(tokens, list) and not isinstance(tokens, spacy.tokens.doc.Doc):
        raise TypeError("Field 'tokens' must be a list or a spacy Doc")
    if not tokens or not src:
        return best_combination

    tokens_to_be_combined, fixed_tokens = [], []
    if fixed_ners:
        for token in tokens:
            if not validator(token):
                continue

            if len(token.ent_type_) > 1:

                if include_ner_types is not None:
                    if not any([1 for tp in include_ner_types if token.ent_type_.startswith(tp)]):
                        tokens_to_be_combined.append(token.i)
                        continue
                fixed_tokens.append(token)
                continue

            tokens_to_be_combined.append(token.i)
    else:
        for token in tokens:
            if validator(token):
                tokens_to_be_combined.append(token.i)

    for amount_tokens in range(1, len(tokens_to_be_combined) + 1):
        for comb in itertools.combinations(tokens_to_be_combined, amount_tokens):
            current_tokens = [tokens[c].text for c in comb] + [t.text for t in fixed_tokens]
            occurrences = find_all_occurs(src, " ".join(current_tokens),
                                              lemmatize=True)

            if 0 < len(occurrences) < best_combination[1]:
                best_combination = [comb, len(occurrences), occurrences]

    if best_combination[0] is None:
        return best_combination
    best_combination[0] = sorted([tokens[c] for c in best_combination[0]] + fixed_tokens, key=lambda t: t.i)
    return best_combination


def get_closest_table_to_index(tables: List[ExtractedTable], target: int) -> Tuple[ExtractedTable, int]:
    """
    :param tables:
    :param target:
    :return: table and distance
    """
    closest_table = [None, 10 ** 6]
    for t in tables:
        if not hasattr(t, "index"):
            continue
        if t.index < target:
            continue
        dist = t.index - target
        if 0 <= dist < 1024:
            if dist < closest_table[1]:
                closest_table = [t, dist]
    return closest_table[0], closest_table[1]


def extract_tech_specs(text: str, tables: List[ExtractedTable], headers, padding: int = 0) -> Union[str, ExtractedTable]:
    """
    :param text:
    :param tables:
    :param headers:
    :param padding:
    :return:
    """
    try:
        synonyms = ["технические характеристики", "техническая характеристика", "основные характеристики", "технические данные",
                    "technical characteristics", "technical parameters", "technical specification", "technical data", ]
        start, end = None, None

        if headers:
            for syn in synonyms:
                for i in range(len(headers) - 1):
                    if headers[i][1].endswith(".....") or headers[i][1][:-3].endswith("....."):
                        continue
                    if syn in headers[i][1].lower():
                        if start is None or headers[i][0] < start:
                            if headers[i + 1][0] - headers[i][0] < 1024 and get_closest_table_to_index(tables, headers[i][0] + padding)[0] is None:
                                continue
                            start = headers[i][0]
                            end = headers[i+1][0]
                            break

        if start in [None, -1] or end in [None, -1] or start > end or (end - start) > 15000:
            occurs, i = find_all_occurs(text, synonyms[0]), 1
            while not occurs and i < len(synonyms):
                occurs = find_all_occurs(text, synonyms[i])
                i += 1
            if len(occurs) == 0:
                return ""
            index_cands, main_cand = [], occurs[0]
            if len(occurs) > 1:
                index_cands, main_cand = [occurs[0]], occurs[-1]
                if len(occurs) > 2:
                    index_cands += occurs[1:-1]
            next_chapter = None
            for i in index_cands:
                if text[i:].find('\n') != -1:
                    frame_end = text[i:].find('\n') + i
                    frame = text[i:frame_end]
                    match = gv.header_pattern.search(frame)
                    if match is None:
                        match = gv.header_pattern.search(frame, re.IGNORECASE)
                    if match is None:
                        match = gv.header_pattern.search(text[frame_end: text[frame_end:].find('\n') + frame_end], re.IGNORECASE)
                    if match:
                        next_chapter = match.group(2).strip()
                        index_cands.remove(i)
                        break
            # if next_chapter is None and len(occurs) > 1:
            #     return ""
            if next_chapter is None:
                frame_end = text[main_cand:].find('\n') + main_cand
                frame = text[ main_cand : frame_end]
                prefix = gv.header_pattern.match(frame)
                if not prefix:
                    prefix = gv.header_pattern.match(frame, re.IGNORECASE)
                if not prefix:
                    prefix = gv.header_pattern.match(text[frame_end: text[frame_end:].find('\n') + frame_end], re.IGNORECASE)
                if not prefix:
                    return ""
                prefix = prefix.group(2).strip()
                if '.' in prefix and prefix[-1] != '.':
                    next_prefix = prefix[:-1] + f"{int(prefix[-1]) + 1}"
                else:
                    next_prefix = f"{int(prefix[0]) + 1}" + prefix[1:]
                next_chapter = next_prefix
            while text[main_cand:].find(next_chapter) == -1 and index_cands:
                main_cand = index_cands.pop(0)
            if text[main_cand:].find(next_chapter) != -1:
                start = main_cand
                end = text[main_cand:].find(next_chapter) + main_cand

        if start in [None, -1] or end in [None, -1] or start > end or (end - start) > 15000:
            return ""

        result = text[start: end+1]
        if end - start < 1024:
            some_res = get_closest_table_to_index(tables, start + padding)
            if some_res[0]:
                result += "\n\nТаблица:" + extracted_table_repr(some_res[0])
        return result
    except Exception as e:
        print("\033[33m{}\033[0m".format(f"\t\tWARNING! Got exception {e} while extracting tech specs"))
        return ""

def fix_homoglyphs(text: str) -> str:
    if not text: return text
    mapping = str.maketrans("aceopxyABEKMHOPCTX", "асеорхуАВЕКМНОРСТХ")
    words = text.split()
    fixed_words = []
    for word in words:
        cyr_count = len(re.findall(r'[а-яА-ЯёЁ]', word))
        lat_count = len(re.findall(r'[a-zA-Z]', word))
        if cyr_count > 0 and lat_count > 0:
            if cyr_count >= lat_count:
                word = word.translate(mapping)
        elif lat_count > 3 and all(c in "aceopxybkmht0123456789-" for c in word.lower()):
            word = word.translate(mapping)
        fixed_words.append(word)
    return " ".join(fixed_words)


def preload_file(file_id: int, current_tag: str) -> dict:
    """
    :param file_id:
    :param current_tag:
    :return:
    """
    pure_text, headers, file_tables = gv.cur.execute("SELECT pure_text, headers, tables FROM files WHERE file_id = ?", (file_id,)).fetchone()
    file_tables = pickle.loads(file_tables)
    headers, relevant_headers = json.loads(headers), []
    cur_tag_occurs, frames, relevant_parts = find_all_occurs(pure_text, current_tag), [], []
    if not cur_tag_occurs:
        current_tag = fix_homoglyphs(current_tag)
        cur_tag_occurs, frames, relevant_parts = find_all_occurs(pure_text, current_tag), [], []
    exit_after_this_frame = False
    for c in cur_tag_occurs:
        frame, end = pure_text[c:], -1
        for other_tag in gv.ALL_TAGS:
            if other_tag[:-1] == current_tag[:-1]:
                continue
            other_tag_occur = frame.find(other_tag)
            if other_tag_occur != -1:
                if other_tag_occur < end or end == -1:
                    end = frame.find(other_tag)
        if end == -1:
            end = len(frame)
            exit_after_this_frame = True
        bulging_part = None
        for s, e in relevant_parts:
            if s <= c <= e:
                bulging_part = max(0, end - e)
                break
        if bulging_part != 0 and len(frame[:end]) > 120:
            frame_start = (end - bulging_part) if bulging_part is not None else c
            frame = pure_text[ frame_start : frame_start + end ] + f"(!@#$)EOF;{frame_start};{frame_start + end};(!@#$)"
            relevant_parts.append((frame_start, frame_start + end))
            frames.append(frame)
            if exit_after_this_frame:
                break
    first_mention = min([r[0] for r in relevant_parts], default=0)
    closest_passport = [None, 10**6]
    relevant_tables = []
    for t in file_tables:
        pure_values = t.df.reset_index().values.flatten()
        pure_values = " ".join(str(val) for val in pure_values if pd.notna(val)).lower()
        passport_matches = sum([1 for k in ["vendor", "contractor", "location", "подрядчик", "поставщик", "заказчик"] if k in pure_values])
        if passport_matches > 1:
            if abs(t.index - first_mention) < closest_passport[1]:
                closest_passport = [t, abs(t.index - first_mention)]
        if hasattr(t, "tags"):
            if current_tag in t.tags:
                relevant_tables.append(t)
                continue
        for s, e in relevant_parts:
            if s <= t.index <= e:
                relevant_tables.append(t)
                break
    for h in headers:
        for s, e in relevant_parts:
            if s <= h[0] <= e:
                relevant_headers.append([h[0] - s, h[1]])
                break
    pure_text = "\n".join(frames)
    tech_specs = extract_tech_specs(pure_text, relevant_tables, relevant_headers, padding=first_mention)
    object_type = None
    match current_tag:
        case "71-V-2002A":
            object_type = "горизонтальная ёмкость"
        case "7230-HV-0011":
            object_type = "задвижка клиновая"
        case "72-P-4001A":
            object_type = "агрегат электронасосный"
        case "7350-SJ-0001B":
            object_type = "УУ дренчерный"
        case "8500-SS-01-T-SX-2001":
            object_type = "коммутационный шкаф"
        case _:
            object_type = ""
    print(f"Object type: {object_type}")
    return {"pure_text": pure_text, "tables": relevant_tables, "object_type": object_type,"tech_specs": tech_specs, "passport": closest_passport[0]}
