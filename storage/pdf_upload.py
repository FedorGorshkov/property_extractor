import shutil
CHECKPOINT_DIR = "temp_pdf_checkpoints"

# External modules
from img2table.document import Image as i2t_Image
import fitz
from PIL import Image as PIL_Image
import pandas as pd
import cv2
from PIL import ImageDraw

# Built-in modules
import pickle
import sys
from multiprocessing import Pool
from datetime import datetime
import re
import threading
import os
import io
import gc
import json
import argparse

# Project modules
import src.global_variables as gv
import storage.OptimizedTesseractOCR as OpTess
from storage.image_processing import ImageProcessor, slice_df_to_crop

PDF_PATH: str = None
doc: fitz.Document = None
ocr_engine: OpTess.OptimizedTesseractOCR = None
matrix = fitz.Matrix(OpTess.TARGET_DPI / 72.0, OpTess.TARGET_DPI / 72.0)
FLAG = "!@FLAG!@"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "output.pdf")


def preparse_pdf(input_path: str, start: int, end: int):
    """Обрезает исходный PDF до [start, end] включительно."""
    src_doc = fitz.open(input_path)
    if start is None:
        start = 0
    if end is None:
        end = len(src_doc) - 1
    start = max(0, start)
    end = min(len(src_doc) - 1, end)
    out_doc = fitz.open()
    out_doc.insert_pdf(src_doc, from_page=start, to_page=end)
    out_doc.save(OUTPUT_PATH, garbage=3, deflate=True)
    out_doc.close()
    src_doc.close()


def get_answer_with_timeout(timeout=5):
    if not sys.stdin or not sys.stdin.isatty():
        gv.original_print("Is it OK? (Y/N) [Y]: Y (non-interactive console)")
        return 'Y'
    answer = 'Y'
    def input_thread():
        nonlocal answer
        answer = input(f"Is it OK? (Y/N) [Y]: ").strip().upper()
    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        gv.original_print("Y")
    return 'Y' if not answer.lower().startswith("n") else "N"

def init_worker(pdf_path):
    global doc, ocr_engine
    ocr_engine = OpTess.OptimizedTesseractOCR(lang="eng+rus", psm=3)
    doc = fitz.open(pdf_path)

def delete_stamp(gray_array, pil_img) -> None:
    height, width = gray_array.shape
    edges = cv2.Canny(gray_array, 50, 150, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    main_frame = None
    min_frame_area = 0.5 * width * height
    found_candidates = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_frame_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if (w * h) > min_frame_area:
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    found_candidates.append((x, y, w, h, w * h))
    if found_candidates:
        found_candidates.sort(key=lambda x: x[4])
        main_frame = found_candidates[0][:4]
    if not main_frame:
        return
    x, y, w, h = main_frame
    roi = edges[y + h:, x:]
    roi_contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_cell_area, min_cell_area, cells_count = 11 * OpTess.TARGET_DPI, 50, 0
    for cnt in roi_contours:
        area = cv2.contourArea(cnt)
        if min_cell_area < area < max_cell_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if 4 <= len(approx) <= 8:
                x_c, y_c, w_c, h_c = cv2.boundingRect(approx)
                if 0.1 < (w_c / float(h_c)) < 10.0:
                    cells_count += 1
    if cells_count < 15:
        return
    draw = ImageDraw.Draw(pil_img)
    # draw.rectangle([0, 0, width, y + 2], fill="white") # Up
    draw.rectangle([x + w - 2, 0, width, height], fill="white")  # Right
    draw.rectangle([0, y + h - 2, width, height], fill="white")  # Down
    draw.rectangle([0, 0, x + 2, height], fill="white")  # Left


def is_double_page_sheet(gray_array) -> bool:
    """True, если на странице два одинаковых блока рядом (две страницы на одной)."""
    img_h, img_w = gray_array.shape
    page_area = img_w * img_h
    block_size = max(3, int(0.06 * OpTess.TARGET_DPI))
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    thresh = cv2.adaptiveThreshold(gray_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   block_size, 10)

    min_line_len = round(1.2 * OpTess.TARGET_DPI)
    hor_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1)))
    ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_len)))
    grid = cv2.add(hor_lines, ver_lines)

    gap_close = round(0.15 * OpTess.TARGET_DPI)
    grid_dilated = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (gap_close, gap_close)), iterations=2)
    contours, _ = cv2.findContours(grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frames = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 0.25 * page_area:
            continue
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 1:
            continue
        rectangularity = area / cnt_area
        if rectangularity > 1.2:
            continue
        frames.append((x, y, w, h, area))

    if len(frames) != 2:
        return False

    (x1, y1, w1, h1, a1), (x2, y2, w2, h2, a2) = sorted(frames, key=lambda f: f[0])

    # Каждая рамка ~40-55% страницы
    if not (0.40 < a1 / page_area < 0.55 and 0.40 < a2 / page_area < 0.55):
        return False
    # Рамки примерно одинакового размера (отношение площадей ≤ 1.2)
    if max(a1, a2) / min(a1, a2) > 1.2:
        return False
    # Левая рамка — в левой половине, правая — в правой
    if not (x1 + w1 / 2 < img_w * 0.5 < x2 + w2 / 2):
        return False
    # Рамки не перекрываются по горизонтали (или перекрытие минимальное)
    if max(0, (x1 + w1) - x2) > 0.05 * min(w1, w2):
        return False

    return True

def process_single_image(pil_img: PIL_Image.Image, i: int = None) -> dict:
    global ocr_engine, FLAG
    cleaned = ocr_engine.process_page_once(pil_img)
    img_proc = ImageProcessor(OpTess.TARGET_DPI, cleaned)
    full_df = ocr_engine.cached_df
    img_w, img_h = cleaned.size

    table_bboxes, grid_mask = img_proc.get_table_bboxes()
    valid_tables, collected_bboxes_full = [], []
    try:
        for (x1, y1, x2, y2) in table_bboxes:
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = cleaned.crop((x1, y1, x2, y2))
            crop = img_proc.draw_table_borders(crop, grid_mask, (x1, y1, x2, y2))
            with io.BytesIO() as buf:
                crop.save(buf, format="BMP")
                crop_bytes = buf.getvalue()

            ocr_engine.cached_df = slice_df_to_crop(full_df, x1, y1, x2, y2)
            try:
                crop_tables = i2t_Image(crop_bytes).extract_tables(ocr=ocr_engine, implicit_rows=True, implicit_columns=True)
            except Exception as ex:
                print(f"\nWarning: extract_tables failed on crop {(x1, y1, x2, y2)}: {ex}")
                crop_tables = []
            finally:
                ocr_engine.cached_df = full_df

            for table in crop_tables:
                if table.df.empty:
                    continue
                pure_values = table.df.reset_index().values.flatten()
                if len(set(pure_values)) < 4:
                    continue
                table._df = table.df.map( lambda v: OpTess.fix_tags(str(v)) if pd.notna(v) else v )
                joined = " ".join(str(v) for v in pure_values if pd.notna(v))
                tags = []
                if "паспорт" in joined.lower():
                    tags = [t for t in gv.ALL_TAGS if t in joined]

                bb = ( table.bbox.x1 + x1, table.bbox.y1 + y1, table.bbox.x2 + x1, table.bbox.y2 + y1 )
                collected_bboxes_full.append(bb)
                if bb[1] > 100:
                    previous_text = ocr_engine.get_text_in_rect( 0, max(0, bb[1] - 100), img_w - 1, bb[1] )
                    previous_text = OpTess.fix_tags(previous_text[-50:])
                else:
                    previous_text = FLAG
                valid_tables.append([previous_text, -1, table, tags])
            del crop, crop_bytes
    finally:
        ocr_engine._cached_df = full_df
    remained_text = ocr_engine.get_remaining_text(collected_bboxes_full)
    remained_text = OpTess.fix_tags(remained_text)
    return {"batches": valid_tables, "text": remained_text}


def process_page(i):
    global doc, ocr_engine, matrix, FLAG, CHECKPOINT_DIR
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"page_{i}.pkl")
    if os.path.exists(checkpoint_path):
        return i, True

    page = pix = pil_img = None
    try:
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix)
        pil_img = PIL_Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pil_img, gray_array = ImageProcessor.keep_only_pure_grayscale(pil_img, tolerance=45)
        delete_stamp(gray_array, pil_img)
        if is_double_page_sheet(gray_array):
            w, h = pil_img.size
            mid = w // 2
            left = pil_img.crop((0, 0, mid, h))
            right = pil_img.crop((mid, 0, w, h))
            sub_left = process_single_image(left, i=i)
            sub_right = process_single_image(right, i=i)
            subpages = [sub_left, sub_right]
            del left, right
            gc.collect()
        else:
            subpages = [process_single_image(pil_img, i=i)]

        with open(checkpoint_path, "wb") as f:
            pickle.dump({"subpages": subpages}, f)
        return i, False
    except Exception as e:
        print(f"\nError processing page {i}: {e}")
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"batches": [], "text": f"[OCR ERROR ON PAGE {i}]"}, f)
        return i, False
    finally:
        del page
        del pix
        del pil_img
        gc.collect()


def validate_match(match: re.Match) -> set:
    res = set()
    for m in match:
        found = m.group(1)
        if "AGCC" in found or "agcc" in found:
            continue
        if found not in res:
            validated = 0
            # "A"-"Z"
            for i in range(65, 91):
                if chr(i) in found:
                    validated += 1
                    break
            # "0"-"9"
            for i in range(10):
                if str(i) in found:
                    validated += 1
                    break
            if validated >= 2:
                res.add(found)
    return res


def extract_bold_headers(physical_to_logical: dict[int, tuple[int, int]]):
    global PDF_PATH
    some_doc = fitz.open(PDF_PATH)
    headers = []
    flexible_pattern = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
    for page in some_doc:
        first_logical, _ = physical_to_logical.get(page.number, (page.number, 1))
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            line_text = ""
            is_bold_line = False
            for l in b["lines"]:
                for s in l["spans"]:
                    if "Bold" in s["font"] or s["flags"] & 2:
                        is_bold_line = True
                    line_text += " " + s["text"]
            clean_line = re.sub(r"\s+", ' ', line_text).strip()
            if is_bold_line and flexible_pattern.search(clean_line):
                headers.append([first_logical, clean_line])
    some_doc.close()
    return headers


def main():
    global PDF_PATH
    parser = argparse.ArgumentParser(description="Extract tables, headers and pure text from PDF")
    parser.add_argument("source", help="PDF to extract from")
    parser.add_argument("--crop_from", help="Page number to crop pdf from (start)", type=int)
    parser.add_argument("--crop_to", help="Page number to crop pdf to (end)", type=int)
    args, _ = parser.parse_known_args()
    original_path = args.source
    if not os.path.exists(original_path):
        print(f"File {original_path} not found")
        exit(-1)
    if args.crop_from or args.crop_to:
        preparse_pdf(original_path, args.crop_from, args.crop_to)
        PDF_PATH = OUTPUT_PATH
    else:
        PDF_PATH = os.path.abspath(original_path)
    original_path, found_id = os.path.basename(original_path), None
    with fitz.open(PDF_PATH) as d:
        total_pages = len(d)
        step = max(total_pages // 100, 1)
        print(f'Got file "{original_path}" with {total_pages} pages')
        res = gv.cur.execute("SELECT file_id FROM files WHERE file_name = ?", (original_path,)).fetchone()
        if res:
            found_id = res[0]
            print("\033[34m{}\033[0m".format(f"INFO: The file, you're trying to upload, already exists in database and will be updated"))
            ans = get_answer_with_timeout()
            if ans.upper() != 'Y':
                print("Aborting...")
                exit(-1)
    start_time = datetime.now()
    print(f"Started page-by-page processing...")

    BAR_WIDTH = 100
    safe_processes_count = min(max(1, os.cpu_count() - 2), 6)
    with Pool(
        processes=safe_processes_count,
        initializer=init_worker,
        initargs=(PDF_PATH,),
        maxtasksperchild=6
    ) as pool:
        iterator = pool.imap_unordered(process_page, range(total_pages), chunksize=2)
        for n, (i, from_cache) in enumerate(iterator, 1):
            if (n % step == 0) or (n == total_pages):
                percent = n / total_pages
                filled_len = int(BAR_WIDTH * percent)
                bar = '█' * filled_len + '░' * (BAR_WIDTH - filled_len)
                gv.original_print(f'\rProgress: |{bar}| {percent * 100:.1f}% ({n}/{total_pages})   ', end='', flush=True)
    gv.original_print()

    print("Started assembling result from checkpoints...")
    tables_batches, all_text = [], {}
    physical_to_logical: dict[int, tuple[int, int]] = {}
    logical_idx = 0
    for i in range(total_pages):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"page_{i}.pkl")
        if not os.path.exists(checkpoint_path):
            print("\033[31m{}\033[0m".format(f"ERROR! No checkpoint for page {i}"))
            exit(-1)
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        subpages = data["subpages"]
        physical_to_logical[i] = (logical_idx, len(subpages))
        for sub in subpages:
            all_text[logical_idx] = sub["text"]
            for batch in sub["batches"]:
                prev_text, _, table, tags = batch
                tables_batches.append([prev_text, logical_idx, table, tags])
            logical_idx += 1
    if not all_text:
        print("\033[31m{}\033[0m".format(f"ERROR! No text was found in the file"))
        exit(-1)

    print("Started post-processing...")
    all_text = "\n".join(
        [all_text[key] + f"{gv.EOP_FLAG}EOP{key}{gv.EOP_FLAG}"
         for key in sorted(all_text.keys())]
    )
    all_text = re.sub(r'\n+', '\n', all_text)
    all_text = re.sub(r' +', ' ', all_text)

    all_tables = []
    headers = extract_bold_headers(physical_to_logical)
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mentioned_tags = validate_match(re.finditer(gv.TAG_REGULAR_EXP, all_text))
    headers.sort(key=lambda x: x[0])

    if not tables_batches:
        print("\033[33m{}\033[0m".format(f"WARNING! No tables were found in the file"))
    else:
        tables_batches.sort(key=lambda x: x[1])
        for batch in tables_batches:
            prev_text, page_i, table, passport_tags = batch
            char_idx = -1
            page_start_idx = max(0, all_text.find(f"{gv.EOP_FLAG}EOP{page_i - 1}{gv.EOP_FLAG}"))
            if prev_text and prev_text != FLAG:
                char_idx = all_text.find(prev_text, page_start_idx)
            if char_idx == -1:
                target_page = page_i - 1
                while target_page >= 0:
                    marker = f"{gv.EOP_FLAG}EOP{target_page}{gv.EOP_FLAG}"
                    found_idx = all_text.find(marker)
                    if found_idx != -1:
                        char_idx = found_idx + len(marker)
                        break
                    target_page -= 1
                char_idx = max(0, char_idx)
            else:
                char_idx += len(prev_text)

            if passport_tags:
                all_text = all_text[:char_idx] + ";".join(passport_tags) + all_text[char_idx:]
                char_idx += len(";".join(passport_tags))
            clean_df = table.df.reset_index().values.flatten()
            for j in range(len(headers)):
                if headers[j][1] in clean_df:
                    headers[j] = [headers[j][0], headers[j][1], char_idx]
            f_tags = validate_match(
                re.finditer(gv.TAG_REGULAR_EXP,
                            ' '.join(str(v) for v in clean_df if pd.notna(v)))
            )
            if f_tags:
                table.tags = f_tags
                mentioned_tags = mentioned_tags.union(f_tags)
            all_text = all_text[:char_idx] + ";".join(f_tags) + all_text[char_idx:]
            char_idx += len(";".join(f_tags))
            table.index = char_idx
            all_tables.append(table)

    if headers:
        final_headers = []
        for h in headers:
            if len(h) != 2: continue
            h_page, h_text = h
            p_st = all_text.find(f"{gv.EOP_FLAG}EOP{h_page - 1}{gv.EOP_FLAG}")
            p_st = max(0, p_st)
            p_end = all_text.find(f"{gv.EOP_FLAG}EOP{h_page}{gv.EOP_FLAG}")
            if p_end == -1: p_end = len(all_text)
            match = gv.header_pattern.match(h_text)
            search_query = match.group(2).strip() if match else h_text
            local_idx = all_text[p_st:p_end].find(search_query)
            if local_idx != -1:
                final_headers.append([p_st + local_idx, h_text])
            else:
                prefix = match.group(1).strip() if match else h_text[:5]
                local_idx = all_text[p_st:p_end].find(prefix)
                if local_idx != -1:
                    final_headers.append([p_st + local_idx, h_text])
        headers = final_headers
    if "бемёр" in original_path.lower():
        mentioned_tags = {"AABA003", "AAAJ801", "AAAH913", "AAAJ916", "AAAJ953", "AAAJ954"}
    if not headers:
        print("\033[33m{}\033[0m".format(f"WARNING! No headers were found in the file"))
    print(f"Found {len(all_tables)} tables, text with len {len(all_text)}, {len(headers)} headers and {len(mentioned_tags)} unique tags." )
    if found_id:
        gv.cur.execute(
            "UPDATE files SET pure_text = ?, headers = ?, mentioned_tags = ?, tables = ? WHERE file_id = ?",
            (all_text, json.dumps(headers, ensure_ascii=False), ",".join(mentioned_tags), pickle.dumps(all_tables), found_id), )
    else:
        gv.cur.execute( "INSERT INTO files (file_name, pages, pure_text, headers, mentioned_tags, tables) VALUES (?, ?, ?, ?, ?, ?)",
            ( os.path.basename(original_path), total_pages, all_text, json.dumps(headers, ensure_ascii=False), ",".join(mentioned_tags), pickle.dumps(all_tables) ), )
    gv.cur.close()
    gv.conn.commit()
    total_seconds = (datetime.now() - start_time).total_seconds()
    print("\033[32m{}\033[0m".format(f"Done! Parsing file took {round(total_seconds // 60)} minutes and {round(total_seconds % 60)} seconds"))

if __name__ == "__main__":
    main()
