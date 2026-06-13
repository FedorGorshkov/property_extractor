import sqlite3
import pickle
import re

from img2table.tables.objects.extraction import ExtractedTable
import tabulate

import src.global_variables as gv

TAG = "7230-HV-0011"
# TAG = "72-P-4001A"
# TAG = "71-V-2002A"
# TAG = "7350-SJ-0001B"
# TAG = "8500-SS-01-T-SX-2001"
OUTPUT_FILE = f"../test2.txt"
#
# print(f"Tag {TAG}", "found" if TAG in gv.ALL_TAGS else "not found", "in global_variables.ALL_TAGS")
conn = sqlite3.connect('../storage/file_bank.db')
cur = conn.cursor()
# Tid = cur.execute("SELECT file_id FROM files WHERE mentioned_tags LIKE ?", (f"%{TAG}%", )).fetchone()[0]
Tid = 13
filenames = '"' + ''", "''.join(cur.execute("SELECT file_name FROM files WHERE file_id = ?", (Tid,)).fetchone()) + '"'
print(f"Found at files: {filenames}")
pure_text, headers, tables = cur.execute("SELECT pure_text, headers, tables FROM files WHERE file_id = ?", (Tid, )).fetchone()
pure_text: str = pure_text
tables: list[ExtractedTable] = pickle.loads(tables)
pure_text = f"HEADERS:\n{headers}\n{pure_text}"
pure_text += "\n\n" + '-' * 100 + "\n" + "ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT ENDOFTEXT\n" + '-' * 100 + "\n\n"
for i in range(len(tables)):
    clean_df = tables[i].df.fillna('')
    index_at_text = tables[i].index
    page_num = re.search(r'\[!@#\$%]EOP(\d+)', pure_text[index_at_text:]).group(1)
    clean_df.index.name = f"Table №{i}\nOccurs at page {page_num} ({index_at_text})"
    pure_text += tabulate.tabulate(clean_df, headers='keys', tablefmt='grid') + "\n\n"
with open(OUTPUT_FILE, "w") as f:
    f.write(pure_text)
conn.close()
