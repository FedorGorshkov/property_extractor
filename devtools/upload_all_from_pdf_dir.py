import sys
import os
import subprocess

prefix = "../pdf/"
# files = os.listdir(prefix)
files = ["AGCC.287-7120-2048993-K32-0002_0.pdf", "AGCC.287-6410-2018142-K32-0001_03_RU.pdf", "AGCC.287-6163-20344024-K32-0001_01_RU.pdf", "AGCC.287-0000-2065880-K32-0001_01_RU.pdf"]
for file in reversed(files):
    subprocess.run([sys.executable, "../storage/pdf_upload.py", prefix + file])
    print('-'*100)
