# Property Extractor

Automated extraction of equipment property values from scanned engineering PDF documentation into Excel property registers.

## What it does

Given an Excel **property register** (a table listing equipment tags and property names) and a **file bank** of indexed technical passports, the tool automatically finds and fills in property values and units of measurement by:

1. Locating the relevant document(s) for each equipment tag
2. Running a cascade of search strategies — table search → full-text search (with lemmatization) → token-level search → technical specification section fallback
3. Passing the narrowed-down context to a local LLM (Gemma 12B via Ollama) to extract the exact value

It targets **scanned (image-only) PDFs** — documents with no text layer — which rules out simple text extraction and requires a full OCR + computer vision pipeline.

## Architecture

```
PDF files ──► pdf_upload.py ──► SQLite (file_bank.db)
                                      │
Excel register ──► main.py ──► Solver │◄── file_bank.db
                                      │
                               LLM (Ollama)
                                      │
                               Filled Excel ◄──────────────
```

### Indexing phase (`storage/pdf_upload.py`)

Each PDF is processed page by page in a multiprocessing pool:

- **Rendering** — PyMuPDF renders pages at 288 DPI (4× scale over the PDF base of 72 DPI), giving 2380×3368 px for A4
- **Preprocessing** — colored pixels (stamps, watermarks) are replaced with background color; the result is converted to grayscale
- **Table detection** — `ImageProcessor` uses OpenCV morphological operators to find horizontal/vertical grid lines, then runs a recursive multi-scale candidate search with IoU-based deduplication and frame-wrapper filtering
- **OCR** — `OptimizedTesseractOCR` (a Cython wrapper over the Tesseract C++ API via `tesserocr`) runs once per page, caches the full hOCR result as a Polars DataFrame, and serves all table crops from that cache without re-running the engine

- **Storage** — extracted text, serialized table objects (pickle), JSON headers, and a comma-separated tag list are written to SQLite

### Extraction phase (`src/main.py` + `src/methods.py`)

`main.py` iterates rows of the input register, queries the file bank for documents mentioning the equipment tag, pre-loads a compact slice of each document (`preload_file`), and hands everything to a `Solver` instance.

**`Solver`** is a class whose search methods are decorated with `@func_props(confidence=...)`. On construction it collects all decorated methods via `__dict__` introspection, sorts them by descending confidence, and `solve()` runs them in order with a rollback mechanism — a weaker result never overwrites a stronger one:

| Method | Confidence | Strategy |
|---|---|---|
| `table_search_en` | 0.85 | Search property name in extracted tables (EN) |
| `table_search_ru` | 0.80 | Search property name in extracted tables (RU) |
| `simple_search_en` | 0.60 | Lemmatized full-text search (EN) |
| `simple_search_ru` | 0.50 | Lemmatized full-text search (RU) |
| `tokens_search_en` | 0.30 | Most-specific token subset search (EN) |
| `tokens_search_ru` | 0.25 | Most-specific token subset search (RU) |
| `search_in_tech_specs` | 0.05 | Search in extracted "Technical Characteristics" section |

Each successful match passes a tight context window to the LLM (`gemma3:12b` via Ollama's OpenAI-compatible API). The LLM must reply in JSON `{"value": "...", "uom": "..."}`. Contexts over 10 000 characters are batched with a 500-character overlap.

## Requirements

- **Python 3.11.9** (exact version required by the pre-built tesserocr wheel)
- **CUDA 12.1** compatible GPU (for torch — used by spaCy models)
- **Ollama** running locally with `gemma3:12b` pulled
- Windows (RUNME.bat, pre-built tesserocr wheel)

## Setup

```bat
RUNME.bat
```

This script:
1. Checks for Python 3.11.9
2. Creates (or recreates) a `.venv` virtual environment
3. Installs the pre-built `tesserocr` wheel (bundled in the repo)
4. Installs PyTorch with CUDA 12.1 support
5. Installs all `requirements.txt` dependencies
6. Downloads spaCy models `en_core_web_md` and `ru_core_news_lg`
7. Installs the project as an editable package (`pip install -e .`)

Tesseract itself is bundled in the `Tesseract-OCR/` directory with `eng`, `rus`, and `kaz` language packs — no separate installation needed.

## Usage

### 1. Index documents

Upload one or more PDF files to the file bank:

```bash
python storage/pdf_upload.py path/to/document.pdf
# optional page range:
python storage/pdf_upload.py path/to/document.pdf --crop_from 5 --crop_to 42
```

To batch-upload all PDFs from the `pdf/` directory, edit `devtools/upload_all_from_pdf_dir.py` and run it.

### 2. Extract properties

Edit `src/main.py` to point `xls_file` at your register, then run:

```bash
cd src
python main.py
```

Output is written to `xls/Test_output.xlsx`.

### 3. Upload property dictionary (one-time)

If you have JSON dumps from a corporate engineering data management system:

```bash
python storage/property_upload.py
```

This populates the `properties` table with English/Russian names, descriptions, search conditions, and LLM hints.

## Project structure

```
.
├── src/
│   ├── main.py               # Entry point: reads register, runs Solver, writes output
│   ├── methods.py            # Solver class with cascading search methods
│   ├── funcs.py              # NLP utilities: lemmatization, token search, table search
│   └── global_variables.py  # Config, DB connection, SpaCy models, LLM client
├── storage/
│   ├── pdf_upload.py         # Indexing pipeline: render → OCR → detect → store
│   ├── OptimizedTesseractOCR.py  # Cached Tesseract OCR engine (tesserocr + hOCR + Polars)
│   ├── image_processing.py  # Table bbox detection via OpenCV morphology
│   ├── property_upload.py   # Loads property dictionary from JSON dumps into DB
│   ├── DB_template.py       # Creates a fresh empty file_bank.db
│   ├── file_bank.db         # SQLite database (indexed documents + property dict)
│   ├── hardcode_config.json # Per-property search hints (conditions, alt keywords, LLM tips)
│   └── tags.csv             # Known equipment tag registry
├── devtools/
│   ├── upload_all_from_pdf_dir.py  # Batch PDF upload helper
│   └── get_parsed_file_by_tag.py   # Debug: dump file bank entry for a given tag
├── Tesseract-OCR/            # Bundled Tesseract 5.x (Windows), with eng/rus/kaz tessdata
├── xls/                      # Excel templates and output files
├── pdf/                      # Place input PDFs here
├── requirements.txt
├── setup.py
└── RUNME.bat                 # One-click setup script
```

## Database schema

### `files`

| Column | Type | Description |
|---|---|---|
| `file_id` | INTEGER PK | Auto-assigned file ID |
| `file_name` | TEXT UNIQUE | Original PDF filename |
| `pages` | INTEGER | Page count |
| `pure_text` | TEXT | Full extracted text with `EOP`/`EOF` markers |
| `headers` | TEXT | JSON array of `[char_position, header_text]` pairs |
| `mentioned_tags` | TEXT | Comma-separated tags found in the document |
| `tables` | BLOB | Pickle-serialized list of `ExtractedTable` objects |
| `last_update` | DATETIME | Auto-updated by trigger on every write |

### `properties`

| Column | Type | Description |
|---|---|---|
| `prop_id` | INTEGER PK | Auto-assigned property ID |
| `external_id` | VARCHAR | ID in the corporate engineering system |
| `prop_name_eng` | VARCHAR UNIQUE | English property name |
| `prop_name_rus` | VARCHAR | Russian property name |
| `prop_desc_eng` | TEXT | English description |
| `prop_desc_rus` | TEXT | Russian description |
| `condition` | VARCHAR | Search mode: `none` / `search_in_header` / `only_if_mentioned` |
| `alt_keywords` | TEXT | Semicolon-separated alternative search keywords |
| `desc_for_llm` | TEXT | Extra hint injected into the LLM prompt for this property |
| `flag` | INT | Special processing flag (0 = off, 1 = on) |

## Key dependencies

| Package | Purpose |
|---|---|
| `PyMuPDF` | PDF rendering at high DPI |
| `tesserocr` | Low-level Cython binding to Tesseract C++ API |
| `img2table` | Table structure extraction from images |
| `opencv-python` | Morphological image processing for table detection |
| `spacy` | NLP: lemmatization, POS tagging, NER (EN + RU models) |
| `polars` | Fast in-memory DataFrame for hOCR spatial filtering |
| `openpyxl` | Reading and writing `.xlsx` registers |
| `requests` | HTTP calls to the local Ollama LLM API |
