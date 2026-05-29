import sqlite3
import os

DB_NAME = 'file_bank.db'

if os.path.exists(DB_NAME):
    os.remove(DB_NAME)

conn = sqlite3.connect(DB_NAME)
cur = conn.cursor()

cur.execute("PRAGMA foreign_keys = ON;")
cur.execute("""
CREATE TABLE files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT UNIQUE NOT NULL,
    pages INTEGER CHECK (pages > 0),
    pure_text TEXT NOT NULL,
    headers TEXT,
    mentioned_tags TEXT DEFAULT '',
    tables BLOB,
    last_update DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
cur.execute("""
CREATE TRIGGER update_files_last_update
AFTER UPDATE ON files
FOR EACH ROW
WHEN NEW.last_update = OLD.last_update
BEGIN
    UPDATE files
       SET last_update = CURRENT_TIMESTAMP
     WHERE file_id = NEW.file_id;
END;
""")

cur.execute("DROP TABLE IF EXISTS properties;")
cur.execute("""
CREATE TABLE properties (
    prop_id INTEGER PRIMARY KEY AUTOINCREMENT,
    external_id VARCHAR(255) UNIQUE DEFAULT 'none',
    prop_name_eng VARCHAR(255) UNIQUE NOT NULL,
    prop_name_rus VARCHAR(255) NOT NULL,
    prop_desc_eng TEXT NOT NULL DEFAULT 'none',
    prop_desc_rus TEXT NOT NULL DEFAULT 'none',
    condition VARCHAR(255) NOT NULL CHECK (condition IN ('none', 'search_in_header', 'only_if_mentioned')),
    alt_keywords TEXT NOT NULL DEFAULT 'none',
    desc_for_llm TEXT,
    flag int DEFAULT 0,
    last_update DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
cur.execute("""
CREATE TRIGGER update_property_last_update
AFTER UPDATE ON properties
FOR EACH ROW
WHEN NEW.last_update = OLD.last_update
BEGIN
    UPDATE properties
       SET last_update = CURRENT_TIMESTAMP
     WHERE prop_id = NEW.prop_id;
END;
""")
conn.commit()
conn.close()
print("Database created successfully")
