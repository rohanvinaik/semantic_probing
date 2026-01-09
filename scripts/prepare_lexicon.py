#!/usr/bin/env python3
"""Extract top 50K words from GSE lexicon for demo distribution."""
import sqlite3
import shutil
import os
from pathlib import Path

SOURCE_DB = Path("/Users/rohanvinaik/gse/data/lexicon.db")
OUTPUT_DB = Path(__file__).parent.parent / "data" / "lexicon_demo.db"
TARGET_COUNT = 50_000

def main():
    if not SOURCE_DB.exists():
        print(f"Source DB not found at {SOURCE_DB}. Skipping.")
        return

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        os.remove(OUTPUT_DB)
        
    shutil.copy(SOURCE_DB, OUTPUT_DB)

    conn = sqlite3.connect(OUTPUT_DB)
    # Check if frequency_rank exists, otherwise use rowid or id
    try:
        conn.execute("""
            DELETE FROM lexemes WHERE id NOT IN (
                SELECT id FROM lexemes
                ORDER BY COALESCE(frequency_rank, 999999) ASC
                LIMIT ?
            )
        """, (TARGET_COUNT,))
    except sqlite3.OperationalError:
        print("Warning: frequency_rank column not found, limiting by ID.")
        conn.execute("""
            DELETE FROM lexemes WHERE id NOT IN (
                SELECT id FROM lexemes
                LIMIT ?
            )
        """, (TARGET_COUNT,))

    try:
        conn.execute("DELETE FROM primitive_values WHERE lexeme_id NOT IN (SELECT id FROM lexemes)")
    except sqlite3.OperationalError:
        pass # table might not exist
        
    conn.execute("VACUUM")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM lexemes").fetchone()[0]
    print(f"Created {OUTPUT_DB}: {count} words")
    conn.close()

if __name__ == "__main__":
    main()
