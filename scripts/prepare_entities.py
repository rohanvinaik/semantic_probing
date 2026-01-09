#!/usr/bin/env python3
"""Extract Vital Articles Level 1-3 (~5K entities) for demo distribution."""
import sqlite3
import shutil
import os
from pathlib import Path

SOURCE_DB = Path("/Users/rohanvinaik/relational-ai/data/sparse_wiki.db")
OUTPUT_DB = Path(__file__).parent.parent / "data" / "entities_demo.db"
MAX_VITAL_LEVEL = 3

def main():
    if not SOURCE_DB.exists():
        print(f"Source DB not found at {SOURCE_DB}. Skipping.")
        return

    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DB.exists():
        os.remove(OUTPUT_DB)

    shutil.copy(SOURCE_DB, OUTPUT_DB)

    conn = sqlite3.connect(OUTPUT_DB)
    try:
        conn.execute("DELETE FROM entities WHERE vital_level IS NULL OR vital_level > ?", (MAX_VITAL_LEVEL,))
        conn.execute("DELETE FROM dimension_positions WHERE entity_id NOT IN (SELECT id FROM entities)")
        conn.execute("DELETE FROM epa_values WHERE entity_id NOT IN (SELECT id FROM entities)")
    except sqlite3.OperationalError as e:
        print(f"Warning during pruning: {e}")
        
    conn.execute("VACUUM")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    print(f"Created {OUTPUT_DB}: {count} entities")
    conn.close()

if __name__ == "__main__":
    main()
