#!/usr/bin/env python3
"""
Simple test script to verify database connection and tables.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from db_config import DB_CONFIG
    import psycopg2

    print("🔍 Testing Database Connection...")
    print("=" * 60)

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"✓ Connected to database: {DB_CONFIG['database']}")

    cursor = conn.cursor()

    # Check image_metadata table
    try:
        cursor.execute('SELECT COUNT(*) FROM image_metadata')
        count = cursor.fetchone()[0]
        print(f"✓ image_metadata table: {count} images")
    except Exception as e:
        print(f"⚠️  image_metadata table: Not found or empty")

    # Check image_quality_metrics table
    try:
        cursor.execute('SELECT COUNT(*) FROM image_quality_metrics')
        count = cursor.fetchone()[0]
        print(f"✓ image_quality_metrics table: {count} images")
    except Exception as e:
        print(f"⚠️  image_quality_metrics table: Not found or empty")

    conn.close()
    print("=" * 60)
    print("✓ Database test complete!")

except ImportError:
    print("✗ Error: Cannot import db_config.py")
    print("  Please ensure db_config.py exists in the scripts directory")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

