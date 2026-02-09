#!/usr/bin/env python3
"""
Simple script to verify database contents
"""
import psycopg2
import sys

try:
    from db_config import DB_CONFIG
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Get total count
    cursor.execute('SELECT COUNT(*) FROM image_metadata')
    total = cursor.fetchone()[0]
    print(f'✓ Total images in database: {total:,}')

    # Get by dataset
    cursor.execute('''
        SELECT dataset_name, COUNT(*) 
        FROM image_metadata 
        GROUP BY dataset_name 
        ORDER BY dataset_name
    ''')
    print('\nDataset breakdown:')
    for dataset, count in cursor.fetchall():
        print(f'  • {dataset}: {count:,} images')

    # Sample records
    cursor.execute('SELECT file_path, width, height, format FROM image_metadata LIMIT 5')
    print('\nSample records:')
    for path, width, height, fmt in cursor.fetchall():
        filename = path.split('/')[-1]
        print(f'  • {filename}: {width}x{height} {fmt}')

    conn.close()
    print('\n✅ Database verification complete!')

except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)

