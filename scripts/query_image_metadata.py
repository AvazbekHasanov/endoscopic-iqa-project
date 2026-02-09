"""
Query Script for Image Metadata Database
Run queries to explore the image metadata stored in PostgreSQL
"""

import psycopg2
from tabulate import tabulate
import sys

try:
    from db_config import DB_CONFIG
except ImportError:
    print("Error: db_config.py not found")
    sys.exit(1)


class ImageMetadataQuery:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        print(f"✓ Connected to database: {db_config['database']}\n")

    def query_all_datasets(self):
        """Get all datasets with counts"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT dataset_name, COUNT(*) as count,
                   AVG(width) as avg_width, AVG(height) as avg_height
            FROM image_metadata
            GROUP BY dataset_name
            ORDER BY count DESC
        """)
        results = cursor.fetchall()
        headers = ['Dataset', 'Images', 'Avg Width', 'Avg Height']
        print("📦 DATASETS SUMMARY")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()

    def query_categories_by_dataset(self, dataset_name):
        """Get all categories for a specific dataset"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT category, subcategory, COUNT(*) as count
            FROM image_metadata
            WHERE dataset_name = %s
            GROUP BY category, subcategory
            ORDER BY count DESC
        """, (dataset_name,))
        results = cursor.fetchall()
        headers = ['Category', 'Subcategory', 'Count']
        print(f"📂 CATEGORIES IN {dataset_name}")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()

    def query_image_formats(self):
        """Get distribution of image formats"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT format, COUNT(*) as count, 
                   AVG(size_bytes)/1024 as avg_size_kb
            FROM image_metadata
            GROUP BY format
            ORDER BY count DESC
        """)
        results = cursor.fetchall()
        headers = ['Format', 'Count', 'Avg Size (KB)']
        print("🖼️  IMAGE FORMATS")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()

    def query_images_by_size_range(self, min_width, max_width, min_height, max_height):
        """Find images within a specific size range"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT file_path, width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            WHERE width BETWEEN %s AND %s
              AND height BETWEEN %s AND %s
            LIMIT 20
        """, (min_width, max_width, min_height, max_height))
        results = cursor.fetchall()
        headers = ['File Path', 'Width', 'Height', 'Size (KB)']
        print(f"🔍 IMAGES WITH SIZE {min_width}-{max_width}x{min_height}-{max_height}")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()

    def query_largest_images(self, limit=10):
        """Get the largest images by file size"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT filename, dataset_name, category, 
                   width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            ORDER BY size_bytes DESC
            LIMIT %s
        """, (limit,))
        results = cursor.fetchall()
        headers = ['Filename', 'Dataset', 'Category', 'Width', 'Height', 'Size (KB)']
        print(f"📏 TOP {limit} LARGEST IMAGES")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid', maxcolwidths=[30, 20, 30, 10, 10, 10]))
        print()

    def query_smallest_images(self, limit=10):
        """Get the smallest images by file size"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT filename, dataset_name, category, 
                   width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            ORDER BY size_bytes ASC
            LIMIT %s
        """, (limit,))
        results = cursor.fetchall()
        headers = ['Filename', 'Dataset', 'Category', 'Width', 'Height', 'Size (KB)']
        print(f"📏 TOP {limit} SMALLEST IMAGES")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid', maxcolwidths=[30, 20, 30, 10, 10, 10]))
        print()

    def query_images_by_category(self, category_pattern):
        """Search for images by category pattern"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT dataset_name, category, subcategory, COUNT(*) as count
            FROM image_metadata
            WHERE category ILIKE %s OR subcategory ILIKE %s
            GROUP BY dataset_name, category, subcategory
            ORDER BY count DESC
        """, (f'%{category_pattern}%', f'%{category_pattern}%'))
        results = cursor.fetchall()
        headers = ['Dataset', 'Category', 'Subcategory', 'Count']
        print(f"🔎 SEARCH RESULTS FOR: '{category_pattern}'")
        print("=" * 80)
        print(tabulate(results, headers=headers, tablefmt='grid'))
        print()

    def export_to_csv(self, output_file='image_metadata.csv'):
        """Export all metadata to CSV"""
        import csv
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT file_path, dataset_name, category, subcategory, 
                   filename, width, height, format, mode, 
                   size_bytes, md5_hash, created_at
            FROM image_metadata
            ORDER BY dataset_name, category, filename
        """)

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_path', 'dataset_name', 'category', 'subcategory',
                           'filename', 'width', 'height', 'format', 'mode',
                           'size_bytes', 'md5_hash', 'created_at'])
            writer.writerows(cursor.fetchall())

        print(f"✓ Exported to {output_file}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    print("=" * 80)
    print("🔍 IMAGE METADATA QUERY TOOL")
    print("=" * 80)
    print()

    query = ImageMetadataQuery(DB_CONFIG)

    # Run various queries
    query.query_all_datasets()
    query.query_image_formats()
    query.query_largest_images(10)
    query.query_smallest_images(10)

    # Query specific datasets
    print("\n")
    query.query_categories_by_dataset('Gastrovision')

    print("\n")
    query.query_categories_by_dataset('lower-gi-tract')

    print("\n")
    query.query_categories_by_dataset('upper-gi-tract')

    # Search for polyps
    print("\n")
    query.query_images_by_category('polyp')

    # Export to CSV
    print("\n")
    query.export_to_csv('image_metadata_export.csv')

    query.close()
    print("\n✓ Query session complete!")


if __name__ == "__main__":
    main()

