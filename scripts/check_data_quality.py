"""
Data Quality Check Script
Analyzes the image metadata to find potential issues
"""

import psycopg2
from collections import defaultdict

try:
    from db_config import DB_CONFIG
except ImportError:
    print("Error: db_config.py not found")
    exit(1)


class DataQualityChecker:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        print(f"✓ Connected to database: {db_config['database']}\n")

    def check_duplicates(self):
        """Find duplicate images by MD5 hash"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT md5_hash, COUNT(*) as count, 
                   array_agg(file_path) as paths
            FROM image_metadata
            GROUP BY md5_hash
            HAVING COUNT(*) > 1
            ORDER BY count DESC
        """)
        results = cursor.fetchall()

        print("🔍 DUPLICATE IMAGES CHECK")
        print("=" * 80)
        if results:
            print(f"Found {len(results)} sets of duplicate images:\n")
            for md5, count, paths in results[:10]:  # Show first 10
                print(f"MD5: {md5} ({count} copies)")
                for path in paths:
                    print(f"  - {path}")
                print()
        else:
            print("✓ No duplicate images found!")
        print()

    def check_outliers(self):
        """Find images with unusual dimensions or file sizes"""
        cursor = self.conn.cursor()

        # Very small images
        cursor.execute("""
            SELECT file_path, width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            WHERE width < 300 OR height < 300
            ORDER BY width * height ASC
            LIMIT 10
        """)
        small_images = cursor.fetchall()

        # Very large images
        cursor.execute("""
            SELECT file_path, width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            WHERE width > 2000 OR height > 2000
            ORDER BY width * height DESC
            LIMIT 10
        """)
        large_images = cursor.fetchall()

        # Unusual file sizes (very small or large)
        cursor.execute("""
            SELECT file_path, width, height, size_bytes/1024 as size_kb
            FROM image_metadata
            WHERE size_bytes < 10000  -- Less than 10KB
            ORDER BY size_bytes ASC
            LIMIT 10
        """)
        small_files = cursor.fetchall()

        cursor.execute("""
            SELECT file_path, width, height, size_bytes/1024/1024 as size_mb
            FROM image_metadata
            WHERE size_bytes > 5000000  -- More than 5MB
            ORDER BY size_bytes DESC
            LIMIT 10
        """)
        large_files = cursor.fetchall()

        print("📏 DIMENSION OUTLIERS")
        print("=" * 80)
        if small_images:
            print("Small images (< 300px):")
            for path, w, h, size in small_images:
                print(f"  {w}x{h} ({size:.1f}KB): {path.split('/')[-1]}")
        print()

        if large_images:
            print("Large images (> 2000px):")
            for path, w, h, size in large_images:
                print(f"  {w}x{h} ({size:.1f}KB): {path.split('/')[-1]}")
        print()

        print("💾 FILE SIZE OUTLIERS")
        print("=" * 80)
        if small_files:
            print("Very small files (< 10KB):")
            for path, w, h, size in small_files:
                print(f"  {size:.1f}KB ({w}x{h}): {path.split('/')[-1]}")
        print()

        if large_files:
            print("Very large files (> 5MB):")
            for path, w, h, size in large_files:
                print(f"  {size:.2f}MB ({w}x{h}): {path.split('/')[-1]}")
        print()

    def check_aspect_ratios(self):
        """Find images with unusual aspect ratios"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT file_path, width, height,
                   ROUND(CAST(width AS NUMERIC) / CAST(height AS NUMERIC), 2) as aspect_ratio
            FROM image_metadata
            WHERE height > 0
              AND (CAST(width AS NUMERIC) / CAST(height AS NUMERIC) > 3 
                   OR CAST(width AS NUMERIC) / CAST(height AS NUMERIC) < 0.33)
            ORDER BY aspect_ratio DESC
            LIMIT 20
        """)
        results = cursor.fetchall()

        print("📐 UNUSUAL ASPECT RATIOS")
        print("=" * 80)
        if results:
            print("Images with extreme aspect ratios (> 3:1 or < 1:3):\n")
            for path, w, h, ratio in results:
                print(f"  {ratio} ({w}x{h}): {path.split('/')[-1]}")
        else:
            print("✓ All images have reasonable aspect ratios!")
        print()

    def check_format_consistency(self):
        """Check image format distribution"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT format, mode, COUNT(*) as count
            FROM image_metadata
            GROUP BY format, mode
            ORDER BY count DESC
        """)
        results = cursor.fetchall()

        print("🖼️  IMAGE FORMAT CONSISTENCY")
        print("=" * 80)
        for fmt, mode, count in results:
            print(f"  {fmt} ({mode}): {count} images")
        print()

    def check_dataset_balance(self):
        """Check dataset balance"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT dataset_name, category, COUNT(*) as count
            FROM image_metadata
            GROUP BY dataset_name, category
            ORDER BY count ASC
            LIMIT 20
        """)
        results = cursor.fetchall()

        print("⚖️  DATASET BALANCE (Smallest Categories)")
        print("=" * 80)
        for dataset, category, count in results:
            if count < 50:
                print(f"  ⚠️  {dataset}/{category}: {count} images (very small)")
            else:
                print(f"  {dataset}/{category}: {count} images")
        print()

    def generate_summary_stats(self):
        """Generate overall summary statistics"""
        cursor = self.conn.cursor()

        # Total stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_images,
                COUNT(DISTINCT dataset_name) as num_datasets,
                COUNT(DISTINCT category) as num_categories,
                AVG(width) as avg_width,
                AVG(height) as avg_height,
                AVG(size_bytes)/1024 as avg_size_kb,
                MIN(width) as min_width,
                MAX(width) as max_width,
                MIN(height) as min_height,
                MAX(height) as max_height,
                MIN(size_bytes)/1024 as min_size_kb,
                MAX(size_bytes)/1024 as max_size_kb
            FROM image_metadata
        """)
        stats = cursor.fetchone()

        print("📊 OVERALL STATISTICS")
        print("=" * 80)
        print(f"Total Images: {stats[0]:,}")
        print(f"Datasets: {stats[1]}")
        print(f"Categories: {stats[2]}")
        print(f"\nDimensions:")
        print(f"  Average: {stats[3]:.0f}x{stats[4]:.0f} pixels")
        print(f"  Width range: {stats[6]:.0f} - {stats[7]:.0f} pixels")
        print(f"  Height range: {stats[8]:.0f} - {stats[9]:.0f} pixels")
        print(f"\nFile Sizes:")
        print(f"  Average: {stats[5]:.2f} KB")
        print(f"  Range: {stats[10]:.2f} - {stats[11]:.2f} KB")
        print()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    print("=" * 80)
    print("🔬 DATA QUALITY CHECK")
    print("=" * 80)
    print()

    checker = DataQualityChecker(DB_CONFIG)

    checker.generate_summary_stats()
    checker.check_duplicates()
    checker.check_outliers()
    checker.check_aspect_ratios()
    checker.check_format_consistency()
    checker.check_dataset_balance()

    checker.close()
    print("=" * 80)
    print("✅ DATA QUALITY CHECK COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

