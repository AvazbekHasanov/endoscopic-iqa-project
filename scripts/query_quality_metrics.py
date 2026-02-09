"""
Query and analyze image quality metrics from the database.
Provides various queries to explore quality assessment results.
"""

import os
import sys
import psycopg2
from pathlib import Path
from tabulate import tabulate


class QualityMetricsQuery:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.connect()

    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'postgres'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', '')
            )
            print(f"✓ Connected to database: {self.db_config.get('database')}")
        except Exception as e:
            print(f"✗ Error connecting to database: {str(e)}")
            sys.exit(1)

    def get_images_by_quality_range(self, min_score=0.0, max_score=1.0, limit=20):
        """Get images within a quality score range"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.filename,
                m.dataset_name,
                m.category,
                q.quality_score,
                q.laplacian_variance,
                q.rms_contrast,
                q.noise_estimate,
                q.entropy
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            WHERE q.quality_score >= %s AND q.quality_score <= %s
            ORDER BY q.quality_score DESC
            LIMIT %s
        """, (min_score, max_score, limit))

        return cursor.fetchall()

    def get_poor_quality_images(self, threshold=0.3, limit=20):
        """Get images with quality score below threshold"""
        return self.get_images_by_quality_range(0.0, threshold, limit)

    def get_high_quality_images(self, threshold=0.7, limit=20):
        """Get images with quality score above threshold"""
        return self.get_images_by_quality_range(threshold, 1.0, limit)

    def get_blurry_images(self, threshold=100.0, limit=20):
        """Get potentially blurry images (low Laplacian variance)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.filename,
                m.dataset_name,
                m.category,
                q.quality_score,
                q.laplacian_variance
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            WHERE q.laplacian_variance < %s
            ORDER BY q.laplacian_variance ASC
            LIMIT %s
        """, (threshold, limit))

        return cursor.fetchall()

    def get_noisy_images(self, threshold=15.0, limit=20):
        """Get potentially noisy images (high noise estimate)"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.filename,
                m.dataset_name,
                m.category,
                q.quality_score,
                q.noise_estimate
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            WHERE q.noise_estimate > %s
            ORDER BY q.noise_estimate DESC
            LIMIT %s
        """, (threshold, limit))

        return cursor.fetchall()

    def get_dataset_quality_summary(self):
        """Get quality summary grouped by dataset"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.dataset_name,
                COUNT(*) as image_count,
                ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality,
                ROUND(MIN(q.quality_score)::numeric, 4) as min_quality,
                ROUND(MAX(q.quality_score)::numeric, 4) as max_quality,
                ROUND(STDDEV(q.quality_score)::numeric, 4) as std_quality
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            GROUP BY m.dataset_name
            ORDER BY avg_quality DESC
        """)

        return cursor.fetchall()

    def get_category_quality_summary(self, dataset_name=None):
        """Get quality summary grouped by category"""
        cursor = self.conn.cursor()

        if dataset_name:
            cursor.execute("""
                SELECT 
                    m.category,
                    COUNT(*) as image_count,
                    ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality,
                    ROUND(MIN(q.quality_score)::numeric, 4) as min_quality,
                    ROUND(MAX(q.quality_score)::numeric, 4) as max_quality
                FROM image_quality_metrics q
                JOIN image_metadata m ON q.image_id = m.id
                WHERE m.dataset_name = %s
                GROUP BY m.category
                ORDER BY avg_quality DESC
            """, (dataset_name,))
        else:
            cursor.execute("""
                SELECT 
                    m.dataset_name,
                    m.category,
                    COUNT(*) as image_count,
                    ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality,
                    ROUND(MIN(q.quality_score)::numeric, 4) as min_quality,
                    ROUND(MAX(q.quality_score)::numeric, 4) as max_quality
                FROM image_quality_metrics q
                JOIN image_metadata m ON q.image_id = m.id
                GROUP BY m.dataset_name, m.category
                ORDER BY m.dataset_name, avg_quality DESC
            """)

        return cursor.fetchall()

    def get_detailed_metrics(self, image_id=None, filename=None):
        """Get detailed metrics for a specific image"""
        cursor = self.conn.cursor()

        if image_id:
            cursor.execute("""
                SELECT 
                    m.id,
                    m.filename,
                    m.file_path,
                    m.dataset_name,
                    m.category,
                    m.width,
                    m.height,
                    q.quality_score,
                    q.laplacian_variance,
                    q.rms_contrast,
                    q.noise_estimate,
                    q.mscn_std,
                    q.gradient_energy,
                    q.entropy,
                    q.tenengrad,
                    q.processing_time_ms,
                    q.computed_at
                FROM image_quality_metrics q
                JOIN image_metadata m ON q.image_id = m.id
                WHERE m.id = %s
            """, (image_id,))
        elif filename:
            cursor.execute("""
                SELECT 
                    m.id,
                    m.filename,
                    m.file_path,
                    m.dataset_name,
                    m.category,
                    m.width,
                    m.height,
                    q.quality_score,
                    q.laplacian_variance,
                    q.rms_contrast,
                    q.noise_estimate,
                    q.mscn_std,
                    q.gradient_energy,
                    q.entropy,
                    q.tenengrad,
                    q.processing_time_ms,
                    q.computed_at
                FROM image_quality_metrics q
                JOIN image_metadata m ON q.image_id = m.id
                WHERE m.filename LIKE %s
            """, (f'%{filename}%',))
        else:
            return None

        return cursor.fetchone()

    def get_quality_distribution(self, bins=10):
        """Get distribution of quality scores"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                ROUND((quality_score * %s)::numeric, 0) / %s as bin_start,
                COUNT(*) as count
            FROM image_quality_metrics
            GROUP BY bin_start
            ORDER BY bin_start
        """, (bins, bins))

        return cursor.fetchall()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def display_table(data, headers):
    """Display data in a formatted table"""
    if data:
        print(tabulate(data, headers=headers, tablefmt='grid'))
    else:
        print("No data found.")


def main():
    # Import database configuration
    try:
        from db_config import DB_CONFIG
        db_config = DB_CONFIG
    except ImportError:
        print("⚠️  db_config.py not found, using defaults")
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': os.environ.get('USER', 'postgres'),
            'password': ''
        }

    print("=" * 80)
    print("🔍 IMAGE QUALITY METRICS QUERY TOOL")
    print("=" * 80)

    query = QualityMetricsQuery(db_config)

    while True:
        print("\n📋 MENU:")
        print("  1. View dataset quality summary")
        print("  2. View category quality summary")
        print("  3. View high quality images (score > 0.7)")
        print("  4. View poor quality images (score < 0.3)")
        print("  5. View potentially blurry images")
        print("  6. View potentially noisy images")
        print("  7. View quality distribution")
        print("  8. Search image by filename")
        print("  9. View images by quality range")
        print("  0. Exit")

        choice = input("\n👉 Select an option: ").strip()

        if choice == '0':
            break

        elif choice == '1':
            print("\n📊 DATASET QUALITY SUMMARY")
            print("=" * 80)
            data = query.get_dataset_quality_summary()
            headers = ['Dataset', 'Count', 'Avg Quality', 'Min', 'Max', 'Std Dev']
            display_table(data, headers)

        elif choice == '2':
            dataset = input("Enter dataset name (or leave empty for all): ").strip() or None
            print("\n📂 CATEGORY QUALITY SUMMARY")
            print("=" * 80)
            data = query.get_category_quality_summary(dataset)
            if dataset:
                headers = ['Category', 'Count', 'Avg Quality', 'Min', 'Max']
            else:
                headers = ['Dataset', 'Category', 'Count', 'Avg Quality', 'Min', 'Max']
            display_table(data, headers)

        elif choice == '3':
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print("\n🏆 HIGH QUALITY IMAGES")
            print("=" * 80)
            data = query.get_high_quality_images(limit=limit)
            headers = ['Filename', 'Dataset', 'Category', 'Quality', 'Laplacian', 'Contrast', 'Noise', 'Entropy']
            display_table(data, headers)

        elif choice == '4':
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print("\n⚠️  POOR QUALITY IMAGES")
            print("=" * 80)
            data = query.get_poor_quality_images(limit=limit)
            headers = ['Filename', 'Dataset', 'Category', 'Quality', 'Laplacian', 'Contrast', 'Noise', 'Entropy']
            display_table(data, headers)

        elif choice == '5':
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print("\n🌫️  POTENTIALLY BLURRY IMAGES")
            print("=" * 80)
            data = query.get_blurry_images(limit=limit)
            headers = ['Filename', 'Dataset', 'Category', 'Quality', 'Laplacian Var']
            display_table(data, headers)

        elif choice == '6':
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print("\n📊 POTENTIALLY NOISY IMAGES")
            print("=" * 80)
            data = query.get_noisy_images(limit=limit)
            headers = ['Filename', 'Dataset', 'Category', 'Quality', 'Noise Estimate']
            display_table(data, headers)

        elif choice == '7':
            print("\n📈 QUALITY SCORE DISTRIBUTION")
            print("=" * 80)
            data = query.get_quality_distribution(bins=10)
            headers = ['Quality Range', 'Count']
            display_table([(f"{start:.1f}-{start+0.1:.1f}", count) for start, count in data], headers)

        elif choice == '8':
            filename = input("Enter filename (or part of it): ").strip()
            if filename:
                result = query.get_detailed_metrics(filename=filename)
                if result:
                    print("\n🔍 IMAGE DETAILS")
                    print("=" * 80)
                    (img_id, fname, fpath, dataset, category, width, height,
                     quality, lap, contrast, noise, mscn, gradient, entropy,
                     tenengrad, proc_time, computed) = result

                    print(f"ID: {img_id}")
                    print(f"Filename: {fname}")
                    print(f"Path: {fpath}")
                    print(f"Dataset: {dataset} / {category}")
                    print(f"Dimensions: {width}x{height}")
                    print(f"\nQuality Score: {quality:.4f}")
                    print(f"\nDetailed Metrics:")
                    print(f"  • Laplacian Variance: {lap:.4f}")
                    print(f"  • RMS Contrast:       {contrast:.4f}")
                    print(f"  • Noise Estimate:     {noise:.4f}")
                    print(f"  • MSCN Std Dev:       {mscn:.4f}")
                    print(f"  • Gradient Energy:    {gradient:.4f}")
                    print(f"  • Entropy:            {entropy:.4f}")
                    print(f"  • Tenengrad:          {tenengrad:.4f}")
                    print(f"\nProcessing Time: {proc_time:.2f}ms")
                    print(f"Computed At: {computed}")
                else:
                    print("Image not found.")

        elif choice == '9':
            min_score = float(input("Minimum quality score (0-1): ").strip() or 0.0)
            max_score = float(input("Maximum quality score (0-1): ").strip() or 1.0)
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print(f"\n📊 IMAGES WITH QUALITY SCORE {min_score:.2f} - {max_score:.2f}")
            print("=" * 80)
            data = query.get_images_by_quality_range(min_score, max_score, limit)
            headers = ['Filename', 'Dataset', 'Category', 'Quality', 'Laplacian', 'Contrast', 'Noise', 'Entropy']
            display_table(data, headers)

        else:
            print("❌ Invalid option. Please try again.")

    print("\n✓ Database connection closed")
    query.close()


if __name__ == "__main__":
    main()

