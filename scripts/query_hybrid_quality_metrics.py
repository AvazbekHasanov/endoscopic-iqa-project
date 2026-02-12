"""
Query and analyze HYBRID image quality metrics from the database.
Provides queries for traditional, deep learning, and ensemble scores.
"""

import os
import sys
import psycopg2
from pathlib import Path
from tabulate import tabulate


class HybridQualityMetricsQuery:
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

    def get_images_by_ensemble_range(self, min_score=0.0, max_score=1.0, limit=20):
        """Get images within an ensemble score range"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.filename,
                m.dataset_name,
                m.category,
                q.ensemble_score,
                q.traditional_score,
                q.deep_learning_score
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            WHERE q.ensemble_score >= %s AND q.ensemble_score <= %s
            ORDER BY q.ensemble_score DESC
            LIMIT %s
        """, (min_score, max_score, limit))

        return cursor.fetchall()

    def get_method_comparison(self, limit=20):
        """Compare traditional vs deep learning scores"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                m.filename,
                m.dataset_name,
                q.traditional_score,
                q.deep_learning_score,
                q.ensemble_score,
                ABS(q.traditional_score - q.deep_learning_score) as score_diff
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY score_diff DESC
            LIMIT %s
        """, (limit,))

        return cursor.fetchall()

    def get_high_quality_images(self, threshold=0.7, method='ensemble', limit=20):
        """Get high quality images by specified method"""
        cursor = self.conn.cursor()

        score_col = {
            'ensemble': 'ensemble_score',
            'traditional': 'traditional_score',
            'deep_learning': 'deep_learning_score'
        }[method]

        cursor.execute(f"""
            SELECT 
                m.filename,
                m.dataset_name,
                m.category,
                q.ensemble_score,
                q.traditional_score,
                q.deep_learning_score
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            WHERE q.{score_col} > %s
            ORDER BY q.{score_col} DESC
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
                ROUND(AVG(q.ensemble_score)::numeric, 4) as avg_ensemble,
                ROUND(AVG(q.traditional_score)::numeric, 4) as avg_traditional,
                ROUND(AVG(q.deep_learning_score)::numeric, 4) as avg_deep_learning,
                ROUND(MIN(q.ensemble_score)::numeric, 4) as min_ensemble,
                ROUND(MAX(q.ensemble_score)::numeric, 4) as max_ensemble
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            GROUP BY m.dataset_name
            ORDER BY avg_ensemble DESC
        """)

        return cursor.fetchall()

    def get_score_correlation(self):
        """Get correlation between traditional and deep learning scores"""
        cursor = self.conn.cursor()

        cursor.execute("""
            SELECT 
                CORR(traditional_score, deep_learning_score) as correlation,
                COUNT(*) as total_images,
                AVG(ABS(traditional_score - deep_learning_score)) as avg_difference
            FROM image_quality_metrics_hybrid
        """)

        return cursor.fetchone()

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
                    q.ensemble_score,
                    q.traditional_score,
                    q.deep_learning_score,
                    q.laplacian_variance,
                    q.rms_contrast,
                    q.noise_estimate,
                    q.mscn_std,
                    q.gradient_energy,
                    q.entropy,
                    q.tenengrad,
                    q.processing_time_ms,
                    q.ensemble_weights_traditional,
                    q.ensemble_weights_dl,
                    q.computed_at
                FROM image_quality_metrics_hybrid q
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
                    q.ensemble_score,
                    q.traditional_score,
                    q.deep_learning_score,
                    q.laplacian_variance,
                    q.rms_contrast,
                    q.noise_estimate,
                    q.mscn_std,
                    q.gradient_energy,
                    q.entropy,
                    q.tenengrad,
                    q.processing_time_ms,
                    q.ensemble_weights_traditional,
                    q.ensemble_weights_dl,
                    q.computed_at
                FROM image_quality_metrics_hybrid q
                JOIN image_metadata m ON q.image_id = m.id
                WHERE m.filename LIKE %s
            """, (f'%{filename}%',))
        else:
            return None

        return cursor.fetchone()

    def get_quality_distribution(self, method='ensemble', bins=10):
        """Get distribution of quality scores"""
        cursor = self.conn.cursor()

        score_col = {
            'ensemble': 'ensemble_score',
            'traditional': 'traditional_score',
            'deep_learning': 'deep_learning_score'
        }[method]

        cursor.execute(f"""
            SELECT 
                ROUND(({score_col} * %s)::numeric, 0) / %s as bin_start,
                COUNT(*) as count
            FROM image_quality_metrics_hybrid
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
    print("🔍 HYBRID QUALITY METRICS QUERY TOOL")
    print("   (Traditional + Deep Learning + Ensemble)")
    print("=" * 80)

    query = HybridQualityMetricsQuery(db_config)

    while True:
        print("\n📋 MENU:")
        print("  1. View dataset quality summary (all methods)")
        print("  2. View high quality images by method")
        print("  3. Compare traditional vs deep learning scores")
        print("  4. View score correlation analysis")
        print("  5. View quality distribution")
        print("  6. Search image by filename")
        print("  7. View images by ensemble score range")
        print("  0. Exit")

        choice = input("\n👉 Select an option: ").strip()

        if choice == '0':
            break

        elif choice == '1':
            print("\n📊 DATASET QUALITY SUMMARY (HYBRID METHOD)")
            print("=" * 80)
            data = query.get_dataset_quality_summary()
            headers = ['Dataset', 'Count', 'Avg Ensemble', 'Avg Traditional',
                      'Avg Deep Learning', 'Min Ensemble', 'Max Ensemble']
            display_table(data, headers)

        elif choice == '2':
            method = input("Select method (ensemble/traditional/deep_learning, default: ensemble): ").strip() or 'ensemble'
            threshold = float(input("Minimum score (default 0.7): ").strip() or 0.7)
            limit = int(input("Number of images (default 20): ").strip() or 20)

            print(f"\n🏆 HIGH QUALITY IMAGES ({method.upper()} > {threshold})")
            print("=" * 80)
            data = query.get_high_quality_images(threshold=threshold, method=method, limit=limit)
            headers = ['Filename', 'Dataset', 'Category', 'Ensemble', 'Traditional', 'Deep Learning']
            display_table(data, headers)

        elif choice == '3':
            limit = int(input("Number of images to show (default 20): ").strip() or 20)
            print("\n🔄 TRADITIONAL vs DEEP LEARNING COMPARISON")
            print("   (Showing largest differences)")
            print("=" * 80)
            data = query.get_method_comparison(limit=limit)
            headers = ['Filename', 'Dataset', 'Traditional', 'Deep Learning', 'Ensemble', 'Difference']
            display_table(data, headers)

        elif choice == '4':
            print("\n📈 SCORE CORRELATION ANALYSIS")
            print("=" * 80)
            correlation, total, avg_diff = query.get_score_correlation()
            print(f"Total Images:                    {total}")
            print(f"Correlation (Trad vs DL):        {correlation:.4f}" if correlation else "Correlation: N/A")
            print(f"Average Score Difference:        {avg_diff:.4f}")
            print(f"\nInterpretation:")
            if correlation:
                if correlation > 0.7:
                    print("  ✓ Strong positive correlation - methods agree well")
                elif correlation > 0.4:
                    print("  → Moderate correlation - some agreement")
                else:
                    print("  ⚠ Weak correlation - methods disagree often")

        elif choice == '5':
            method = input("Select method (ensemble/traditional/deep_learning, default: ensemble): ").strip() or 'ensemble'
            print(f"\n📊 QUALITY SCORE DISTRIBUTION ({method.upper()})")
            print("=" * 80)
            data = query.get_quality_distribution(method=method, bins=10)
            headers = ['Quality Range', 'Count']
            display_table([(f"{start:.1f}-{start+0.1:.1f}", count) for start, count in data], headers)

        elif choice == '6':
            filename = input("Enter filename (or part of it): ").strip()
            if filename:
                result = query.get_detailed_metrics(filename=filename)
                if result:
                    print("\n🔍 IMAGE DETAILS (HYBRID METRICS)")
                    print("=" * 80)
                    (img_id, fname, fpath, dataset, category, width, height,
                     ensemble, trad, dl, lap, contrast, noise, mscn, gradient, entropy,
                     tenengrad, proc_time, weight_trad, weight_dl, computed) = result

                    print(f"ID: {img_id}")
                    print(f"Filename: {fname}")
                    print(f"Path: {fpath}")
                    print(f"Dataset: {dataset} / {category}")
                    print(f"Dimensions: {width}x{height}")

                    print(f"\n🎯 Quality Scores:")
                    print(f"  Ensemble Score:      {ensemble:.4f}")
                    print(f"  Traditional Score:   {trad:.4f}")
                    print(f"  Deep Learning Score: {dl:.4f}")

                    print(f"\n📊 Ensemble Weights:")
                    print(f"  Traditional:    {weight_trad:.2f}")
                    print(f"  Deep Learning:  {weight_dl:.2f}")

                    print(f"\n📏 Traditional Metrics:")
                    print(f"  • Laplacian Variance: {lap:.4f}")
                    print(f"  • RMS Contrast:       {contrast:.4f}")
                    print(f"  • Noise Estimate:     {noise:.4f}")
                    print(f"  • MSCN Std Dev:       {mscn:.4f}")
                    print(f"  • Gradient Energy:    {gradient:.4f}")
                    print(f"  • Entropy:            {entropy:.4f}")
                    print(f"  • Tenengrad:          {tenengrad:.4f}")

                    print(f"\n⏱️  Processing Time: {proc_time:.2f}ms")
                    print(f"📅 Computed At: {computed}")
                else:
                    print("Image not found.")

        elif choice == '7':
            min_score = float(input("Minimum ensemble score (0-1): ").strip() or 0.0)
            max_score = float(input("Maximum ensemble score (0-1): ").strip() or 1.0)
            limit = int(input("Number of images (default 20): ").strip() or 20)

            print(f"\n📊 IMAGES WITH ENSEMBLE SCORE {min_score:.2f} - {max_score:.2f}")
            print("=" * 80)
            data = query.get_images_by_ensemble_range(min_score, max_score, limit)
            headers = ['Filename', 'Dataset', 'Category', 'Ensemble', 'Traditional', 'Deep Learning']
            display_table(data, headers)

        else:
            print("❌ Invalid option. Please try again.")

    print("\n✓ Database connection closed")
    query.close()


if __name__ == "__main__":
    main()

