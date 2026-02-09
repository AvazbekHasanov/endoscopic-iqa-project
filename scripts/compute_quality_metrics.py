"""
Compute quality metrics for all images in the database and save results.
Processes images using traditional IQA methods and stores individual metrics
along with overall quality scores.
"""

import os
import sys
import psycopg2
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.traditional.traditional_iqa import TraditionalIQA


class QualityMetricsProcessor:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.iqa_model = TraditionalIQA()
        self.setup_database()

    def setup_database(self):
        """Create database connection and table for storing quality metrics"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'postgres'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', '')
            )
            print(f"✓ Connected to PostgreSQL database: {self.db_config.get('database')}")

            cursor = self.conn.cursor()

            # Create quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_quality_metrics (
                    id SERIAL PRIMARY KEY,
                    image_id INTEGER REFERENCES image_metadata(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    
                    -- Overall quality score
                    quality_score REAL,
                    
                    -- Individual traditional metrics
                    laplacian_variance REAL,
                    rms_contrast REAL,
                    noise_estimate REAL,
                    mscn_std REAL,
                    gradient_energy REAL,
                    entropy REAL,
                    tenengrad REAL,
                    
                    -- Metadata
                    processing_time_ms REAL,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(image_id)
                )
            ''')

            # Create indexes for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_quality_score ON image_quality_metrics(quality_score)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_path_metrics ON image_quality_metrics(file_path)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_id ON image_quality_metrics(image_id)
            ''')

            self.conn.commit()
            print("✓ Database table 'image_quality_metrics' and indexes created/verified")

        except Exception as e:
            print(f"✗ Error connecting to database: {str(e)}")
            sys.exit(1)

    def load_image(self, file_path):
        """Load image from file path"""
        try:
            # Read image
            image = cv2.imread(str(file_path))
            if image is None:
                print(f"✗ Could not read image: {file_path}")
                return None

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        except Exception as e:
            print(f"✗ Error loading image {file_path}: {str(e)}")
            return None

    def compute_metrics_for_image(self, file_path, image_id):
        """Compute all quality metrics for a single image"""
        import time

        # Load image
        image = self.load_image(file_path)
        if image is None:
            return None

        # Start timing
        start_time = time.time()

        try:
            # Compute all traditional metrics
            metrics = self.iqa_model.compute_all_metrics(image)

            # Compute overall quality score
            quality_score = self.iqa_model.compute_quality_score(image, method='combined')

            # End timing
            processing_time_ms = (time.time() - start_time) * 1000

            return {
                'image_id': image_id,
                'file_path': str(file_path),
                'quality_score': quality_score,
                'laplacian_variance': metrics['laplacian_variance'],
                'rms_contrast': metrics['rms_contrast'],
                'noise_estimate': metrics['noise_estimate'],
                'mscn_std': metrics['mscn_std'],
                'gradient_energy': metrics['gradient_energy'],
                'entropy': metrics['entropy'],
                'tenengrad': metrics['tenengrad'],
                'processing_time_ms': processing_time_ms
            }

        except Exception as e:
            print(f"✗ Error computing metrics for {file_path}: {str(e)}")
            return None

    def process_all_images(self, batch_size=100, limit=None):
        """Process all images in the database"""
        cursor = self.conn.cursor()

        # Get all images from metadata table
        if limit:
            cursor.execute("""
                SELECT id, file_path 
                FROM image_metadata 
                ORDER BY id
                LIMIT %s
            """, (limit,))
        else:
            cursor.execute("""
                SELECT id, file_path 
                FROM image_metadata 
                ORDER BY id
            """)

        images = cursor.fetchall()
        total_images = len(images)

        print(f"\n🔍 Found {total_images} images to process")
        print("=" * 80)

        processed = 0
        errors = 0
        skipped = 0

        for idx, (image_id, file_path) in enumerate(images, 1):
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                skipped += 1
                continue

            # Compute metrics
            result = self.compute_metrics_for_image(file_path, image_id)

            if result:
                try:
                    # Insert or update metrics in database
                    cursor.execute('''
                        INSERT INTO image_quality_metrics
                        (image_id, file_path, quality_score, laplacian_variance, 
                         rms_contrast, noise_estimate, mscn_std, gradient_energy, 
                         entropy, tenengrad, processing_time_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (image_id) DO UPDATE SET
                            quality_score = EXCLUDED.quality_score,
                            laplacian_variance = EXCLUDED.laplacian_variance,
                            rms_contrast = EXCLUDED.rms_contrast,
                            noise_estimate = EXCLUDED.noise_estimate,
                            mscn_std = EXCLUDED.mscn_std,
                            gradient_energy = EXCLUDED.gradient_energy,
                            entropy = EXCLUDED.entropy,
                            tenengrad = EXCLUDED.tenengrad,
                            processing_time_ms = EXCLUDED.processing_time_ms,
                            updated_at = CURRENT_TIMESTAMP
                    ''', (
                        result['image_id'],
                        result['file_path'],
                        result['quality_score'],
                        result['laplacian_variance'],
                        result['rms_contrast'],
                        result['noise_estimate'],
                        result['mscn_std'],
                        result['gradient_energy'],
                        result['entropy'],
                        result['tenengrad'],
                        result['processing_time_ms']
                    ))

                    processed += 1

                    # Print progress
                    if processed % 10 == 0:
                        print(f"📊 Processed {processed}/{total_images} images "
                              f"({processed*100//total_images}%) - "
                              f"Quality Score: {result['quality_score']:.3f}")

                    # Commit in batches
                    if processed % batch_size == 0:
                        self.conn.commit()
                        print(f"💾 Committed batch of {batch_size} images to database")

                except Exception as e:
                    errors += 1
                    print(f"✗ Error inserting metrics for {file_path}: {str(e)}")
            else:
                errors += 1

        # Final commit
        self.conn.commit()

        print("=" * 80)
        print(f"✓ Total images processed: {processed}")
        print(f"⚠️  Skipped (file not found): {skipped}")
        if errors > 0:
            print(f"✗ Errors encountered: {errors}")

        return processed, errors, skipped

    def get_statistics(self):
        """Get statistics about quality metrics"""
        cursor = self.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM image_quality_metrics")
        total = cursor.fetchone()[0]

        # Quality score statistics
        cursor.execute("""
            SELECT 
                MIN(quality_score) as min_score,
                MAX(quality_score) as max_score,
                AVG(quality_score) as avg_score,
                STDDEV(quality_score) as std_score
            FROM image_quality_metrics
        """)
        score_stats = cursor.fetchone()

        # Average metrics
        cursor.execute("""
            SELECT 
                AVG(laplacian_variance) as avg_laplacian,
                AVG(rms_contrast) as avg_contrast,
                AVG(noise_estimate) as avg_noise,
                AVG(entropy) as avg_entropy,
                AVG(gradient_energy) as avg_gradient,
                AVG(processing_time_ms) as avg_time
            FROM image_quality_metrics
        """)
        avg_metrics = cursor.fetchone()

        # Top 10 best quality images
        cursor.execute("""
            SELECT m.file_path, m.dataset_name, m.category, q.quality_score
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY q.quality_score DESC
            LIMIT 10
        """)
        top_quality = cursor.fetchall()

        # Bottom 10 worst quality images
        cursor.execute("""
            SELECT m.file_path, m.dataset_name, m.category, q.quality_score
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY q.quality_score ASC
            LIMIT 10
        """)
        bottom_quality = cursor.fetchall()

        # Quality distribution by dataset
        cursor.execute("""
            SELECT m.dataset_name, 
                   COUNT(*) as count,
                   AVG(q.quality_score) as avg_score,
                   MIN(q.quality_score) as min_score,
                   MAX(q.quality_score) as max_score
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            GROUP BY m.dataset_name
            ORDER BY avg_score DESC
        """)
        by_dataset = cursor.fetchall()

        return {
            'total': total,
            'score_stats': score_stats,
            'avg_metrics': avg_metrics,
            'top_quality': top_quality,
            'bottom_quality': bottom_quality,
            'by_dataset': by_dataset
        }

    def display_sample_results(self, num_samples=5):
        """Display sample results from the database"""
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
                q.mscn_std,
                q.gradient_energy,
                q.entropy,
                q.tenengrad,
                q.processing_time_ms
            FROM image_quality_metrics q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY RANDOM()
            LIMIT %s
        """, (num_samples,))

        samples = cursor.fetchall()

        print("\n📸 SAMPLE RESULTS")
        print("=" * 80)

        for idx, sample in enumerate(samples, 1):
            (filename, dataset, category, quality_score, lap_var, contrast,
             noise, mscn, gradient, entropy, tenengrad, proc_time) = sample

            print(f"\n{idx}. {filename}")
            print(f"   Dataset: {dataset} / {category}")
            print(f"   Traditional Quality Score: {quality_score:.3f}")
            print(f"   Detailed Metrics:")
            print(f"     Traditional - laplacian_variance:  {lap_var:.4f}")
            print(f"     Traditional - rms_contrast:        {contrast:.4f}")
            print(f"     Traditional - noise_estimate:      {noise:.4f}")
            print(f"     Traditional - mscn_std:            {mscn:.4f}")
            print(f"     Traditional - gradient_energy:     {gradient:.4f}")
            print(f"     Traditional - entropy:             {entropy:.4f}")
            print(f"     Traditional - tenengrad:           {tenengrad:.4f}")
            print(f"   Processing time: {proc_time:.2f}ms")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    # Import database configuration
    try:
        from db_config import DB_CONFIG
        db_config = DB_CONFIG
        print(f"📝 Using database: {db_config['database']} with user: {db_config['user']}")
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
    print("🔬 ENDOSCOPIC IMAGE QUALITY METRICS PROCESSOR")
    print("=" * 80)

    processor = QualityMetricsProcessor(db_config)

    print("\n📁 Starting image quality assessment...")

    # Process all images (set limit=100 for testing, None for all images)
    # Change limit=None to process all images
    processed, errors, skipped = processor.process_all_images(
        batch_size=100,
        limit=None  # Set to a number like 100 for testing
    )

    if processed > 0:
        print("\n📈 STATISTICS")
        print("=" * 80)
        stats = processor.get_statistics()

        print(f"\n📊 Total images with quality metrics: {stats['total']}")

        if stats['score_stats']:
            min_score, max_score, avg_score, std_score = stats['score_stats']
            print(f"\n🎯 Quality Score Statistics:")
            print(f"  • Minimum:     {min_score:.4f}")
            print(f"  • Maximum:     {max_score:.4f}")
            print(f"  • Average:     {avg_score:.4f}")
            print(f"  • Std Dev:     {std_score:.4f}" if std_score else "  • Std Dev:     N/A")

        if stats['avg_metrics']:
            (avg_lap, avg_contrast, avg_noise, avg_entropy,
             avg_gradient, avg_time) = stats['avg_metrics']
            print(f"\n📏 Average Metric Values:")
            print(f"  • Laplacian Variance:  {avg_lap:.4f}")
            print(f"  • RMS Contrast:        {avg_contrast:.4f}")
            print(f"  • Noise Estimate:      {avg_noise:.4f}")
            print(f"  • Entropy:             {avg_entropy:.4f}")
            print(f"  • Gradient Energy:     {avg_gradient:.4f}")
            print(f"  • Avg Processing Time: {avg_time:.2f}ms")

        if stats['by_dataset']:
            print(f"\n📦 Quality by Dataset:")
            for dataset, count, avg_score, min_score, max_score in stats['by_dataset']:
                print(f"  • {dataset}: {count} images")
                print(f"    Avg: {avg_score:.4f}, Min: {min_score:.4f}, Max: {max_score:.4f}")

        if stats['top_quality']:
            print(f"\n🏆 Top 10 Highest Quality Images:")
            for file_path, dataset, category, score in stats['top_quality']:
                filename = Path(file_path).name
                print(f"  • {filename} ({dataset}/{category}): {score:.4f}")

        if stats['bottom_quality']:
            print(f"\n⚠️  Bottom 10 Lowest Quality Images:")
            for file_path, dataset, category, score in stats['bottom_quality']:
                filename = Path(file_path).name
                print(f"  • {filename} ({dataset}/{category}): {score:.4f}")

        # Display sample results
        processor.display_sample_results(num_samples=5)

    print("\n" + "=" * 80)
    print("✅ QUALITY METRICS PROCESSING COMPLETE!")
    print("=" * 80)

    processor.close()


if __name__ == "__main__":
    main()

