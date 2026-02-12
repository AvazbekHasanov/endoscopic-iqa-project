"""
Compute HYBRID quality metrics (Traditional + Deep Learning) for all images
in the database and save results.

Processes images using both traditional IQA methods and deep learning CNN,
then computes an ensemble score combining both approaches.
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

from models.hybrid_iqa import HybridIQAPredictor


class HybridQualityMetricsProcessor:
    def __init__(self, db_config, dl_model_path=None):
        self.db_config = db_config
        self.conn = None
        self.predictor = HybridIQAPredictor(
            dl_model_path=dl_model_path,
            model_type='lightweight',
            device='auto'
        )
        self.setup_database()

    def setup_database(self):
        """Create database connection and table for storing hybrid quality metrics"""
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

            # Create hybrid quality metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_quality_metrics_hybrid (
                    id SERIAL PRIMARY KEY,
                    image_id INTEGER REFERENCES image_metadata(id) ON DELETE CASCADE,
                    file_path TEXT NOT NULL,
                    
                    -- Ensemble score (combination of both methods)
                    ensemble_score REAL,
                    
                    -- Traditional IQA score
                    traditional_score REAL,
                    
                    -- Deep learning score
                    deep_learning_score REAL,
                    
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
                    model_type TEXT,
                    ensemble_weights_traditional REAL,
                    ensemble_weights_dl REAL,
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    UNIQUE(image_id)
                )
            ''')

            # Create indexes for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_ensemble_score ON image_quality_metrics_hybrid(ensemble_score)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_traditional_score ON image_quality_metrics_hybrid(traditional_score)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_dl_score ON image_quality_metrics_hybrid(deep_learning_score)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_file_path_hybrid ON image_quality_metrics_hybrid(file_path)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_id_hybrid ON image_quality_metrics_hybrid(image_id)
            ''')

            self.conn.commit()
            print("✓ Database table 'image_quality_metrics_hybrid' and indexes created/verified")

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
        """Compute all quality metrics (hybrid) for a single image"""
        import time

        # Load image
        image = self.load_image(file_path)
        if image is None:
            return None

        # Start timing
        start_time = time.time()

        try:
            # Compute hybrid metrics (traditional + deep learning + ensemble)
            result = self.predictor.predict(
                image,
                method='ensemble',
                return_details=True
            )

            # End timing
            processing_time_ms = (time.time() - start_time) * 1000

            # Get ensemble weights
            weights = self.predictor.ensemble_weights

            return {
                'image_id': image_id,
                'file_path': str(file_path),
                'ensemble_score': result['ensemble_score'],
                'traditional_score': result['traditional_score'],
                'deep_learning_score': result['deep_learning_score'],
                'laplacian_variance': result['traditional_metrics']['laplacian_variance'],
                'rms_contrast': result['traditional_metrics']['rms_contrast'],
                'noise_estimate': result['traditional_metrics']['noise_estimate'],
                'mscn_std': result['traditional_metrics']['mscn_std'],
                'gradient_energy': result['traditional_metrics']['gradient_energy'],
                'entropy': result['traditional_metrics']['entropy'],
                'tenengrad': result['traditional_metrics']['tenengrad'],
                'processing_time_ms': processing_time_ms,
                'model_type': 'lightweight',
                'ensemble_weights_traditional': weights['traditional'],
                'ensemble_weights_dl': weights['deep_learning']
            }

        except Exception as e:
            print(f"✗ Error computing metrics for {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
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

        # Display predictor info
        info = self.predictor.get_info()
        print(f"📊 Predictor Configuration:")
        print(f"   • Device: {info['device']}")
        print(f"   • DL Model Loaded: {info['dl_model_loaded']}")
        print(f"   • Traditional Weight: {info['ensemble_weights']['traditional']:.2f}")
        print(f"   • Deep Learning Weight: {info['ensemble_weights']['deep_learning']:.2f}")
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
                        INSERT INTO image_quality_metrics_hybrid
                        (image_id, file_path, ensemble_score, traditional_score, deep_learning_score,
                         laplacian_variance, rms_contrast, noise_estimate, mscn_std, 
                         gradient_energy, entropy, tenengrad, processing_time_ms,
                         model_type, ensemble_weights_traditional, ensemble_weights_dl)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (image_id) DO UPDATE SET
                            ensemble_score = EXCLUDED.ensemble_score,
                            traditional_score = EXCLUDED.traditional_score,
                            deep_learning_score = EXCLUDED.deep_learning_score,
                            laplacian_variance = EXCLUDED.laplacian_variance,
                            rms_contrast = EXCLUDED.rms_contrast,
                            noise_estimate = EXCLUDED.noise_estimate,
                            mscn_std = EXCLUDED.mscn_std,
                            gradient_energy = EXCLUDED.gradient_energy,
                            entropy = EXCLUDED.entropy,
                            tenengrad = EXCLUDED.tenengrad,
                            processing_time_ms = EXCLUDED.processing_time_ms,
                            model_type = EXCLUDED.model_type,
                            ensemble_weights_traditional = EXCLUDED.ensemble_weights_traditional,
                            ensemble_weights_dl = EXCLUDED.ensemble_weights_dl,
                            updated_at = CURRENT_TIMESTAMP
                    ''', (
                        result['image_id'],
                        result['file_path'],
                        result['ensemble_score'],
                        result['traditional_score'],
                        result['deep_learning_score'],
                        result['laplacian_variance'],
                        result['rms_contrast'],
                        result['noise_estimate'],
                        result['mscn_std'],
                        result['gradient_energy'],
                        result['entropy'],
                        result['tenengrad'],
                        result['processing_time_ms'],
                        result['model_type'],
                        result['ensemble_weights_traditional'],
                        result['ensemble_weights_dl']
                    ))

                    processed += 1

                    # Print progress
                    if processed % 10 == 0:
                        print(f"📊 Processed {processed}/{total_images} images "
                              f"({processed*100//total_images}%) - "
                              f"Ensemble: {result['ensemble_score']:.3f} "
                              f"(Trad: {result['traditional_score']:.3f}, "
                              f"DL: {result['deep_learning_score']:.3f})")

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
        """Get statistics about hybrid quality metrics"""
        cursor = self.conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM image_quality_metrics_hybrid")
        total = cursor.fetchone()[0]

        # Score statistics (all three types)
        cursor.execute("""
            SELECT 
                MIN(ensemble_score) as min_ensemble,
                MAX(ensemble_score) as max_ensemble,
                AVG(ensemble_score) as avg_ensemble,
                STDDEV(ensemble_score) as std_ensemble,
                MIN(traditional_score) as min_trad,
                MAX(traditional_score) as max_trad,
                AVG(traditional_score) as avg_trad,
                MIN(deep_learning_score) as min_dl,
                MAX(deep_learning_score) as max_dl,
                AVG(deep_learning_score) as avg_dl
            FROM image_quality_metrics_hybrid
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
            FROM image_quality_metrics_hybrid
        """)
        avg_metrics = cursor.fetchone()

        # Top 10 by ensemble score
        cursor.execute("""
            SELECT m.file_path, m.dataset_name, m.category, 
                   q.ensemble_score, q.traditional_score, q.deep_learning_score
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY q.ensemble_score DESC
            LIMIT 10
        """)
        top_quality = cursor.fetchall()

        # Bottom 10 by ensemble score
        cursor.execute("""
            SELECT m.file_path, m.dataset_name, m.category,
                   q.ensemble_score, q.traditional_score, q.deep_learning_score
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY q.ensemble_score ASC
            LIMIT 10
        """)
        bottom_quality = cursor.fetchall()

        # Quality distribution by dataset
        cursor.execute("""
            SELECT m.dataset_name, 
                   COUNT(*) as count,
                   AVG(q.ensemble_score) as avg_ensemble,
                   AVG(q.traditional_score) as avg_trad,
                   AVG(q.deep_learning_score) as avg_dl,
                   MIN(q.ensemble_score) as min_ensemble,
                   MAX(q.ensemble_score) as max_ensemble
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            GROUP BY m.dataset_name
            ORDER BY avg_ensemble DESC
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
                q.processing_time_ms
            FROM image_quality_metrics_hybrid q
            JOIN image_metadata m ON q.image_id = m.id
            ORDER BY RANDOM()
            LIMIT %s
        """, (num_samples,))

        samples = cursor.fetchall()

        print("\n📸 SAMPLE RESULTS (HYBRID METHOD)")
        print("=" * 80)

        for idx, sample in enumerate(samples, 1):
            (filename, dataset, category, ensemble, trad, dl, lap_var, contrast,
             noise, mscn, gradient, entropy, tenengrad, proc_time) = sample

            print(f"\n{idx}. {filename}")
            print(f"   Dataset: {dataset} / {category}")
            print(f"   🎯 Ensemble Score: {ensemble:.3f}")
            print(f"   Traditional Score: {trad:.3f}")
            print(f"   Deep Learning Score: {dl:.3f}")
            print(f"   Detailed Metrics:")
            print(f"     • laplacian_variance:  {lap_var:.4f}")
            print(f"     • rms_contrast:        {contrast:.4f}")
            print(f"     • noise_estimate:      {noise:.4f}")
            print(f"     • mscn_std:            {mscn:.4f}")
            print(f"     • gradient_energy:     {gradient:.4f}")
            print(f"     • entropy:             {entropy:.4f}")
            print(f"     • tenengrad:           {tenengrad:.4f}")
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
    print("🔬 HYBRID IMAGE QUALITY METRICS PROCESSOR")
    print("   (Traditional + Deep Learning)")
    print("=" * 80)

    # Optional: Specify pretrained model path
    dl_model_path = None  # Set to path if you have a pretrained model

    processor = HybridQualityMetricsProcessor(db_config, dl_model_path=dl_model_path)

    print("\n📁 Starting hybrid image quality assessment...")

    # Process all images (set limit=100 for testing, None for all images)
    processed, errors, skipped = processor.process_all_images(
        batch_size=100,
        limit=None  # Set to a number like 10 for testing
    )

    if processed > 0:
        print("\n📈 STATISTICS")
        print("=" * 80)
        stats = processor.get_statistics()

        print(f"\n📊 Total images with hybrid quality metrics: {stats['total']}")

        if stats['score_stats']:
            (min_ens, max_ens, avg_ens, std_ens,
             min_trad, max_trad, avg_trad,
             min_dl, max_dl, avg_dl) = stats['score_stats']

            print(f"\n🎯 Ensemble Score Statistics:")
            print(f"  • Minimum:     {min_ens:.4f}")
            print(f"  • Maximum:     {max_ens:.4f}")
            print(f"  • Average:     {avg_ens:.4f}")
            print(f"  • Std Dev:     {std_ens:.4f}" if std_ens else "  • Std Dev:     N/A")

            print(f"\n📊 Traditional Score Statistics:")
            print(f"  • Min: {min_trad:.4f}, Max: {max_trad:.4f}, Avg: {avg_trad:.4f}")

            print(f"\n🤖 Deep Learning Score Statistics:")
            print(f"  • Min: {min_dl:.4f}, Max: {max_dl:.4f}, Avg: {avg_dl:.4f}")

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
            print(f"\n📦 Quality by Dataset (Hybrid Method):")
            for dataset, count, avg_ens, avg_trad, avg_dl, min_ens, max_ens in stats['by_dataset']:
                print(f"  • {dataset}: {count} images")
                print(f"    Ensemble: {avg_ens:.4f} (Min: {min_ens:.4f}, Max: {max_ens:.4f})")
                print(f"    Traditional: {avg_trad:.4f}, Deep Learning: {avg_dl:.4f}")

        if stats['top_quality']:
            print(f"\n🏆 Top 10 Highest Quality Images (by Ensemble Score):")
            for file_path, dataset, category, ens, trad, dl in stats['top_quality']:
                filename = Path(file_path).name
                print(f"  • {filename} ({dataset}/{category})")
                print(f"    Ensemble: {ens:.4f} (Trad: {trad:.4f}, DL: {dl:.4f})")

        if stats['bottom_quality']:
            print(f"\n⚠️  Bottom 10 Lowest Quality Images (by Ensemble Score):")
            for file_path, dataset, category, ens, trad, dl in stats['bottom_quality']:
                filename = Path(file_path).name
                print(f"  • {filename} ({dataset}/{category})")
                print(f"    Ensemble: {ens:.4f} (Trad: {trad:.4f}, DL: {dl:.4f})")

        # Display sample results
        processor.display_sample_results(num_samples=5)

    print("\n" + "=" * 80)
    print("✅ HYBRID QUALITY METRICS PROCESSING COMPLETE!")
    print("=" * 80)

    processor.close()


if __name__ == "__main__":
    main()

