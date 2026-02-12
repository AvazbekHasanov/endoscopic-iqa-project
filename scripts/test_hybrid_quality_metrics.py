#!/usr/bin/env python3
"""
Quick test script for the HYBRID quality metrics system.
Tests both traditional and deep learning methods on sample images.
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
from scripts.db_config import DB_CONFIG


def test_hybrid_quality_computation():
    """Test hybrid quality metrics computation on a few images"""

    print("=" * 80)
    print("🧪 HYBRID QUALITY METRICS TEST")
    print("   (Traditional + Deep Learning + Ensemble)")
    print("=" * 80)

    # Initialize Hybrid IQA predictor
    print("\n📦 Initializing hybrid predictor...")
    predictor = HybridIQAPredictor(
        dl_model_path=None,  # No pretrained model yet
        model_type='lightweight',
        device='auto'
    )

    info = predictor.get_info()
    print(f"✓ Device: {info['device']}")
    print(f"✓ DL Model Loaded: {info['dl_model_loaded']}")
    print(f"✓ Ensemble Weights: Traditional={info['ensemble_weights']['traditional']:.2f}, "
          f"Deep Learning={info['ensemble_weights']['deep_learning']:.2f}")

    if not info['dl_model_loaded']:
        print("\n⚠️  Note: Deep learning model is not pretrained yet.")
        print("   DL scores will be from an untrained model (random predictions).")
        print("   Traditional IQA will work perfectly and provide meaningful results!")

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"\n✓ Connected to database: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"\n✗ Database connection failed: {e}")
        print("\n💡 Tip: Check your db_config.py settings")
        return False

    cursor = conn.cursor()

    # Get a few sample images
    try:
        cursor.execute("""
            SELECT id, file_path, filename, dataset_name, category
            FROM image_metadata
            ORDER BY RANDOM()
            LIMIT 5
        """)
        samples = cursor.fetchall()

        if not samples:
            print("\n⚠️  No images found in database")
            print("\n💡 Tip: Run extract_image_metadata.py first")
            return False

        print(f"✓ Found {len(samples)} sample images to test\n")
    except Exception as e:
        print(f"\n✗ Error querying database: {e}")
        return False

    # Test each sample
    success_count = 0

    for idx, (image_id, file_path, filename, dataset, category) in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"{idx}. Testing: {filename}")
        print(f"   Dataset: {dataset} / {category}")
        print(f"{'='*80}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"   ✗ File not found: {file_path}")
            continue

        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                print(f"   ✗ Could not read image")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"   ✓ Loaded image: {image.shape}")

            # Compute hybrid metrics
            import time
            start = time.time()

            result = predictor.predict(image, method='ensemble', return_details=True)

            elapsed = (time.time() - start) * 1000

            print(f"   ✓ Computed metrics in {elapsed:.2f}ms")

            print(f"\n   🎯 QUALITY SCORES:")
            print(f"      Ensemble Score:      {result['ensemble_score']:.4f} ⭐")
            print(f"      Traditional Score:   {result['traditional_score']:.4f}")
            print(f"      Deep Learning Score: {result['deep_learning_score']:.4f}")

            print(f"\n   📊 TRADITIONAL METRICS:")
            metrics = result['traditional_metrics']
            print(f"      • laplacian_variance:  {metrics['laplacian_variance']:.4f}")
            print(f"      • rms_contrast:        {metrics['rms_contrast']:.4f}")
            print(f"      • noise_estimate:      {metrics['noise_estimate']:.4f}")
            print(f"      • mscn_std:            {metrics['mscn_std']:.4f}")
            print(f"      • gradient_energy:     {metrics['gradient_energy']:.4f}")
            print(f"      • entropy:             {metrics['entropy']:.4f}")
            print(f"      • tenengrad:           {metrics['tenengrad']:.4f}")

            # Quality assessment
            score = result['ensemble_score']
            if score >= 0.8:
                quality_label = "⭐⭐⭐⭐⭐ EXCELLENT"
            elif score >= 0.6:
                quality_label = "⭐⭐⭐⭐ GOOD"
            elif score >= 0.4:
                quality_label = "⭐⭐⭐ FAIR"
            elif score >= 0.2:
                quality_label = "⭐⭐ POOR"
            else:
                quality_label = "⭐ VERY POOR"

            print(f"\n   💡 ASSESSMENT: {quality_label}")

            success_count += 1

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    conn.close()

    print("\n" + "=" * 80)
    print(f"✅ Test complete: {success_count}/{len(samples)} images processed successfully")
    print("=" * 80)

    if success_count == len(samples):
        print("\n🎉 All tests passed! You're ready to process your dataset!")
        print("\n📝 Next steps:")
        print("   1. Review the results above")
        print("   2. For HYBRID (Traditional + Deep Learning):")
        print("      python3 compute_hybrid_quality_metrics.py")
        print("   3. Or for TRADITIONAL only:")
        print("      python3 compute_quality_metrics.py")
        print("   4. Start with limit=100 for testing, then limit=None for full dataset")
        print("\n💡 Note: Deep learning will use untrained model unless you train one first.")
        print("   Traditional IQA works perfectly and provides accurate quality assessment!")
        return True
    else:
        print(f"\n⚠️  Some tests failed ({len(samples) - success_count} errors)")
        print("   Check the error messages above and fix any issues")
        return False


if __name__ == "__main__":
    try:
        success = test_hybrid_quality_computation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

