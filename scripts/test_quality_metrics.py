#!/usr/bin/env python3
"""
Quick test script to compute quality metrics for a few sample images.
This helps verify everything is working before processing the full dataset.
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
from scripts.db_config import DB_CONFIG


def test_quality_computation():
    """Test quality metrics computation on a few images"""

    print("=" * 80)
    print("🧪 QUALITY METRICS TEST")
    print("=" * 80)

    # Initialize IQA model
    iqa = TraditionalIQA()
    print("✓ Initialized TraditionalIQA model")

    # Connect to database
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print(f"✓ Connected to database: {DB_CONFIG['database']}")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
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
            print("⚠️  No images found in database")
            print("\n💡 Tip: Run extract_image_metadata.py first")
            return False

        print(f"✓ Found {len(samples)} sample images to test\n")
    except Exception as e:
        print(f"✗ Error querying database: {e}")
        return False

    # Test each sample
    success_count = 0

    for idx, (image_id, file_path, filename, dataset, category) in enumerate(samples, 1):
        print(f"\n{idx}. Testing: {filename}")
        print(f"   Dataset: {dataset} / {category}")

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

            # Compute metrics
            import time
            start = time.time()

            metrics = iqa.compute_all_metrics(image)
            quality_score = iqa.compute_quality_score(image, method='combined')

            elapsed = (time.time() - start) * 1000

            print(f"   ✓ Computed metrics in {elapsed:.2f}ms")
            print(f"\n   📊 Results:")
            print(f"      Quality Score: {quality_score:.4f}")
            print(f"      Detailed Metrics:")
            print(f"        • laplacian_variance:  {metrics['laplacian_variance']:.4f}")
            print(f"        • rms_contrast:        {metrics['rms_contrast']:.4f}")
            print(f"        • noise_estimate:      {metrics['noise_estimate']:.4f}")
            print(f"        • mscn_std:            {metrics['mscn_std']:.4f}")
            print(f"        • gradient_energy:     {metrics['gradient_energy']:.4f}")
            print(f"        • entropy:             {metrics['entropy']:.4f}")
            print(f"        • tenengrad:           {metrics['tenengrad']:.4f}")

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
        print("\n🎉 All tests passed! You're ready to run compute_quality_metrics.py")
        print("\n📝 Next steps:")
        print("   1. Review the results above")
        print("   2. Run: python3 compute_quality_metrics.py")
        print("   3. Start with limit=100 for testing, then limit=None for full dataset")
        return True
    else:
        print(f"\n⚠️  Some tests failed ({len(samples) - success_count} errors)")
        print("   Check the error messages above and fix any issues")
        return False


if __name__ == "__main__":
    try:
        success = test_quality_computation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

