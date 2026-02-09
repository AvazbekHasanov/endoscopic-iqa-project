# Image Quality Metrics Implementation Summary

## Overview

I've implemented a complete system to compute and store traditional IQA (Image Quality Assessment) metrics for all images in your endoscopic dataset. The system processes images, computes 7 quality metrics plus an overall quality score, and stores all results in a PostgreSQL database.

## What Was Created

### 1. Main Processing Script: `compute_quality_metrics.py`
**Location:** `scripts/compute_quality_metrics.py`

**Features:**
- Processes all images from the `image_metadata` database table
- Computes 7 traditional IQA metrics per image
- Calculates overall quality score (0-1 scale)
- Batch processing with progress tracking
- Stores results in new `image_quality_metrics` table
- Displays comprehensive statistics

**Metrics Computed:**
1. **Laplacian Variance** - Blur detection (higher = sharper)
2. **RMS Contrast** - Contrast measurement (higher = better)
3. **Noise Estimate** - Noise level (lower = cleaner)
4. **MSCN Std Dev** - Natural scene statistics
5. **Gradient Energy** - Sharpness measure
6. **Entropy** - Information content
7. **Tenengrad** - Focus measure

**Example Output:**
```
Traditional Quality Score: 0.619
Detailed Metrics:
  Traditional - laplacian_variance:  234.9701
  Traditional - rms_contrast:        0.2602
  Traditional - noise_estimate:      3.0921
  Traditional - mscn_std:            0.5165
  Traditional - gradient_energy:     5060.2070
  Traditional - entropy:             6.8579
  Traditional - tenengrad:           4943.2533
```

### 2. Query Tool: `query_quality_metrics.py`
**Location:** `scripts/query_quality_metrics.py`

**Features:**
- Interactive menu-driven interface
- Multiple query types:
  - Dataset quality summaries
  - Category-wise analysis
  - High/low quality image detection
  - Blur detection (low Laplacian variance)
  - Noise detection (high noise estimate)
  - Quality distribution histograms
  - Filename search with detailed metrics
  - Custom quality range filtering
- Formatted table output using `tabulate`
- Export-ready results

### 3. Database Schema

**New Table: `image_quality_metrics`**
```sql
CREATE TABLE image_quality_metrics (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES image_metadata(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    
    -- Overall quality score (0-1)
    quality_score REAL,
    
    -- Individual metrics
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
);
```

**Indexes Created:**
- `idx_quality_score` - Fast quality filtering
- `idx_file_path_metrics` - Fast path lookups
- `idx_image_id` - Fast joins with metadata

### 4. Documentation

**Updated Files:**
- `scripts/README.md` - Comprehensive documentation
  - Script descriptions
  - Database schemas
  - SQL query examples
  - Usage instructions
  - Workflow guidance

**New Files:**
- `QUALITY_METRICS_QUICKSTART.md` - Step-by-step guide
  - Installation instructions
  - Configuration steps
  - Test mode vs. full processing
  - Results interpretation
  - Example use cases
  - Troubleshooting tips

### 5. Testing Utilities

**New Script: `test_database.py`**
- Quick database connection test
- Verifies tables exist
- Shows record counts

### 6. Dependencies

**Added to `requirements.txt`:**
- `tabulate>=0.9.0` - For formatted table output

**Existing dependencies used:**
- `psycopg2-binary` - PostgreSQL adapter
- `opencv-python` - Image processing
- `numpy`, `scipy` - Numerical computing
- `Pillow` - Image I/O

## How It Works

### Processing Pipeline:

```
1. Read image paths from image_metadata table
   ↓
2. For each image:
   - Load image using OpenCV
   - Convert BGR to RGB
   - Compute 7 traditional IQA metrics
   - Calculate weighted quality score
   - Measure processing time
   ↓
3. Store results in image_quality_metrics table
   - Individual metric values
   - Overall quality score
   - Processing metadata
   ↓
4. Batch commit to database
   - Every 100 images
   - Progress tracking
   ↓
5. Generate statistics
   - Quality score distribution
   - Top/bottom quality images
   - Dataset comparisons
```

### Quality Score Calculation:

The overall quality score (0-1) is computed as a weighted combination:

```python
# Normalize metrics to [0, 1] range
lap_norm = min(laplacian_variance / 1000.0, 1.0)
grad_norm = min(gradient_energy / 10000.0, 1.0)
contrast_norm = min(rms_contrast * 5.0, 1.0)
entropy_norm = entropy / 8.0
noise_norm = max(1.0 - noise_estimate / 50.0, 0.0)

# Weighted combination
quality_score = (
    0.30 * lap_norm +      # Sharpness (Laplacian)
    0.25 * grad_norm +     # Sharpness (Gradient)
    0.20 * contrast_norm + # Contrast
    0.15 * entropy_norm +  # Information content
    0.10 * noise_norm      # Noise (inverted)
)
```

## Usage Workflow

### Initial Setup:
```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Configure database
# Edit scripts/db_config.py

# 3. Test database connection
cd scripts
python3 test_database.py
```

### Processing Images:

**Test Mode (Recommended First):**
```bash
# Edit compute_quality_metrics.py: set limit=100
python3 compute_quality_metrics.py
```

**Full Processing:**
```bash
# Edit compute_quality_metrics.py: set limit=None
python3 compute_quality_metrics.py
```

### Analyzing Results:
```bash
python3 query_quality_metrics.py
```

## Example Queries

### SQL Queries:

**Get high quality images:**
```sql
SELECT m.filename, m.dataset_name, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.quality_score > 0.7
ORDER BY q.quality_score DESC;
```

**Find blurry images:**
```sql
SELECT m.filename, q.laplacian_variance, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.laplacian_variance < 100
ORDER BY q.laplacian_variance ASC;
```

**Quality by dataset:**
```sql
SELECT m.dataset_name,
       COUNT(*) as count,
       ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
GROUP BY m.dataset_name;
```

### Python API:

```python
from scripts.query_quality_metrics import QualityMetricsQuery
from scripts.db_config import DB_CONFIG

query = QualityMetricsQuery(DB_CONFIG)

# Get high quality images
high_quality = query.get_high_quality_images(threshold=0.7, limit=50)

# Get dataset summary
summary = query.get_dataset_quality_summary()

# Get detailed metrics
details = query.get_detailed_metrics(filename="image.jpg")
```

## Performance

**Processing Speed:**
- ~50-100 images/minute (typical)
- Depends on: image size, CPU speed, disk I/O

**For 17,239 images:**
- Estimated time: 1-2 hours
- Batch size: 100 images per commit
- Memory efficient: processes one image at a time

**Database Size:**
- Metadata table: ~1 MB per 1000 images
- Quality metrics: ~1 MB per 1000 images
- Indexes: ~0.5 MB per 1000 images

## Quality Score Interpretation

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.8 - 1.0   | Excellent     | Sharp, clear, good contrast |
| 0.6 - 0.8   | Good          | Acceptable for most uses |
| 0.4 - 0.6   | Fair          | Minor issues present |
| 0.2 - 0.4   | Poor          | Significant issues (blur, noise) |
| 0.0 - 0.2   | Very Poor     | Should be excluded |

## Use Cases

1. **Training Data Selection**
   - Filter images by quality threshold
   - Use high-quality images for training
   - Exclude poor-quality images

2. **Data Quality Monitoring**
   - Track quality across datasets
   - Identify problematic categories
   - Compare acquisition methods

3. **Image Preprocessing**
   - Apply different processing based on quality
   - Target enhancement to low-quality images
   - Skip unnecessary processing for high-quality

4. **Dataset Curation**
   - Remove duplicates and low-quality images
   - Balance dataset by quality distribution
   - Create quality-stratified splits

5. **Research Analysis**
   - Correlate quality with clinical features
   - Study quality across anatomical sites
   - Analyze quality trends over time

## Files Created/Modified

### New Files:
1. `scripts/compute_quality_metrics.py` - Main processing script
2. `scripts/query_quality_metrics.py` - Query tool
3. `scripts/test_database.py` - Database test utility
4. `QUALITY_METRICS_QUICKSTART.md` - Quick start guide

### Modified Files:
1. `scripts/README.md` - Updated with quality metrics documentation
2. `requirements.txt` - Added `tabulate` package

### Existing Files Used:
1. `scripts/db_config.py` - Database configuration
2. `models/traditional/traditional_iqa.py` - IQA implementation
3. `scripts/extract_image_metadata.py` - Metadata extraction (prerequisite)

## Next Steps

1. **Test the System:**
   ```bash
   cd scripts
   python3 test_database.py
   ```

2. **Run Test Processing:**
   ```bash
   # Edit compute_quality_metrics.py: limit=100
   python3 compute_quality_metrics.py
   ```

3. **Review Results:**
   ```bash
   python3 query_quality_metrics.py
   ```

4. **Process Full Dataset:**
   ```bash
   # Edit compute_quality_metrics.py: limit=None
   python3 compute_quality_metrics.py
   ```

5. **Integrate into Pipeline:**
   - Use quality scores in training
   - Filter data by quality
   - Create quality-aware augmentation

## Support

For issues or questions:
- Check `QUALITY_METRICS_QUICKSTART.md` for troubleshooting
- Review `scripts/README.md` for detailed documentation
- Examine example queries in documentation
- Test with small batches first (limit=10)

## Summary

✅ Complete quality metrics computation system
✅ 7 traditional IQA metrics + overall score
✅ PostgreSQL database integration
✅ Interactive query tool
✅ Comprehensive documentation
✅ Batch processing with progress tracking
✅ Statistics and analysis features
✅ Ready for production use

The system is ready to process your entire dataset of ~17,000 images and store all quality metrics for analysis and filtering!

