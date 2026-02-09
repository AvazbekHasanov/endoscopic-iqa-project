# Image Metadata and Quality Metrics Scripts

## Overview
These scripts extract and store metadata from all endoscopic images in the dataset into a PostgreSQL database, compute image quality metrics using traditional IQA methods, and provide tools for querying and analyzing the results.

## Files

### 1. `db_config.py`
Database configuration file. **Edit this file** with your PostgreSQL credentials:
```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'your_username',  # Update this
    'password': 'your_password'  # Update this if needed
}
```

### 2. `extract_image_metadata.py`
Main script that scans all images and extracts metadata including:
- File path
- Dataset name, category, and subcategory
- Filename
- Image dimensions (width, height)
- Image format and mode
- File size
- MD5 hash (for duplicate detection)
- Timestamps

**Usage:**
```bash
python3 scripts/extract_image_metadata.py
```

### 3. `compute_quality_metrics.py`
**NEW:** Computes traditional IQA (Image Quality Assessment) metrics for all images in the database.

**Features:**
- Processes all images from the metadata database
- Computes 7 traditional quality metrics:
  1. **Laplacian Variance** - Blur detection (higher = sharper)
  2. **RMS Contrast** - Contrast measurement (higher = better)
  3. **Noise Estimate** - Noise level (lower = cleaner)
  4. **MSCN Std Dev** - Natural scene statistics
  5. **Gradient Energy** - Sharpness measure (higher = sharper)
  6. **Entropy** - Information content (higher = more detail)
  7. **Tenengrad** - Focus measure (higher = better focus)
- Calculates overall quality score (0-1 scale, higher is better)
- Batch processing with progress tracking

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

**Usage:**
```bash
python3 scripts/compute_quality_metrics.py
```

**Configuration:**
Edit the script to adjust processing options:
- `batch_size=100`: Images per commit batch
- `limit=None`: Max images to process (None = all, or set to number for testing)

### 4. `query_quality_metrics.py`
**NEW:** Interactive tool to query and analyze quality metrics from the database.

**Features:**
- Dataset quality summaries
- Category-wise quality analysis
- Find high/low quality images
- Detect potentially blurry images (low Laplacian variance)
- Detect potentially noisy images (high noise estimate)
- Quality distribution histograms
- Search by filename
- Custom quality range queries
- Formatted table output

**Usage:**
```bash
python3 scripts/query_quality_metrics.py
```

**Menu Options:**
1. Dataset quality summary
2. Category quality summary
3. High quality images (score > 0.7)
4. Poor quality images (score < 0.3)
5. Potentially blurry images
6. Potentially noisy images
7. Quality distribution
8. Search by filename
9. Custom quality range

### 5. `query_image_metadata.py`
Query tool to explore the metadata database with various pre-built queries:
- Dataset statistics
- Category breakdowns
- Image format distribution
- Largest/smallest images
- Search by category
- Export to CSV

**Usage:**
```bash
python3 scripts/query_image_metadata.py
```

## Workflow

### Initial Setup:
1. **Configure database** - Edit `db_config.py` with your PostgreSQL credentials
2. **Extract metadata** - Run `extract_image_metadata.py`
3. **Compute quality metrics** - Run `compute_quality_metrics.py`
4. **Analyze results** - Use `query_quality_metrics.py`

### Step-by-Step Example:
```bash
# Step 1: Extract metadata
cd scripts
python3 extract_image_metadata.py

# Step 2: Test quality metrics on 100 images first
# Edit compute_quality_metrics.py: set limit=100
python3 compute_quality_metrics.py

# Step 3: Review test results
python3 query_quality_metrics.py

# Step 4: Process all images (if satisfied)
# Edit compute_quality_metrics.py: set limit=None
python3 compute_quality_metrics.py

# Step 5: Analyze full results
python3 query_quality_metrics.py
```

## Database Schema

### Table: `image_metadata`
Stores basic image information and metadata.

```sql
CREATE TABLE image_metadata (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    dataset_name TEXT,
    category TEXT,
    subcategory TEXT,
    filename TEXT,
    width INTEGER,
    height INTEGER,
    format TEXT,
    mode TEXT,
    size_bytes BIGINT,
    md5_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `image_quality_metrics`
Stores computed quality metrics for each image.

```sql
CREATE TABLE image_quality_metrics (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES image_metadata(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    
    -- Overall quality score (0-1, higher is better)
    quality_score REAL,
    
    -- Individual traditional metrics
    laplacian_variance REAL,       -- Blur detection (higher = sharper)
    rms_contrast REAL,              -- Contrast (higher = better contrast)
    noise_estimate REAL,            -- Noise level (lower = cleaner)
    mscn_std REAL,                  -- Natural scene statistics
    gradient_energy REAL,           -- Sharpness (higher = sharper)
    entropy REAL,                   -- Information content (higher = more detail)
    tenengrad REAL,                 -- Focus measure (higher = better focus)
    
    -- Processing metadata
    processing_time_ms REAL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(image_id)
);
```

### Relationships:
- `image_quality_metrics.image_id` → `image_metadata.id` (with CASCADE DELETE)

## Results Summary

Total images processed: **17,239**

### By Dataset:
- **Gastrovision**: 7,999 images
- **lower-gi-tract**: 7,210 images
- **upper-gi-tract**: 2,030 images

### Statistics:
- Average dimensions: 812x649 pixels
- Average file size: 267 KB
- Format: JPEG

## SQL Queries

### Image Metadata Queries:

#### Get all images from a specific category:
```sql
SELECT * FROM image_metadata 
WHERE category = 'polyps';
```

#### Find duplicates by MD5 hash:
```sql
SELECT md5_hash, COUNT(*) as count 
FROM image_metadata 
GROUP BY md5_hash 
HAVING COUNT(*) > 1;
```

#### Get images in size range:
```sql
SELECT file_path, width, height 
FROM image_metadata 
WHERE width BETWEEN 512 AND 1024 
  AND height BETWEEN 512 AND 1024;
```

#### Count images per dataset and category:
```sql
SELECT dataset_name, category, COUNT(*) as count 
FROM image_metadata 
GROUP BY dataset_name, category 
ORDER BY count DESC;
```

### Quality Metrics Queries:

#### Get high quality images:
```sql
SELECT m.filename, m.dataset_name, m.category, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.quality_score > 0.7
ORDER BY q.quality_score DESC;
```

#### Find blurry images (low Laplacian variance):
```sql
SELECT m.filename, m.dataset_name, q.laplacian_variance, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.laplacian_variance < 100
ORDER BY q.laplacian_variance ASC;
```

#### Get quality statistics by dataset:
```sql
SELECT m.dataset_name,
       COUNT(*) as count,
       ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality,
       ROUND(MIN(q.quality_score)::numeric, 4) as min_quality,
       ROUND(MAX(q.quality_score)::numeric, 4) as max_quality
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
GROUP BY m.dataset_name
ORDER BY avg_quality DESC;
```

#### Find images with specific quality issues:
```sql
-- Low contrast images
SELECT m.filename, q.rms_contrast, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.rms_contrast < 0.15
ORDER BY q.rms_contrast ASC;

-- Noisy images
SELECT m.filename, q.noise_estimate, q.quality_score
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.noise_estimate > 15
ORDER BY q.noise_estimate DESC;
```

#### Get detailed metrics for a specific image:
```sql
SELECT m.filename, m.dataset_name, m.category,
       q.quality_score,
       q.laplacian_variance,
       q.rms_contrast,
       q.noise_estimate,
       q.mscn_std,
       q.gradient_energy,
       q.entropy,
       q.tenengrad
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE m.filename = 'your_image.jpg';
```

## Requirements

- Python 3.7+
- PostgreSQL database
- Python packages (install via `pip install -r requirements.txt`):
  - `psycopg2-binary` - PostgreSQL adapter
  - `Pillow` - Image I/O
  - `opencv-python` - Image processing for quality metrics
  - `numpy` - Numerical computing
  - `scipy` - Scientific computing
  - `tabulate` - Table formatting for query tools

## Notes

- The script creates indexes on `dataset_name`, `category`, and `md5_hash` for faster queries
- Re-running the extraction script will update existing records (upsert behavior)
- The MD5 hash can be used to detect duplicate images across datasets

