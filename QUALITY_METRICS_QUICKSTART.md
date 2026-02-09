# Quick Start Guide: Image Quality Metrics

This guide will help you get started with computing and analyzing quality metrics for your endoscopic image dataset.

## Prerequisites

1. **PostgreSQL Database**: Ensure PostgreSQL is installed and running
2. **Python 3.7+**: With required packages installed
3. **Dataset**: Images already in the `data/datasets/` directory

## Step 1: Install Dependencies

```bash
cd /Users/hasanov_avazbek/Desktop/Projects/Study/endoscopic-iqa-project
pip3 install -r requirements.txt
```

Key packages needed:
- `psycopg2-binary` - PostgreSQL adapter
- `opencv-python` - Image processing
- `numpy`, `scipy` - Numerical computing
- `Pillow` - Image I/O
- `tabulate` - Table formatting

## Step 2: Configure Database

Edit `scripts/db_config.py` with your PostgreSQL credentials:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'your_username',      # Update this
    'password': 'your_password'   # Update this
}
```

## Step 3: Extract Image Metadata

Run the metadata extraction script to populate the database:

```bash
cd scripts
python3 extract_image_metadata.py
```

This will:
- Scan all images in `data/datasets/`
- Extract metadata (dimensions, format, size, etc.)
- Store in `image_metadata` table
- Display statistics

**Example Output:**
```
✓ Connected to PostgreSQL database: postgres
✓ Database table and indexes created/verified
🔍 Scanning directory: /path/to/datasets
📊 Processed 100 images...
📊 Processed 200 images...
...
✓ Total images processed: 17,239
```

## Step 4: Compute Quality Metrics (TEST MODE)

**IMPORTANT:** Start with a small test first!

Edit `scripts/compute_quality_metrics.py` line ~387:
```python
# Test with 100 images first
processor.process_all_images(batch_size=100, limit=100)
```

Then run:
```bash
python3 compute_quality_metrics.py
```

This will:
- Process 100 test images
- Compute 7 traditional quality metrics per image
- Calculate overall quality score
- Store in `image_quality_metrics` table
- Display progress and statistics

**Example Output:**
```
✓ Connected to PostgreSQL database: postgres
✓ Database table 'image_quality_metrics' and indexes created/verified
🔍 Found 100 images to process
📊 Processed 10/100 images (10%) - Quality Score: 0.645
📊 Processed 20/100 images (20%) - Quality Score: 0.732
...
✓ Total images processed: 100
```

## Step 5: Review Test Results

Run the query tool to explore results:

```bash
python3 query_quality_metrics.py
```

**Interactive Menu:**
```
📋 MENU:
  1. View dataset quality summary
  2. View category quality summary
  3. View high quality images (score > 0.7)
  4. View poor quality images (score < 0.3)
  5. View potentially blurry images
  6. View potentially noisy images
  7. View quality distribution
  8. Search image by filename
  9. View images by quality range
  0. Exit
```

Try options 1, 3, and 7 to get an overview of the quality metrics.

## Step 6: Process All Images (FULL MODE)

If test results look good, process all images:

Edit `scripts/compute_quality_metrics.py` line ~387:
```python
# Process ALL images
processor.process_all_images(batch_size=100, limit=None)
```

Then run:
```bash
python3 compute_quality_metrics.py
```

**Note:** This may take a while depending on:
- Number of images (~17,000 in this dataset)
- Image sizes
- CPU speed
- Estimated time: ~1-2 hours for full dataset

Processing speed: ~50-100 images/minute typically

## Step 7: Analyze Results

Use the query tool to explore full results:

```bash
python3 query_quality_metrics.py
```

### Useful Queries:

**1. Dataset Quality Summary (Option 1)**
```
Dataset Quality Summary:
╒════════════════╤═══════╤═══════════════╤═══════╤═══════╤══════════╕
│ Dataset        │ Count │ Avg Quality   │ Min   │ Max   │ Std Dev  │
╞════════════════╪═══════╪═══════════════╪═══════╪═══════╪══════════╡
│ Gastrovision   │  7999 │ 0.6234       │ 0.123 │ 0.891 │ 0.145    │
│ lower-gi-tract │  7210 │ 0.5987       │ 0.089 │ 0.876 │ 0.167    │
│ upper-gi-tract │  2030 │ 0.6112       │ 0.134 │ 0.899 │ 0.152    │
╘════════════════╧═══════╧═══════════════╧═══════╧═══════╧══════════╛
```

**2. High Quality Images (Option 3)**
Find images with score > 0.7 for potential use as reference images.

**3. Poor Quality Images (Option 4)**
Find images with score < 0.3 that may need exclusion or special handling.

**4. Blurry Images (Option 5)**
Identify out-of-focus images (low Laplacian variance).

**5. Quality Distribution (Option 7)**
Visualize the distribution of quality scores across your dataset.

## Understanding Quality Metrics

### Overall Quality Score (0-1 scale)
- **0.8-1.0**: Excellent quality - Sharp, clear, good contrast
- **0.6-0.8**: Good quality - Acceptable for most uses
- **0.4-0.6**: Fair quality - May have minor issues
- **0.2-0.4**: Poor quality - Significant issues (blur, noise, low contrast)
- **0.0-0.2**: Very poor quality - Should be excluded

### Individual Metrics:

**Laplacian Variance** (Blur Detection)
- Higher = Sharper
- Typical range: 0-1000
- < 100 = Likely blurry

**RMS Contrast**
- Higher = Better contrast
- Typical range: 0-1
- < 0.15 = Low contrast

**Noise Estimate**
- Lower = Cleaner
- Typical range: 0-50
- > 15 = Potentially noisy

**Gradient Energy** (Sharpness)
- Higher = Sharper
- Typical range: 0-10000

**Entropy** (Information Content)
- Higher = More detail
- Typical range: 0-8

**Tenengrad** (Focus)
- Higher = Better focus
- Typical range: 0-10000

**MSCN Std Dev**
- Natural scene statistics
- Typical range: 0-1

## Example Use Cases

### 1. Filter High-Quality Images for Training
```sql
SELECT file_path FROM image_quality_metrics
WHERE quality_score > 0.7
ORDER BY quality_score DESC;
```

### 2. Identify Images Needing Manual Review
```sql
SELECT m.file_path, q.quality_score, q.laplacian_variance, q.noise_estimate
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.quality_score < 0.4
ORDER BY q.quality_score ASC;
```

### 3. Compare Quality Across Datasets
```sql
SELECT m.dataset_name,
       ROUND(AVG(q.quality_score)::numeric, 4) as avg_quality
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
GROUP BY m.dataset_name;
```

### 4. Export Quality Report
From the query tool, use SQL or Python to export results:

```python
import pandas as pd
import psycopg2

conn = psycopg2.connect(...)
df = pd.read_sql("""
    SELECT m.filename, m.dataset_name, m.category,
           q.quality_score, q.laplacian_variance, q.rms_contrast
    FROM image_quality_metrics q
    JOIN image_metadata m ON q.image_id = m.id
""", conn)

df.to_csv('quality_report.csv', index=False)
```

## Troubleshooting

### Issue: "Could not read image"
- **Cause**: File path incorrect or file corrupted
- **Solution**: Check file exists and is a valid image

### Issue: Database connection failed
- **Cause**: PostgreSQL not running or wrong credentials
- **Solution**: 
  - Check PostgreSQL is running: `pg_isready`
  - Verify credentials in `db_config.py`
  - Test connection: `python3 verify_db.py`

### Issue: Processing is slow
- **Cause**: Large images or CPU constraints
- **Solution**: 
  - Process in batches with breaks
  - Consider processing overnight
  - Use `limit` parameter for selective processing

### Issue: Memory errors
- **Cause**: Too many large images in memory
- **Solution**:
  - Reduce `batch_size` in processing
  - Close other applications
  - Process in smaller chunks

## Next Steps

1. **Train Models**: Use quality scores to filter training data
2. **Data Augmentation**: Focus on high-quality images
3. **Quality Monitoring**: Track quality over time
4. **Custom Queries**: Build specific analyses for your needs
5. **Integration**: Incorporate into data pipelines

## Additional Resources

- Full documentation: `scripts/README.md`
- Database schema: See `scripts/README.md`
- SQL query examples: See `scripts/README.md`
- Traditional IQA code: `models/traditional/traditional_iqa.py`

## Summary of Commands

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Extract metadata
cd scripts
python3 extract_image_metadata.py

# 3. Test quality metrics (100 images)
# Edit compute_quality_metrics.py: limit=100
python3 compute_quality_metrics.py

# 4. Review results
python3 query_quality_metrics.py

# 5. Process all images
# Edit compute_quality_metrics.py: limit=None
python3 compute_quality_metrics.py

# 6. Analyze full results
python3 query_quality_metrics.py
```

Good luck with your image quality assessment! 🔬📊

