# Image Metadata Extraction - Summary

## ✅ Completed Tasks

### 1. Database Setup
- Created PostgreSQL database schema
- Set up `image_metadata` table with proper indexes
- Configured database connection via `db_config.py`

### 2. Metadata Extraction
Successfully processed **17,239 images** from three datasets:

#### Dataset Breakdown:
- **Gastrovision**: 7,999 images
  - 27 categories including polyps, cancer, inflammation, anatomical landmarks
- **lower-gi-tract**: 7,210 images
  - 4 main categories: anatomical landmarks, pathological findings, quality views, therapeutic interventions
- **upper-gi-tract**: 2,030 images
  - 3 subcategories: pylorus, retroflex-stomach, z-line

### 3. Extracted Metadata
For each image, the following information is stored:
- ✓ File path (absolute)
- ✓ Dataset name
- ✓ Category and subcategory
- ✓ Filename
- ✓ Dimensions (width, height)
- ✓ Image format (JPEG)
- ✓ Color mode (RGB)
- ✓ File size (bytes)
- ✓ MD5 hash (for duplicate detection)
- ✓ Timestamps (created, updated)

### 4. Image Statistics
- **Average dimensions**: 812 × 649 pixels
- **Average file size**: 267 KB
- **Format**: All JPEG
- **Total storage**: ~4.6 GB of images

## 📁 Created Files

### Scripts (`/scripts/`)
1. **`db_config.py`** - Database configuration
2. **`extract_image_metadata.py`** - Main extraction script
3. **`query_image_metadata.py`** - Query tool with pre-built queries
4. **`check_data_quality.py`** - Data quality analysis tool
5. **`verify_db.py`** - Simple database verification
6. **`README.md`** - Documentation

## 🗄️ Database Schema

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

-- Indexes for performance
CREATE INDEX idx_dataset_name ON image_metadata(dataset_name);
CREATE INDEX idx_category ON image_metadata(category);
CREATE INDEX idx_md5_hash ON image_metadata(md5_hash);
```

## 🔍 Usage Examples

### Query all images from a dataset:
```sql
SELECT * FROM image_metadata 
WHERE dataset_name = 'Gastrovision';
```

### Find polyp images:
```sql
SELECT file_path, category, width, height 
FROM image_metadata 
WHERE category ILIKE '%polyp%';
```

### Get statistics by category:
```sql
SELECT category, 
       COUNT(*) as count,
       AVG(width) as avg_width,
       AVG(height) as avg_height,
       AVG(size_bytes)/1024 as avg_size_kb
FROM image_metadata
GROUP BY category
ORDER BY count DESC;
```

### Find images in specific size range:
```sql
SELECT file_path, width, height
FROM image_metadata
WHERE width BETWEEN 512 AND 1024
  AND height BETWEEN 512 AND 1024;
```

### Detect duplicates:
```sql
SELECT md5_hash, array_agg(file_path) as files, COUNT(*) 
FROM image_metadata 
GROUP BY md5_hash 
HAVING COUNT(*) > 1;
```

## 📊 Category Breakdown (Top 10)

| Category | Count |
|----------|-------|
| Normal mucosa and vascular pattern in the large bowel | 1,467 |
| Accessory tools | 1,266 |
| Normal stomach | 969 |
| Small bowel terminal ileum | 846 |
| Colon polyps | 820 |
| Pylorus | 393 |
| Gastroesophageal junction normal z-line | 330 |
| Dyed-resection-margins | 246 |
| Duodenal bulb | 205 |
| Ileocecal valve | 200 |

## 🚀 Next Steps

You can now:
1. **Query the database** for specific image sets
2. **Filter by category** for model training
3. **Analyze image quality** metrics
4. **Balance datasets** for ML training
5. **Export subsets** to CSV for further analysis
6. **Track duplicates** across datasets
7. **Generate reports** on dataset composition

## 💡 Pro Tips

1. Use the MD5 hash to find duplicate images across datasets
2. Filter by image dimensions for consistent training batches
3. Use the file_path for direct image loading in PyTorch/TensorFlow
4. Export to CSV for pandas-based analysis
5. Create views for commonly used queries

## 🔧 Maintenance

To re-run the extraction (e.g., after adding new images):
```bash
cd /Users/hasanov_avazbek/Desktop/Projects/Study/endoscopic-iqa-project
python3 scripts/extract_image_metadata.py
```

The script uses `ON CONFLICT DO UPDATE` so it will:
- Add new images
- Update changed images
- Keep existing images unchanged

## 📦 Dependencies

Already installed:
- `psycopg2-binary==2.9.11`
- `Pillow>=10.0.0`
- `tabulate==0.9.0`

## ✅ Success Metrics

- [x] All 17,239 images processed
- [x] 0 errors during extraction
- [x] Metadata stored in PostgreSQL
- [x] Indexed for fast queries
- [x] Query tools provided
- [x] Documentation complete

