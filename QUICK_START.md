# Quick Start Guide - Image Metadata Database

## ✅ What Was Done

Successfully extracted and stored metadata for **17,239 endoscopic images** in your PostgreSQL database:
- **Gastrovision**: 7,999 images (27 categories)
- **lower-gi-tract**: 7,210 images (4 main categories)
- **upper-gi-tract**: 2,030 images (3 subcategories)

## 🚀 Quick Commands

### 1. Run the Extraction Script (if you add more images)
```bash
cd /Users/hasanov_avazbek/Desktop/Projects/Study/endoscopic-iqa-project
python3 scripts/extract_image_metadata.py
```

### 2. Query the Database
```bash
# Direct SQL query
psql -U hasanov_avazbek -d postgres

# Then run queries like:
SELECT COUNT(*) FROM image_metadata;
SELECT dataset_name, COUNT(*) FROM image_metadata GROUP BY dataset_name;
```

### 3. Use Python to Query
```python
import psycopg2
from scripts.db_config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# Get all polyp images
cursor.execute("""
    SELECT file_path, width, height 
    FROM image_metadata 
    WHERE category ILIKE '%polyp%'
""")

for path, width, height in cursor.fetchall():
    print(f"{path}: {width}x{height}")

conn.close()
```

### 4. Export to CSV
```python
import psycopg2
import csv
from scripts.db_config import DB_CONFIG

conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("SELECT * FROM image_metadata")
with open('images.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([desc[0] for desc in cursor.description])
    writer.writerows(cursor.fetchall())

conn.close()
```

## 📊 Useful Queries

### Get images by category
```sql
SELECT file_path, category, width, height 
FROM image_metadata 
WHERE category = 'Colon polyps';
```

### Find images in specific dimensions
```sql
SELECT file_path, width, height 
FROM image_metadata 
WHERE width BETWEEN 800 AND 1024 
  AND height BETWEEN 600 AND 768;
```

### Count by dataset
```sql
SELECT dataset_name, COUNT(*) as count, 
       AVG(width) as avg_width, 
       AVG(height) as avg_height
FROM image_metadata 
GROUP BY dataset_name;
```

### Get sample from each category
```sql
SELECT DISTINCT ON (category) 
    file_path, category, width, height
FROM image_metadata
ORDER BY category, RANDOM();
```

## 🔧 Scripts Available

All scripts are in `/scripts/` directory:

1. **`extract_image_metadata.py`** - Main extraction (already run)
2. **`query_image_metadata.py`** - Pre-built queries
3. **`check_data_quality.py`** - Data quality checks
4. **`verify_db.py`** - Simple verification
5. **`db_config.py`** - Database settings

## 💡 For PyTorch DataLoader

```python
import psycopg2
from PIL import Image
from torch.utils.data import Dataset

class EndoscopicDataset(Dataset):
    def __init__(self, db_config, category=None):
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT file_path, category FROM image_metadata WHERE category = %s",
                (category,)
            )
        else:
            cursor.execute("SELECT file_path, category FROM image_metadata")
        
        self.data = cursor.fetchall()
        conn.close()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, category = self.data[idx]
        image = Image.open(path)
        # Apply transforms here
        return image, category

# Usage
dataset = EndoscopicDataset(DB_CONFIG, category='Colon polyps')
```

## 📈 Statistics

- **Total Images**: 17,239
- **Average Size**: 812×649 pixels, 267 KB
- **Storage**: ~4.6 GB
- **Formats**: JPEG (RGB)
- **Categories**: 34 unique categories across datasets

## 🎯 Next Steps

1. Use the database to filter images for training
2. Create balanced datasets by sampling from categories
3. Export subsets for different experiments
4. Track which images are used in which models
5. Add quality scores after IQA model inference

---

**Database**: `postgres` @ `localhost:5432`  
**Table**: `image_metadata` (17,239 rows)  
**Last Updated**: February 10, 2026

