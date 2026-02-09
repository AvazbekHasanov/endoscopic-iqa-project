# 🚀 Quick Reference Card - Image Quality Metrics

## 📋 Commands Cheat Sheet

```bash
# Navigate to scripts directory
cd /Users/hasanov_avazbek/Desktop/Projects/Study/endoscopic-iqa-project/scripts

# 1. TEST: Verify everything works (5 sample images)
python3 test_quality_metrics.py

# 2. PROCESS: Compute metrics for all images
python3 compute_quality_metrics.py

# 3. QUERY: Interactive analysis tool
python3 query_quality_metrics.py

# 4. TEST DB: Verify database connection
python3 test_database.py
```

---

## 🎯 Processing Configuration

**Edit `compute_quality_metrics.py` line ~387:**

```python
# TEST MODE: Process 100 images
processor.process_all_images(batch_size=100, limit=100)

# FULL MODE: Process all images
processor.process_all_images(batch_size=100, limit=None)
```

---

## 📊 Metrics Computed

| Metric | What It Measures | Range | Better When |
|--------|------------------|-------|-------------|
| **quality_score** | Overall quality | 0-1 | Higher |
| **laplacian_variance** | Blur | 0-1000 | Higher |
| **rms_contrast** | Contrast | 0-1 | Higher |
| **noise_estimate** | Noise level | 0-50 | Lower |
| **gradient_energy** | Sharpness | 0-10000 | Higher |
| **entropy** | Detail | 0-8 | Higher |
| **tenengrad** | Focus | 0-10000 | Higher |
| **mscn_std** | Scene stats | 0-1 | - |

---

## 🎨 Quality Score Interpretation

```
0.8-1.0  ⭐⭐⭐⭐⭐  Excellent  → Use for training
0.6-0.8  ⭐⭐⭐⭐    Good       → Use for training
0.4-0.6  ⭐⭐⭐      Fair       → Review case-by-case
0.2-0.4  ⭐⭐        Poor       → Consider excluding
0.0-0.2  ⭐          Very Poor → Exclude from training
```

---

## 💻 SQL Quick Queries

```sql
-- High quality images
SELECT file_path, quality_score 
FROM image_quality_metrics 
WHERE quality_score > 0.7 
ORDER BY quality_score DESC;

-- Blurry images
SELECT m.filename, q.laplacian_variance 
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE q.laplacian_variance < 100;

-- Quality by dataset
SELECT m.dataset_name, AVG(q.quality_score) 
FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
GROUP BY m.dataset_name;

-- All metrics for an image
SELECT * FROM image_quality_metrics q
JOIN image_metadata m ON q.image_id = m.id
WHERE m.filename = 'your_image.jpg';
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Can't connect to DB** | Check `db_config.py`, verify PostgreSQL is running |
| **No images found** | Run `extract_image_metadata.py` first |
| **File not found** | Check file paths in metadata table |
| **Processing slow** | Normal! ~50-100 images/min, be patient |
| **Import errors** | Run from `scripts/` directory |

---

## 📁 Database Tables

```
image_metadata           →  Basic image info (17,239 rows)
image_quality_metrics    →  Quality scores & metrics
```

**Relationship:** `image_quality_metrics.image_id` → `image_metadata.id`

---

## 🔍 Query Tool Menu

```
1. Dataset quality summary       → Overview by dataset
2. Category quality summary      → Overview by category
3. High quality images           → Score > 0.7
4. Poor quality images           → Score < 0.3
5. Potentially blurry images     → Low Laplacian
6. Potentially noisy images      → High noise
7. Quality distribution          → Histogram
8. Search by filename            → Find specific image
9. Custom quality range          → Filter by score
0. Exit
```

---

## ⚡ Expected Performance

- **Processing speed**: 50-100 images/minute
- **For 17,239 images**: ~1-2 hours
- **Database size**: ~20-30 MB
- **Memory usage**: Low (processes 1 image at a time)

---

## 📚 Documentation Files

```
IMPLEMENTATION_COMPLETE.md         → This summary (START HERE!)
QUALITY_METRICS_QUICKSTART.md      → Detailed step-by-step guide
QUALITY_METRICS_IMPLEMENTATION.md  → Technical details
scripts/README.md                  → Scripts reference
```

---

## ✅ Workflow Checklist

- [ ] Install dependencies: `pip3 install -r requirements.txt`
- [ ] Configure database: Edit `scripts/db_config.py`
- [ ] Extract metadata: `python3 extract_image_metadata.py`
- [ ] Test quality metrics: `python3 test_quality_metrics.py`
- [ ] Process 100 images: Edit `limit=100`, run `compute_quality_metrics.py`
- [ ] Review test results: `python3 query_quality_metrics.py`
- [ ] Process all images: Edit `limit=None`, run `compute_quality_metrics.py`
- [ ] Analyze full results: `python3 query_quality_metrics.py`

---

## 🎯 Example Use Cases

**Training Data Selection:**
```sql
SELECT file_path FROM image_quality_metrics 
WHERE quality_score > 0.7;
```

**Problem Detection:**
```sql
SELECT file_path, quality_score 
FROM image_quality_metrics 
WHERE quality_score < 0.4 OR laplacian_variance < 100;
```

**Export for Analysis:**
```python
import pandas as pd
df = pd.read_sql("SELECT * FROM image_quality_metrics q JOIN image_metadata m ON q.image_id = m.id", conn)
df.to_csv('quality_report.csv')
```

---

## 💡 Pro Tips

1. **Always test first** with `limit=100` before processing all images
2. **Run test_quality_metrics.py** to verify setup before full processing
3. **Monitor progress** - prints update every 10 images
4. **Batch commits** happen every 100 images automatically
5. **Re-running is safe** - uses UPSERT to update existing records
6. **Processing can be interrupted** - Ctrl+C is safe, commits are batched

---

## 📞 Quick Help

```bash
# Stuck? Run these diagnostics:
cd scripts
python3 test_database.py          # Test DB connection
python3 test_quality_metrics.py   # Test 5 images

# Check database status:
psql -U your_username -d postgres -c "SELECT COUNT(*) FROM image_metadata;"
psql -U your_username -d postgres -c "SELECT COUNT(*) FROM image_quality_metrics;"
```

---

## 🎉 Success Indicators

✅ **test_quality_metrics.py** shows 5/5 images processed  
✅ **compute_quality_metrics.py** completes without errors  
✅ **query_quality_metrics.py** shows results  
✅ **Quality scores** are between 0 and 1  
✅ **Database queries** return in < 1 second  

---

**Need more help?** Read the full documentation:
- `QUALITY_METRICS_QUICKSTART.md`
- `QUALITY_METRICS_IMPLEMENTATION.md`

**Ready to start?** Run: `cd scripts && python3 test_quality_metrics.py`

🚀 **Good luck!**

