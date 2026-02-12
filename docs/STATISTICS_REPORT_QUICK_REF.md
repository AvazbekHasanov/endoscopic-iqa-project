# Statistics Report - Quick Reference

## What Changed?

### Statistics Reduced: 9 → 5

**OLD (9 statistics)**:
```
count | mean | median | std | min | 25% | 50% | 75% | max
```

**NEW (5 statistics)**:
```
count | mean | std | min | max
```

**Why?**
- ✅ Less cognitive load
- ✅ Dashboard-friendly
- ✅ Focuses on essentials
- ✅ Median = 50% (redundant)
- ✅ Quartiles less useful with mean+std

## New Features

### 1. Executive Summary Dashboard 📊
```
✓ Dataset overview (total images, metrics)
✓ Quality classification (Good/Fair/Poor)
✓ Key findings summary
```

### 2. Correlation Analysis 🔗
```
✓ Heatmap showing metric relationships
✓ Identifies redundant metrics
✓ Helps optimize metric selection
```

### 3. Quality Classification 🎯
```
Good:  45% (high sharpness, low noise)
Fair:  40% (medium quality)
Poor:  15% (low quality)
```

## Quick Statistics Guide

| Statistic | Purpose | When to Use |
|-----------|---------|-------------|
| **Count** | Sample size | Validate data quality |
| **Mean** | Average | Central tendency |
| **Std** | Variability | Consistency check |
| **Min** | Lower bound | Worst case |
| **Max** | Upper bound | Best case |

## Report Structure

```
📄 Image_Quality_Report_Optimized.docx
├── 📊 Executive Summary Dashboard
│   ├── Dataset overview
│   ├── Quality distribution
│   └── Key findings
├── 🔗 Correlation Analysis
│   └── Metric relationship heatmap
└── 📋 Detailed Metrics (7 metrics)
    ├── 5-statistic table
    ├── Distribution histogram
    └── Boxplot
```

## How to Use

### Run the Report
```bash
python scripts/generate_statistics_report.py
```

### Output Location
```
reports/
├── Image_Quality_Report_Optimized.docx
├── laplacian_hist.png
├── laplacian_box.png
├── correlation_heatmap.png
└── ... (other metric visualizations)
```

## Metric Interpretation

### Sharpness (Higher is Better)
- **Laplacian**: Edge sharpness
- **Tenengrad**: Overall focus
- **Gradient Energy**: Edge intensity

### Quality (Context Dependent)
- **RMS Contrast**: Higher = More contrast
- **Entropy**: Higher = More information

### Degradation (Lower is Better)
- **Noise**: Lower = Cleaner
- **MSCN Std**: Natural image statistics

## Quality Classification Logic

```python
# Good Quality
if laplacian > (mean + 0.5*std) and noise < (mean - 0.3*std):
    return "Good"

# Poor Quality  
elif laplacian < (mean - 0.5*std) or noise > (mean + 0.5*std):
    return "Poor"

# Fair Quality
else:
    return "Fair"
```

## Best Practices

### For Dashboards
1. Use the 5 key statistics for overview cards
2. Show quality distribution as pie/bar chart
3. Display correlation matrix for metric selection
4. Track trends over time

### For Decision Making
1. Set quality thresholds (e.g., >70% Good)
2. Monitor Poor percentage (alert if >20%)
3. Use correlation to optimize metrics
4. Compare reports over time

### For Analysis
1. Check histograms for distribution shape
2. Use boxplots to find outliers
3. Correlation >0.7 = redundant metrics
4. Focus on metrics with low correlation

## Common Use Cases

### 1. Dataset Quality Check
```
Goal: Verify dataset quality before training
Focus: Quality distribution, Poor percentage
Action: Remove/fix Poor quality images
```

### 2. Metric Selection
```
Goal: Choose best metrics for your application
Focus: Correlation matrix
Action: Keep diverse (low correlation) metrics
```

### 3. Quality Monitoring
```
Goal: Track quality over time
Focus: Mean values, quality distribution
Action: Compare reports periodically
```

### 4. Threshold Setting
```
Goal: Define acceptance criteria
Focus: Mean, Std, Min, Max
Action: Set thresholds based on distribution
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Database connection error | Check `scripts/db_config.py` |
| Module not found | `pip install psycopg2-binary pandas matplotlib seaborn python-docx` |
| No data available | Run quality metrics computation first |
| Report generation slow | Reduce `histogram_bins` or dataset size |

## Testing

```bash
# Run tests
python tests/test_generate_statistics_report.py

# Expected output
✓ Import Test PASSED
✓ Class Structure Test PASSED
✓ Metrics Mapping Test PASSED
✓ Optimized Statistics Test PASSED
```

## Comparison: Old vs New

| Feature | Old | New |
|---------|-----|-----|
| Statistics per metric | 9 | 5 ✅ |
| Executive summary | ❌ | ✅ |
| Quality classification | ❌ | ✅ |
| Correlation analysis | ❌ | ✅ |
| Report filename | Image_Quality_Report.docx | Image_Quality_Report_Optimized.docx |
| Dashboard-ready | Partial | Full ✅ |

## Key Takeaways

1. **Simpler is Better**: 5 statistics are more actionable than 9
2. **Context Matters**: Summary dashboard provides context
3. **Relationships Matter**: Correlation analysis reveals redundancy
4. **Quality Classification**: Makes decisions easier
5. **Dashboard-Ready**: Perfect for visualization tools

---

**Quick Start**: `python scripts/generate_statistics_report.py`  
**Full Guide**: See `docs/STATISTICS_REPORT_GUIDE.md`  
**Tests**: `python tests/test_generate_statistics_report.py`
