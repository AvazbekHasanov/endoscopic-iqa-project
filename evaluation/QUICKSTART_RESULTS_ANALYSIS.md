# Results Analysis V2 - Quick Start Guide

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install psycopg2-binary python-docx pandas matplotlib seaborn scipy scikit-learn numpy
```

### 2. Run the Demo (No Database Required)
```bash
python3 evaluation/demo_results_analysis.py
```

This generates example visualizations with mock data in `/tmp/results_analysis_demo/`

### 3. Run Full Analysis (With Database)
```bash
python3 evaluation/results_analysis_v2.py
```

This connects to your PostgreSQL database and generates the complete analysis with Word document.

---

## 📊 What You Get

### Output Files

After running the full analysis, you'll find:

```
results_analysis_v2/
├── Comprehensive_Results_Analysis.docx  ⭐ Main Word document (all-in-one)
├── statistical_overview.csv
├── visual_examples.csv
├── disagreement_cases.csv
└── plots/
    ├── score_distributions.png
    ├── bland_altman_plot.png
    ├── scatter_matrix.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── ensemble_weights.png
    ├── quality_categories.png
    ├── performance_efficiency.png
    └── model_type_analysis.png
```

### Word Document Contents

The Word document includes:

1. **Statistical Overview** - Mean, median, std, min, max for all scores
2. **Score Distribution** - Violin and box plots
3. **Method Agreement** - Bland-Altman plot, scatter matrix, correlation heatmap
4. **Feature Importance** - Analysis of 7 traditional features
5. **Ensemble Weights** - Weight distribution and analysis
6. **Quality Categories** - Performance by quality level
7. **Performance Efficiency** - Processing time analysis
8. **Visual Examples** - Representative images from each category
9. **Disagreement Cases** - Where methods disagree
10. **Conclusion** - Summary of findings

---

## 🔧 Configuration

### Database Configuration

Edit `scripts/db_config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'your_username',
    'password': 'your_password'
}
```

### Custom Output Directory

```python
from evaluation.results_analysis_v2 import ResultsAnalyzer

analyzer = ResultsAnalyzer(db_config, output_dir='my_custom_results')
analyzer.run_complete_analysis()
```

---

## 📋 Prerequisites

### Database Tables Required

Your PostgreSQL database must have:

1. **image_quality_metrics_hybrid** table with columns:
   - `ensemble_score`, `traditional_score`, `deep_learning_score`
   - `laplacian_variance`, `rms_contrast`, `noise_estimate`, `mscn_std`
   - `gradient_energy`, `entropy`, `tenengrad`
   - `processing_time_ms`
   - `ensemble_weights_traditional`, `ensemble_weights_dl`
   - `model_type`

2. **image_metadata** table with columns:
   - `filename`, `file_path`, `dataset_name`, `category`
   - `width`, `height`

### Check if Data Exists

```bash
python3 scripts/verify_db.py
```

---

## 🎨 Visualizations Preview

### 1. Score Distributions
Shows the distribution of quality scores across all three methods (Ensemble, Traditional, Deep Learning) using violin and box plots.

### 2. Bland-Altman Plot
Assesses agreement between traditional and deep learning methods. Points should cluster around the mean difference line with most falling within ±1.96 SD.

### 3. Correlation Heatmap
Displays PLCC (Pearson) and SROCC (Spearman) correlations. Values closer to 1 indicate strong agreement.

### 4. Feature Importance
Shows which traditional features (e.g., sharpness, contrast, entropy) contribute most to quality assessment.

### 5. Ensemble Weights
Reveals how the system balances traditional and deep learning methods. Balanced weights indicate both methods contribute meaningfully.

---

## 🐛 Troubleshooting

### "Connection refused" Error
**Problem**: Cannot connect to PostgreSQL database  
**Solution**: 
- Check if PostgreSQL is running: `sudo systemctl status postgresql`
- Verify credentials in `scripts/db_config.py`
- Check firewall settings

### "No module named 'xxx'" Error
**Problem**: Missing Python package  
**Solution**: Install all dependencies:
```bash
pip install psycopg2-binary python-docx pandas matplotlib seaborn scipy scikit-learn numpy
```

### "No data found" Error
**Problem**: Database tables are empty  
**Solution**: 
- Run the hybrid IQA analysis first to populate the database
- Check table names match: `image_quality_metrics_hybrid`, `image_metadata`

### Empty or Missing Plots
**Problem**: Plots directory is empty  
**Solution**: 
- Check write permissions on output directory
- Verify matplotlib backend is configured
- Run demo script to test visualization generation

### Word Document Not Generated
**Problem**: Missing docx file  
**Solution**:
- Ensure all plots are generated first
- Check python-docx is installed: `pip install python-docx`
- Review console output for specific errors

---

## 💡 Tips & Best Practices

### Before Running Analysis

1. ✅ Verify database connection
2. ✅ Check data exists and is complete
3. ✅ Ensure enough disk space (~50 MB)
4. ✅ Close any open Word documents with same name

### Customizing the Analysis

Want to modify the analysis? Edit `results_analysis_v2.py`:

```python
# Change output image resolution
plt.savefig(path, dpi=600, bbox_inches='tight')  # Higher quality

# Modify color scheme
COLORS = {
    'traditional': '#your_color_hex',
    'deep_learning': '#your_color_hex',
    'ensemble': '#your_color_hex'
}

# Adjust quality categories
bins = [0, 0.25, 0.5, 0.75, 1.0]  # Custom bins
labels = ['Low', 'Medium', 'High', 'Excellent']
```

### Performance Tips

- **Large datasets (>10,000 images)**: Consider running analysis on a subset first
- **Memory issues**: Close other applications before running
- **Faster execution**: Use SSD storage for output directory

---

## 📚 Additional Resources

- **Full Documentation**: See `README_RESULTS_ANALYSIS.md`
- **Demo Script**: `evaluation/demo_results_analysis.py`
- **Test Script**: `evaluation/test_results_analysis.py`
- **Main Analysis**: `evaluation/results_analysis_v2.py`

---

## 🎯 Example Workflow

### Complete Analysis Workflow

```bash
# 1. Run demo to verify installation
python3 evaluation/demo_results_analysis.py

# 2. Check database connectivity
python3 scripts/verify_db.py

# 3. Run full analysis
python3 evaluation/results_analysis_v2.py

# 4. Open the Word document
# Open: results_analysis_v2/Comprehensive_Results_Analysis.docx

# 5. Review plots individually (optional)
# Open: results_analysis_v2/plots/*.png
```

### Custom Analysis in Python

```python
from evaluation.results_analysis_v2 import ResultsAnalyzer

# Setup
db_config = {...}  # Your database config
analyzer = ResultsAnalyzer(db_config, output_dir='my_analysis')

# Run specific analyses
analyzer.connect_db()
analyzer.fetch_data()

# Statistical overview
stats = analyzer.generate_statistical_overview()
print(stats)

# Visualizations
analyzer.plot_score_distributions()
analyzer.plot_correlation_heatmap()
analyzer.plot_feature_importance()

# Save results
# ... (generate Word doc)

analyzer.close_db()
```

---

## 📞 Support

For questions or issues:

1. Check this quick start guide
2. Review full documentation in `README_RESULTS_ANALYSIS.md`
3. Run the test script: `python3 evaluation/test_results_analysis.py`
4. Open a GitHub issue with:
   - Error message
   - Python version
   - Database version
   - Steps to reproduce

---

## 🎉 That's It!

You're ready to generate comprehensive results analysis for your IQA system!

**Happy analyzing! 📊🔬**
