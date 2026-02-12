# Comprehensive Results Analysis - Version 2

## Overview

The `results_analysis_v2.py` script generates comprehensive statistical analysis and visualizations for the endoscopic Image Quality Assessment (IQA) system, and exports all results to a Word document.

## Features

This script performs the following analyses:

### 1. **Statistical Overview Table**
- Descriptive statistics (mean, median, std, min, max) for:
  - Ensemble scores
  - Traditional scores
  - Deep learning scores
  - Processing times

### 2. **Score Distribution Analysis**
- Violin plots showing distribution shapes
- Box plots showing quartiles and outliers
- Side-by-side comparison of all three methods

### 3. **Method Agreement & Correlation**
- **Bland-Altman plot**: Traditional vs Deep Learning scores (bias and limits of agreement)
- **Scatter plot matrix**: All three scores with regression lines and R² values
- **Correlation heatmap**: PLCC and SROCC values between methods

### 4. **Traditional Feature Importance Analysis**
Analyzes the 7 traditional features:
- Laplacian variance
- RMS contrast
- Noise estimate
- MSCN std
- Gradient energy
- Entropy
- Tenengrad

Visualizations:
- Heatmap: Correlation between features and ensemble score
- Radar chart: Normalized mean values of all features
- Feature contribution plot: Which features correlate strongest with quality

### 5. **Ensemble Weight Analysis**
- Histogram: Distribution of traditional and deep learning weights
- Scatter plot: Weight relationships
- Pie chart: Average ensemble weights
- Analysis: Percentage of images favoring each method

### 6. **Quality Categories Performance**
Quality bins: Poor (0-0.3), Fair (0.3-0.6), Good (0.6-0.8), Excellent (0.8-1.0)
- Stacked bar chart: Distribution across categories for all methods
- Pie chart: Ensemble score distribution by category

### 7. **Performance Efficiency**
- Box plot: Processing time distribution
- Scatter plot: Ensemble score vs processing time (quality-speed tradeoff)
- Average processing time statistics

### 8. **Qualitative Visual Examples**
Table of representative images:
- Examples from each quality category
- Shows ensemble, traditional, and deep learning scores
- Dataset and filename information

### 9. **Method Disagreement Cases**
Identifies images where traditional and deep learning methods disagree significantly:
- Top 10 cases with highest score differences
- Useful for understanding method limitations

### 10. **Model Type Analysis** (if applicable)
- Comparison of performance across different deep learning model architectures
- Average scores grouped by model type

## Output

The script generates:

1. **Word Document**: `Comprehensive_Results_Analysis.docx`
   - Complete report with all analyses, visualizations, and interpretations
   - Professional formatting with tables, charts, and captions
   - Ready for presentation or publication

2. **Plots Directory**: `plots/`
   - High-resolution PNG images (300 DPI)
   - Individual plot files for each analysis
   - Color-blind friendly palettes

3. **CSV Files**:
   - `statistical_overview.csv`: Statistical summary table
   - `visual_examples.csv`: Representative examples from each category
   - `disagreement_cases.csv`: Cases where methods disagree

## Requirements

### Python Dependencies
```bash
pip install psycopg2-binary python-docx pandas matplotlib seaborn scipy scikit-learn numpy
```

### Database Requirements
- PostgreSQL database with the following tables:
  - `image_quality_metrics_hybrid`: Contains quality metrics
  - `image_metadata`: Contains image information

### Database Schema
The script expects the following columns:

**image_quality_metrics_hybrid table:**
- `id`, `image_id`
- `ensemble_score`, `traditional_score`, `deep_learning_score`
- `laplacian_variance`, `rms_contrast`, `noise_estimate`, `mscn_std`
- `gradient_energy`, `entropy`, `tenengrad`
- `processing_time_ms`
- `ensemble_weights_traditional`, `ensemble_weights_dl`
- `model_type`, `computed_at`

**image_metadata table:**
- `id`, `filename`, `file_path`
- `dataset_name`, `category`
- `width`, `height`

## Usage

### Basic Usage

```bash
python3 evaluation/results_analysis_v2.py
```

### Advanced Usage (from Python)

```python
from evaluation.results_analysis_v2 import ResultsAnalyzer

# Configure database
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'your_username',
    'password': 'your_password'
}

# Create analyzer
analyzer = ResultsAnalyzer(db_config, output_dir='my_results')

# Run complete analysis
analyzer.run_complete_analysis()
```

### Custom Analysis

```python
from evaluation.results_analysis_v2 import ResultsAnalyzer

# Initialize
analyzer = ResultsAnalyzer(db_config, output_dir='custom_analysis')

# Connect and fetch data
analyzer.connect_db()
analyzer.fetch_data()

# Run specific analyses
stats_df = analyzer.generate_statistical_overview()
analyzer.plot_score_distributions()
analyzer.plot_correlation_heatmap()

# Generate Word document
analyzer.generate_word_document(...)

# Close connection
analyzer.close_db()
```

## Configuration

The script uses the database configuration from `scripts/db_config.py`:

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'user': 'your_username',
    'password': 'your_password'
}
```

## Output Directory Structure

```
results_analysis_v2/
├── Comprehensive_Results_Analysis.docx  # Main Word document
├── statistical_overview.csv             # Statistics table
├── visual_examples.csv                  # Representative examples
├── disagreement_cases.csv               # Method disagreements
└── plots/                               # All visualizations
    ├── score_distributions.png
    ├── bland_altman_plot.png
    ├── scatter_matrix.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── ensemble_weights.png
    ├── quality_categories.png
    ├── performance_efficiency.png
    └── model_type_analysis.png (if applicable)
```

## Visualization Details

### Color Scheme
The script uses a consistent, color-blind friendly palette:
- **Blue**: Traditional method
- **Orange**: Deep Learning method
- **Green**: Ensemble method

### Statistical Significance
- Correlation values include both PLCC (Pearson) and SROCC (Spearman)
- Bland-Altman plots show mean bias and 95% limits of agreement (±1.96 SD)
- Scatter plots include regression lines and R² values

### Image Quality
- All plots are saved at 300 DPI for publication quality
- Figures use consistent styling and formatting
- Grid lines and legends enhance readability

## Interpretation Guide

### Understanding the Results

1. **High PLCC/SROCC (> 0.8)**: Strong agreement between methods
2. **Bland-Altman bias near 0**: Good agreement with minimal systematic bias
3. **Tight limits of agreement**: Consistent predictions across methods
4. **High feature correlation**: Feature strongly contributes to quality assessment
5. **Balanced ensemble weights**: Both methods contribute meaningfully

### Common Insights

- **High traditional weight**: Image quality driven by sharpness, contrast
- **High DL weight**: Image quality affected by artifacts, contextual features
- **Large disagreement**: Conflicting quality aspects (e.g., sharp but noisy)

## Troubleshooting

### Database Connection Issues
```
Error: connection refused
Solution: Verify PostgreSQL is running and credentials are correct
```

### Missing Data
```
Error: No data found
Solution: Ensure the hybrid quality metrics have been computed and stored
```

### Memory Issues
```
Error: Out of memory
Solution: Process data in batches or increase available memory
```

### Import Errors
```
Error: No module named 'xxx'
Solution: Install all required dependencies using pip
```

## Best Practices

1. **Run after data collection**: Ensure all quality metrics are computed
2. **Check data quality**: Verify no NULL values in critical columns
3. **Review plots**: Check that visualizations make sense before generating Word doc
4. **Customize as needed**: Modify color schemes, plot sizes for your needs
5. **Version control**: Keep track of different analysis versions

## Performance

- **Typical runtime**: 1-5 minutes for 1,000-10,000 images
- **Memory usage**: ~500 MB - 2 GB depending on dataset size
- **Disk space**: ~10-50 MB for output files

## Citation

If you use this analysis script in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information here]

## Contact

For questions or issues:
- GitHub Issues: [repository issues page]
- Email: [your email]

## Changelog

### Version 2.0 (Current)
- Complete rewrite with comprehensive analysis
- Word document generation
- Enhanced visualizations
- Feature importance analysis
- Ensemble weight analysis
- Quality category performance
- Method disagreement cases

### Version 1.0
- Initial release with basic visualizations
