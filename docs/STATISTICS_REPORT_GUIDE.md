# Statistics Report Generation Guide

## Overview

The optimized statistics report generator creates comprehensive, dashboard-ready reports for image quality metrics. The report has been streamlined from 9 statistics per metric to 5 most useful metrics, with additional executive summary and analysis features.

## Key Improvements

### 1. Optimized Statistics (5 Key Metrics)

**Before**: 9 statistics per metric (count, mean, median, std, min, 25%, 50%, 75%, max)

**After**: 5 essential statistics per metric:
- **Count**: Number of samples (data quality indicator)
- **Mean**: Average value (central tendency)
- **Std Dev**: Standard deviation (consistency/variability indicator)
- **Min**: Minimum value (worst-case scenario)
- **Max**: Maximum value (best-case scenario)

**Rationale**: 
- Reduces cognitive load and information overload
- Focuses on actionable metrics
- Easier to display in dashboards
- Quartiles removed as they're redundant with min/max/mean/std for most use cases

### 2. Executive Summary Dashboard

A new high-level overview section includes:

#### Dataset Overview
- Total number of images analyzed
- Number of metrics computed

#### Quality Distribution
Images are automatically classified into three categories:
- **Good**: High sharpness, low noise
- **Fair**: Medium quality metrics
- **Poor**: Low sharpness or high noise

Classification criteria:
```python
Good:  laplacian > (mean + 0.5*std) AND noise < (mean - 0.3*std)
Poor:  laplacian < (mean - 0.5*std) OR noise > (mean + 0.5*std)
Fair:  Everything else
```

#### Key Findings
Automated summary of:
- Average sharpness (Laplacian variance)
- Average noise level
- Average contrast (RMS)

### 3. Correlation Analysis

New feature that shows relationships between metrics:

- **Correlation Matrix Heatmap**: Visual representation of how metrics relate to each other
- **Insight**: Identifies redundant metrics (high correlation >0.7)
- **Insight**: Identifies complementary metrics (low correlation)
- **Use Case**: Helps optimize metric selection for your specific use case

### 4. Enhanced Visualizations

Each metric includes:
- **Distribution Histogram**: Shows how values are distributed
- **Boxplot**: Identifies outliers and quartile ranges
- **5-Metric Summary Table**: Quick reference for key statistics

## Report Structure

```
1. Title Page
   - Report title
   - Generation timestamp

2. Executive Summary Dashboard
   - Dataset overview
   - Quality distribution (Good/Fair/Poor)
   - Key findings

3. Correlation Analysis
   - Heatmap showing metric relationships
   - Explanation of correlation significance

4. Detailed Metric Analysis
   - One section per metric:
     * Metric description (in Uzbek)
     * 5-statistic summary table
     * Distribution histogram
     * Boxplot
```

## Usage

### Basic Usage

```python
from scripts.generate_statistics_report import StatisticsReportGenerator
from scripts.db_config import DB_CONFIG

# Create generator
generator = StatisticsReportGenerator(
    DB_CONFIG,
    output_dir='reports',
    n_sample_images=3,
    histogram_bins='auto'
)

# Generate report
generator.generate_report()
```

### Configuration Options

```python
StatisticsReportGenerator(
    db_config,              # Database connection configuration
    output_dir='reports',   # Output directory for report and visualizations
    n_sample_images=3,      # Number of sample images (currently not used in optimized version)
    histogram_bins='auto'   # Histogram bins: 'auto' or integer value
)
```

### Running from Command Line

```bash
cd /path/to/endoscopic-iqa-project
python scripts/generate_statistics_report.py
```

Output:
- Word document: `reports/Image_Quality_Report_Optimized.docx`
- Visualizations: `reports/*.png`

## Best Practices for Dashboard Integration

### 1. Use the 5 Key Statistics

These statistics provide a complete picture while being concise:

```python
{
    'count': 1000,      # Sample size - validate data quality
    'mean': 25.5,       # Central tendency - average quality
    'std': 5.2,         # Variability - consistency indicator
    'min': 10.0,        # Lower bound - worst case
    'max': 45.0         # Upper bound - best case
}
```

### 2. Quality Classification for Decision Making

Use the Good/Fair/Poor classification for:
- **Quality monitoring**: Track percentage of "Good" images over time
- **Threshold setting**: Set acceptance criteria (e.g., >70% Good images)
- **Alerts**: Trigger alerts when "Poor" percentage exceeds threshold

### 3. Correlation Analysis for Metric Selection

Use correlation matrix to:
- **Identify redundant metrics**: High correlation (>0.7) means similar information
- **Optimize metric set**: Keep diverse metrics with low correlation
- **Reduce computation**: Remove redundant metrics in production

### 4. Visualization Tips

- **Histograms**: Show distribution shape (normal, skewed, bimodal)
- **Boxplots**: Quickly identify outliers
- **Trends**: Compare reports over time to track quality improvements

## Metrics Explained

### Sharpness Metrics

1. **Laplacian Variance**: Edge sharpness detector
   - Higher = Sharper image
   - Most reliable for blur detection

2. **Tenengrad**: Overall sharpness measure
   - Higher = Better focus
   - Sensitive to texture

3. **Gradient Energy**: Edge intensity
   - Higher = More defined edges
   - Good for structure assessment

### Quality Metrics

4. **RMS Contrast**: Contrast level
   - Higher = Better contrast
   - Important for visibility

5. **Entropy**: Information content
   - Higher = More detail/complexity
   - Indicator of image richness

### Degradation Metrics

6. **Noise Estimate**: Noise level
   - Lower = Cleaner image
   - Critical for quality assessment

7. **MSCN Std**: Local contrast variation
   - Indicates natural image statistics
   - Used in BRISQUE algorithm

## Interpretation Guide

### Good Image Indicators
- High Laplacian (>mean + std)
- High Tenengrad (>mean + std)
- Low Noise (<mean - std)
- High RMS Contrast (>mean)
- High Entropy (>mean)

### Poor Image Indicators
- Low Laplacian (<mean - std)
- Low Tenengrad (<mean - std)
- High Noise (>mean + std)
- Low RMS Contrast (<mean)
- Low Entropy (<mean - std)

## Troubleshooting

### Report Generation Fails

```
✗ Error connecting to database
```
**Solution**: Check database configuration in `scripts/db_config.py`

### No Data Available

```
⚠ No data available for [metric_name]
```
**Solution**: Verify that quality metrics have been computed for your dataset

### Import Errors

```
ModuleNotFoundError: No module named 'psycopg2'
```
**Solution**: Install dependencies:
```bash
pip install psycopg2-binary pandas matplotlib seaborn python-docx
```

## Examples

### Example Output Summary

```
✓ Optimized report saved to: reports/Image_Quality_Report_Optimized.docx
✓ Visualizations saved to: reports/
✓ Report includes:
  - Executive summary dashboard
  - Quality classification
  - Correlation analysis
  - 5 key statistics per metric (reduced from 9)
  - Detailed visualizations
```

### Example Quality Distribution

```
Category | Count | Percentage
---------|-------|------------
Good     |   450 |      45.0%
Fair     |   400 |      40.0%
Poor     |   150 |      15.0%
```

### Example Statistics Table

```
Metric: Laplacian Variance
Count | Mean   | Std Dev | Min    | Max
------|--------|---------|--------|--------
1000  | 25.543 | 5.234   | 10.123 | 45.678
```

## Future Enhancements

Potential additions for future versions:
1. Time-series analysis (quality trends over time)
2. Per-category statistics (if images are categorized)
3. Automated recommendations based on quality distribution
4. Export to JSON/CSV for external dashboard tools
5. Interactive HTML dashboard version
6. Comparative analysis (compare two datasets)

## References

- **Statistical Best Practices**: Focus on mean, std, min, max for quick insights
- **Dashboard Design**: Cognitive load reduction through focused metrics
- **Quality Assessment**: BRISQUE, NIQE, and traditional IQA approaches

## Support

For issues or questions:
1. Check test file: `tests/test_generate_statistics_report.py`
2. Review code: `scripts/generate_statistics_report.py`
3. Submit GitHub issue with details and error messages

---

**Version**: 2.0 (Optimized)  
**Last Updated**: 2026-02-12  
**Authors**: Avazbek Hasanov
