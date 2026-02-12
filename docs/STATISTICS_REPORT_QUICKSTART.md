# Statistical Report Generation - Quick Start

## Overview
The `generate_statistics_report.py` script creates comprehensive statistical reports from image quality metrics stored in the PostgreSQL database.

## Prerequisites
1. PostgreSQL database with computed metrics (run `compute_hybrid_quality_metrics.py` first)
2. Required Python packages (install via `pip install -r requirements.txt`)
3. Database configuration in `scripts/db_config.py`

## Basic Usage

```bash
# Navigate to scripts directory
cd scripts

# Run the report generator
python3 generate_statistics_report.py
```

## Output
The script generates:
- `reports/Image_Quality_Report.docx` - Complete Word document with statistics and visualizations
- `reports/*_hist.png` - Histogram for each metric
- `reports/*_box.png` - Boxplot for each metric

## Customization

You can customize the report by editing the `main()` function in the script:

```python
generator = StatisticsReportGenerator(
    DB_CONFIG, 
    output_dir='reports',           # Change output directory
    n_sample_images=5,              # Include 5 sample images instead of 3
    histogram_bins=30               # Use 30 bins instead of auto
)
```

### Configuration Options

- **output_dir** (str): Directory to save reports and visualizations
  - Default: `'reports'`
  - Example: `'my_reports'`, `'/tmp/analysis'`

- **n_sample_images** (int): Number of sample images to include in the report
  - Default: `3`
  - Example: `5`, `10`, `0` (no images)

- **histogram_bins** (int or 'auto'): Number of bins for histogram visualization
  - Default: `'auto'` (automatically calculated based on data size)
  - Example: `20`, `30`, `50`

## Metrics Included

The report includes statistics for 7 quality metrics:

1. **Laplacian Variance** - Measures blur and edge sharpness
2. **Gradient Energy** - Evaluates texture and edge clarity
3. **RMS Contrast** - Measures brightness and contrast levels
4. **Entropy** - Indicates information content and detail
5. **Noise Estimate** - Quantifies noise level in images
6. **Tenengrad** - Overall sharpness measure
7. **MSCN Std Dev** - Local contrast assessment

## Statistics Calculated

For each metric, the report includes:
- **count**: Number of images analyzed
- **mean**: Average value
- **median**: Middle value (50th percentile)
- **std**: Standard deviation (data spread)
- **min**: Minimum value
- **25%**: First quartile
- **50%**: Median (same as median above)
- **75%**: Third quartile
- **max**: Maximum value

## Example Output

```
============================================================
GENERATING STATISTICAL REPORT
============================================================

✓ Connected to database: postgres
✓ Fetched 17239 records from database

📊 Calculating statistics...

📄 Creating Word document...
  Processing: laplacian
  Processing: gradient_energy
  Processing: rms_contrast
  Processing: entropy
  Processing: noise
  Processing: tenengrad
  Processing: mscn

============================================================
✓ Report saved to: reports/Image_Quality_Report.docx
✓ Visualizations saved to: reports
============================================================

✓ Word hisobot tayyor!
```

## Troubleshooting

### Database Connection Error
- Verify database credentials in `scripts/db_config.py`
- Ensure PostgreSQL server is running
- Check that the database exists

### No Data Available
- Run `compute_hybrid_quality_metrics.py` first to populate the database
- Verify the `image_quality_metrics_hybrid` table exists and has data

### Missing Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install python-docx pandas matplotlib seaborn psycopg2-binary
```

## Advanced Usage

### Programmatic Usage

```python
from scripts.generate_statistics_report import StatisticsReportGenerator
from scripts.db_config import DB_CONFIG

# Create generator with custom settings
generator = StatisticsReportGenerator(
    DB_CONFIG,
    output_dir='monthly_reports',
    n_sample_images=10,
    histogram_bins=25
)

# Generate the report
generator.generate_report()
```

### Batch Processing

```python
# Generate multiple reports with different configurations
configs = [
    {'output_dir': 'reports_3samples', 'n_sample_images': 3},
    {'output_dir': 'reports_5samples', 'n_sample_images': 5},
    {'output_dir': 'reports_10samples', 'n_sample_images': 10}
]

for config in configs:
    generator = StatisticsReportGenerator(DB_CONFIG, **config)
    generator.generate_report()
```

## Notes

- The script creates the output directory automatically if it doesn't exist
- Generated PNG files are temporary visualization files; the main output is the Word document
- Sample images are only included if the file paths in the database are valid and accessible
- All metric descriptions are provided in Uzbek language

## Related Scripts

- `compute_hybrid_quality_metrics.py` - Compute metrics before generating reports
- `query_hybrid_quality_metrics.py` - Interactive query tool for metrics
- `extract_image_metadata.py` - Extract image metadata to database
