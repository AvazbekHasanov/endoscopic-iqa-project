"""
Generate statistical report for image quality metrics from the database.
Creates a Word document with statistics tables, visualizations, and sample images.
"""

import os
import sys
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from docx import Document
from docx.shared import Inches

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.db_config import DB_CONFIG


class StatisticsReportGenerator:
    """Generate comprehensive statistical report for image quality metrics"""
    
    def __init__(self, db_config, output_dir='reports', n_sample_images=3, histogram_bins='auto'):
        self.db_config = db_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.conn = None
        self.df = None
        self.n_sample_images = n_sample_images
        self.histogram_bins = histogram_bins  # Can be int or 'auto'
        
        # Metric names mapping (database column names)
        self.metrics = {
            'laplacian_variance': 'laplacian',
            'gradient_energy': 'gradient_energy',
            'rms_contrast': 'rms_contrast',
            'entropy': 'entropy',
            'noise_estimate': 'noise',
            'tenengrad': 'tenengrad',
            'mscn_std': 'mscn'
        }
        
        # Metric descriptions in Uzbek
        self.metric_descriptions = {
            'laplacian': "Laplacian o'lchovi tasvirning burchak va kontur aniqligini ko'rsatadi. Yuqori qiymat tasvir aniqroq ekanini bildiradi.",
            'gradient_energy': "Gradient energy tasvirning tekstura va qirralarining aniqligini baholaydi. Yuqori qiymatlar ko'proq qirralar va teksturani bildiradi.",
            'rms_contrast': "RMS kontrast tasvir yorqinligi va kontrast darajasini o'lchaydi. Yuqori RMS kontrast tasvir ko'proq kontrastli ekanini bildiradi.",
            'entropy': "Entropiya tasvirdagi axborot miqdorini ifodalaydi. Yuqori entropiya ko'proq tafsilot va murakkablikni bildiradi.",
            'noise': "Noise tasvirdagi shovqin darajasini o'lchaydi. Yuqori qiymat tasvirda ko'p shovqin borligini bildiradi.",
            'tenengrad': "Tenengrad o'lchovi tasvirning o'rtacha aniqligini ko'rsatadi. Yuqori qiymat tasvir aniqroq ekanini bildiradi.",
            'mscn': "MSCN tasvirning lokal kontrastini baholaydi. Yuqori qiymat ko'proq lokal kontrastni bildiradi."
        }
        
    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'postgres'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', '')
            )
            print(f"✓ Connected to database: {self.db_config.get('database')}")
        except Exception as e:
            print(f"✗ Error connecting to database: {str(e)}")
            sys.exit(1)
    
    def fetch_data(self):
        """Fetch image quality metrics from database"""
        query = """
        SELECT 
            id,
            laplacian_variance,
            gradient_energy,
            rms_contrast,
            entropy,
            noise_estimate,
            tenengrad,
            mscn_std,
            file_path
        FROM image_quality_metrics_hybrid
        """
        
        try:
            self.df = pd.read_sql(query, self.conn)
            print(f"✓ Fetched {len(self.df)} records from database")
        except Exception as e:
            print(f"✗ Error fetching data: {str(e)}")
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()
    
    def calculate_statistics(self):
        """Calculate statistics for each metric"""
        stats_summary = {}
        
        for db_col, metric_name in self.metrics.items():
            if db_col in self.df.columns:
                data = self.df[db_col].dropna()
                
                if len(data) > 0:
                    stats_summary[metric_name] = {
                        'count': int(data.count()),
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'min': float(data.min()),
                        '25%': float(data.quantile(0.25)),
                        '50%': float(data.quantile(0.5)),
                        '75%': float(data.quantile(0.75)),
                        'max': float(data.max())
                    }
                else:
                    print(f"⚠ No data available for {metric_name}")
        
        return stats_summary
    
    def create_visualizations(self, metric_name, db_col):
        """Create histogram and boxplot for a metric"""
        if db_col not in self.df.columns:
            print(f"⚠ Column {db_col} not found in dataframe")
            return None, None
        
        data = self.df[db_col].dropna()
        
        if len(data) == 0:
            print(f"⚠ No data available for {metric_name}")
            return None, None
        
        # Determine bins (use auto or configured value)
        bins = self.histogram_bins if self.histogram_bins != 'auto' else min(50, max(10, len(data) // 100))
        
        # Create histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(data, bins=bins, kde=True, color='skyblue')
        plt.title(f"{metric_name} distribution")
        plt.xlabel(metric_name)
        plt.ylabel("Frequency")
        hist_path = self.output_dir / f"{metric_name}_hist.png"
        plt.savefig(hist_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Create boxplot
        plt.figure(figsize=(6, 2))
        sns.boxplot(x=data, color='lightgreen')
        plt.title(f"{metric_name} Boxplot")
        box_path = self.output_dir / f"{metric_name}_box.png"
        plt.savefig(box_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return hist_path, box_path
    
    def add_statistics_table(self, doc, metric_name, stats):
        """Add statistics table to Word document"""
        table = doc.add_table(rows=2, cols=9)
        table.style = 'Light Grid Accent 1'
        
        # Header row
        hdr_cells = table.rows[0].cells
        headers = ['count', 'mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max']
        for i, header in enumerate(headers):
            hdr_cells[i].text = header
        
        # Data row
        data_cells = table.rows[1].cells
        for i, header in enumerate(headers):
            value = stats.get(header, 0)
            # Format count as integer, others as decimal
            if header == 'count':
                data_cells[i].text = str(int(value))
            else:
                data_cells[i].text = str(round(value, 3))
    
    def get_sample_images(self, n_samples=None):
        """Get sample image paths from database"""
        if n_samples is None:
            n_samples = self.n_sample_images
        if 'file_path' in self.df.columns:
            sample_paths = self.df['file_path'].dropna().head(n_samples)
            # Filter existing paths
            existing_paths = [path for path in sample_paths if os.path.exists(path)]
            return existing_paths
        return []
    
    def generate_report(self):
        """Generate complete Word report"""
        print("\n" + "="*60)
        print("GENERATING STATISTICAL REPORT")
        print("="*60)
        
        # Connect and fetch data
        self.connect_database()
        self.fetch_data()
        
        if self.df is None or len(self.df) == 0:
            print("✗ No data available to generate report")
            return
        
        # Calculate statistics
        print("\n📊 Calculating statistics...")
        stats_summary = self.calculate_statistics()
        
        # Create Word document
        print("\n📄 Creating Word document...")
        doc = Document()
        doc.add_heading("Endoskopik Tasvir Sifat Statistikasi", 0)
        doc.add_paragraph(f"Datasetdagi jami tasvirlar soni: {len(self.df)}\n")
        
        # Generate sections for each metric
        for db_col, metric_name in self.metrics.items():
            if metric_name not in stats_summary:
                continue
            
            print(f"  Processing: {metric_name}")
            
            # Add metric heading and description
            doc.add_heading(metric_name.upper(), level=1)
            doc.add_paragraph(self.metric_descriptions.get(metric_name, ""))
            
            # Add statistics table
            self.add_statistics_table(doc, metric_name, stats_summary[metric_name])
            
            # Generate and add visualizations
            hist_path, box_path = self.create_visualizations(metric_name, db_col)
            
            if hist_path and os.path.exists(hist_path):
                doc.add_paragraph("\nHistogram (tasvir sifatining taqsimoti)")
                doc.add_picture(str(hist_path), width=Inches(5))
            
            if box_path and os.path.exists(box_path):
                doc.add_paragraph("\nBoxplot (tasvir sifatidagi dispersiya va outlierlar)")
                doc.add_picture(str(box_path), width=Inches(5))
            
            # Add sample images (if available)
            sample_images = self.get_sample_images()
            if sample_images:
                doc.add_paragraph("\nNamuna tasvirlar:")
                for img_path in sample_images:
                    try:
                        doc.add_picture(img_path, width=Inches(3))
                        doc.add_paragraph(f"  Path: {img_path}")
                    except Exception as e:
                        print(f"    ⚠ Could not add image {img_path}: {e}")
            
            doc.add_page_break()
        
        # Save document
        output_path = self.output_dir / "Image_Quality_Report.docx"
        doc.save(str(output_path))
        
        print("\n" + "="*60)
        print(f"✓ Report saved to: {output_path}")
        print(f"✓ Visualizations saved to: {self.output_dir}")
        print("="*60)


def main():
    """Main function"""
    # Create report generator
    generator = StatisticsReportGenerator(DB_CONFIG, output_dir='reports')
    
    # Generate report
    generator.generate_report()
    
    print("\n✓ Word hisobot tayyor!")


if __name__ == "__main__":
    main()
