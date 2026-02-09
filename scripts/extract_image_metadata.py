import os
import psycopg2
from pathlib import Path
from PIL import Image
from datetime import datetime
import hashlib
import sys

class ImageMetadataExtractor:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.setup_database()

    def setup_database(self):
        """Create database connection and table for storing image metadata"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database', 'postgres'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', '')
            )
            print(f"✓ Connected to PostgreSQL database: {self.db_config.get('database')}")

            cursor = self.conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS image_metadata (
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
                )
            ''')

            # Create indexes for better query performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_dataset_name ON image_metadata(dataset_name)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON image_metadata(category)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_md5_hash ON image_metadata(md5_hash)
            ''')

            self.conn.commit()
            print("✓ Database table and indexes created/verified")

        except Exception as e:
            print(f"✗ Error connecting to database: {str(e)}")
            sys.exit(1)

    def get_image_metadata(self, image_path):
        """Extract metadata from a single image"""
        try:
            with Image.open(image_path) as img:
                file_stats = os.stat(image_path)

                with open(image_path, 'rb') as f:
                    md5_hash = hashlib.md5(f.read()).hexdigest()

                return {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'size_bytes': file_stats.st_size,
                    'md5_hash': md5_hash
                }
        except Exception as e:
            print(f"✗ Error processing {image_path}: {str(e)}")
            return None

    def parse_path_structure(self, file_path, base_path):
        """Extract dataset, category, and subcategory from file path"""
        relative_path = Path(file_path).relative_to(base_path)
        parts = relative_path.parts

        dataset_name = parts[0] if len(parts) > 0 else None
        category = parts[1] if len(parts) > 1 else None
        subcategory = parts[2] if len(parts) > 2 else None

        return dataset_name, category, subcategory

    def scan_directory(self, base_path):
        """Recursively scan directory for images"""
        base_path = Path(base_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.tif'}

        cursor = self.conn.cursor()
        count = 0
        errors = 0

        print(f"\n🔍 Scanning directory: {base_path}")
        print("=" * 80)

        for file_path in base_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                metadata = self.get_image_metadata(file_path)

                if metadata:
                    dataset_name, category, subcategory = self.parse_path_structure(
                        file_path, base_path
                    )

                    try:
                        cursor.execute('''
                            INSERT INTO image_metadata
                            (file_path, dataset_name, category, subcategory, filename,
                             width, height, format, mode, size_bytes, md5_hash)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (file_path) DO UPDATE SET
                                width = EXCLUDED.width,
                                height = EXCLUDED.height,
                                format = EXCLUDED.format,
                                mode = EXCLUDED.mode,
                                size_bytes = EXCLUDED.size_bytes,
                                md5_hash = EXCLUDED.md5_hash,
                                updated_at = CURRENT_TIMESTAMP
                        ''', (
                            str(file_path),
                            dataset_name,
                            category,
                            subcategory,
                            file_path.name,
                            metadata['width'],
                            metadata['height'],
                            metadata['format'],
                            metadata['mode'],
                            metadata['size_bytes'],
                            metadata['md5_hash']
                        ))
                        count += 1

                        if count % 100 == 0:
                            print(f"📊 Processed {count} images...")
                            self.conn.commit()

                    except Exception as e:
                        errors += 1
                        print(f"✗ Error inserting {file_path}: {str(e)}")
                else:
                    errors += 1

        self.conn.commit()
        print("=" * 80)
        print(f"✓ Total images processed: {count}")
        if errors > 0:
            print(f"✗ Errors encountered: {errors}")

    def get_statistics(self):
        """Get statistics from the database"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM image_metadata")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT dataset_name, COUNT(*) as count
            FROM image_metadata
            GROUP BY dataset_name
            ORDER BY dataset_name
        """)
        by_dataset = cursor.fetchall()

        cursor.execute("""
            SELECT dataset_name, category, COUNT(*) as count
            FROM image_metadata
            GROUP BY dataset_name, category
            ORDER BY dataset_name, category
        """)
        by_category = cursor.fetchall()

        cursor.execute("""
            SELECT 
                AVG(width) as avg_width,
                AVG(height) as avg_height,
                AVG(size_bytes) as avg_size
            FROM image_metadata
        """)
        avg_stats = cursor.fetchone()

        return {
            'total': total,
            'by_dataset': by_dataset,
            'by_category': by_category,
            'averages': avg_stats
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\n✓ Database connection closed")


def main():
    # Import database configuration
    try:
        from db_config import DB_CONFIG
        db_config = DB_CONFIG
        print(f"📝 Using database: {db_config['database']} with user: {db_config['user']}")
    except ImportError:
        print("⚠️  db_config.py not found, using defaults")
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'postgres',
            'user': os.environ.get('USER', 'postgres'),
            'password': ''
        }

    # Base path to datasets
    base_path = "/Users/hasanov_avazbek/Desktop/Projects/Study/endoscopic-iqa-project/data/datasets"

    print("=" * 80)
    print("🔬 ENDOSCOPIC IMAGE METADATA EXTRACTOR")
    print("=" * 80)

    extractor = ImageMetadataExtractor(db_config)

    print("\n📁 Starting directory scan...")
    extractor.scan_directory(base_path)

    print("\n📈 STATISTICS")
    print("=" * 80)
    stats = extractor.get_statistics()

    print(f"\n📊 Total images in database: {stats['total']}")

    print("\n📦 Images by dataset:")
    for dataset, count in stats['by_dataset']:
        print(f"  • {dataset}: {count} images")

    print("\n📂 Images by category:")
    for dataset, category, count in stats['by_category']:
        if category:
            print(f"  • {dataset}/{category}: {count} images")

    if stats['averages']:
        avg_width, avg_height, avg_size = stats['averages']
        print(f"\n📏 Average image dimensions: {int(avg_width)}x{int(avg_height)} pixels")
        print(f"💾 Average file size: {int(avg_size / 1024):.2f} KB")

    print("\n" + "=" * 80)
    print("✅ METADATA EXTRACTION COMPLETE!")
    print("=" * 80)

    extractor.close()


if __name__ == "__main__":
    main()



