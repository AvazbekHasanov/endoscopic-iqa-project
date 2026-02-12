"""
Test script for results_analysis_v2.py
Tests the script without requiring a database connection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("Testing results_analysis_v2.py...")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    from evaluation.results_analysis_v2 import ResultsAnalyzer
    print("✓ Successfully imported ResultsAnalyzer")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test class instantiation
print("\n2. Testing class instantiation...")
try:
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'postgres',
        'user': 'test_user',
        'password': 'test_pass'
    }
    
    analyzer = ResultsAnalyzer(db_config, output_dir='/tmp/test_results_analysis')
    print("✓ Successfully created ResultsAnalyzer instance")
    print(f"✓ Output directory: {analyzer.output_dir}")
    print(f"✓ Plots directory: {analyzer.plots_dir}")
except Exception as e:
    print(f"✗ Error creating analyzer: {e}")
    sys.exit(1)

# Test method existence
print("\n3. Testing method existence...")
methods = [
    'connect_db',
    'fetch_data',
    'generate_statistical_overview',
    'plot_score_distributions',
    'plot_bland_altman',
    'plot_scatter_matrix',
    'plot_correlation_heatmap',
    'plot_feature_importance',
    'plot_ensemble_weights',
    'plot_quality_categories',
    'plot_performance_efficiency',
    'generate_visual_examples_table',
    'generate_disagreement_cases',
    'plot_model_type_analysis',
    'generate_word_document',
    'run_complete_analysis'
]

missing_methods = []
for method in methods:
    if not hasattr(analyzer, method):
        missing_methods.append(method)
        print(f"✗ Missing method: {method}")
    else:
        print(f"✓ Method exists: {method}")

if missing_methods:
    print(f"\n✗ Missing {len(missing_methods)} methods")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nThe script is ready to use.")
print("To run the analysis, execute:")
print("  python3 evaluation/results_analysis_v2.py")
print("\nNote: This requires a PostgreSQL database with the image_quality_metrics_hybrid table.")
