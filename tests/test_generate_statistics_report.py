"""
Test script for generate_statistics_report.py
Validates the script structure and logic without requiring database connection.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        # Test script import
        import scripts.generate_statistics_report as report_module
        print("✓ Script imports successfully")
        
        # Test class exists
        assert hasattr(report_module, 'StatisticsReportGenerator')
        print("✓ StatisticsReportGenerator class exists")
        
        # Test main function exists
        assert hasattr(report_module, 'main')
        print("✓ main function exists")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_class_structure():
    """Test the class structure"""
    try:
        import scripts.generate_statistics_report as report_module
        
        # Create instance with test config
        test_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass'
        }
        
        generator = report_module.StatisticsReportGenerator(test_config, output_dir='/tmp/test_reports')
        print("✓ StatisticsReportGenerator instance created")
        
        # Test attributes
        assert hasattr(generator, 'db_config')
        assert hasattr(generator, 'metrics')
        assert hasattr(generator, 'metric_descriptions')
        print("✓ Required attributes exist")
        
        # Test methods
        assert hasattr(generator, 'connect_database')
        assert hasattr(generator, 'fetch_data')
        assert hasattr(generator, 'calculate_statistics')
        assert hasattr(generator, 'create_visualizations')
        assert hasattr(generator, 'generate_report')
        print("✓ Required methods exist")
        
        # Test metrics mapping
        assert 'laplacian_variance' in generator.metrics
        assert 'gradient_energy' in generator.metrics
        assert 'rms_contrast' in generator.metrics
        assert 'entropy' in generator.metrics
        assert 'noise_estimate' in generator.metrics
        assert 'tenengrad' in generator.metrics
        assert 'mscn_std' in generator.metrics
        print("✓ All 7 metrics defined")
        
        # Test metric descriptions
        for metric_name in generator.metrics.values():
            assert metric_name in generator.metric_descriptions
        print("✓ All metric descriptions defined")
        
        return True
    except Exception as e:
        print(f"✗ Class structure test failed: {e}")
        return False

def test_metrics_mapping():
    """Test that metrics mapping is correct"""
    try:
        import scripts.generate_statistics_report as report_module
        
        test_config = {'host': 'localhost', 'database': 'test'}
        generator = report_module.StatisticsReportGenerator(test_config)
        
        expected_metrics = {
            'laplacian_variance': 'laplacian',
            'gradient_energy': 'gradient_energy',
            'rms_contrast': 'rms_contrast',
            'entropy': 'entropy',
            'noise_estimate': 'noise',
            'tenengrad': 'tenengrad',
            'mscn_std': 'mscn'
        }
        
        for db_col, metric_name in expected_metrics.items():
            assert generator.metrics[db_col] == metric_name, f"Metric mapping mismatch for {db_col}"
        
        print("✓ Metric mapping is correct")
        return True
    except Exception as e:
        print(f"✗ Metrics mapping test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Testing generate_statistics_report.py")
    print("="*60 + "\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Class Structure Test", test_class_structure),
        ("Metrics Mapping Test", test_metrics_mapping)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED\n")
        else:
            failed += 1
            print(f"✗ {test_name} FAILED\n")
    
    print("="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
