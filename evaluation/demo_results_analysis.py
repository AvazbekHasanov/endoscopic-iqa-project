"""
Demo script for results_analysis_v2.py with mock data.
Demonstrates the functionality without requiring a database.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("📊 RESULTS ANALYSIS V2 - DEMO WITH MOCK DATA")
print("=" * 80)

# Generate mock data
print("\n1. Generating mock data...")
np.random.seed(42)

n_images = 500

# Generate correlated scores (traditional and DL should be somewhat correlated)
traditional_scores = np.random.beta(5, 2, n_images)  # Slightly higher quality
dl_scores = traditional_scores * 0.7 + np.random.beta(4, 3, n_images) * 0.3
ensemble_scores = (traditional_scores + dl_scores) / 2

# Add some noise
traditional_scores = np.clip(traditional_scores + np.random.normal(0, 0.05, n_images), 0, 1)
dl_scores = np.clip(dl_scores + np.random.normal(0, 0.05, n_images), 0, 1)
ensemble_scores = np.clip(ensemble_scores + np.random.normal(0, 0.03, n_images), 0, 1)

# Generate traditional features
features = {
    'laplacian_variance': np.random.lognormal(3, 1, n_images),
    'rms_contrast': np.random.gamma(2, 50, n_images),
    'noise_estimate': np.random.exponential(0.05, n_images),
    'mscn_std': np.random.gamma(3, 0.3, n_images),
    'gradient_energy': np.random.lognormal(5, 1, n_images),
    'entropy': np.random.uniform(4, 7, n_images),
    'tenengrad': np.random.lognormal(6, 1, n_images)
}

# Generate ensemble weights
weights_trad = np.random.beta(5, 5, n_images)
weights_dl = 1 - weights_trad

# Generate processing times
processing_times = np.random.lognormal(4, 0.5, n_images)

# Create DataFrame
data = pd.DataFrame({
    'id': range(1, n_images + 1),
    'image_id': range(1, n_images + 1),
    'filename': [f'image_{i:04d}.jpg' for i in range(1, n_images + 1)],
    'file_path': [f'/data/images/image_{i:04d}.jpg' for i in range(1, n_images + 1)],
    'dataset_name': np.random.choice(['Kvasir', 'Hyper-Kvasir', 'CVC-Clinic'], n_images),
    'category': np.random.choice(['polyp', 'normal', 'inflammation'], n_images),
    'width': 1920,
    'height': 1080,
    'ensemble_score': ensemble_scores,
    'traditional_score': traditional_scores,
    'deep_learning_score': dl_scores,
    'laplacian_variance': features['laplacian_variance'],
    'rms_contrast': features['rms_contrast'],
    'noise_estimate': features['noise_estimate'],
    'mscn_std': features['mscn_std'],
    'gradient_energy': features['gradient_energy'],
    'entropy': features['entropy'],
    'tenengrad': features['tenengrad'],
    'processing_time_ms': processing_times,
    'ensemble_weights_traditional': weights_trad,
    'ensemble_weights_dl': weights_dl,
    'model_type': np.random.choice(['mobilenet_v2', 'resnet18'], n_images),
    'computed_at': pd.Timestamp.now()
})

print(f"✓ Generated {len(data)} mock images")
print(f"✓ Datasets: {data['dataset_name'].unique().tolist()}")
print(f"✓ Score ranges:")
print(f"  - Ensemble: {data['ensemble_score'].min():.3f} - {data['ensemble_score'].max():.3f}")
print(f"  - Traditional: {data['traditional_score'].min():.3f} - {data['traditional_score'].max():.3f}")
print(f"  - Deep Learning: {data['deep_learning_score'].min():.3f} - {data['deep_learning_score'].max():.3f}")

# Save mock data
output_dir = Path('/tmp/results_analysis_demo')
output_dir.mkdir(exist_ok=True)
data.to_csv(output_dir / 'mock_data.csv', index=False)
print(f"\n✓ Mock data saved to: {output_dir / 'mock_data.csv'}")

# Create a mock analyzer class that uses DataFrame instead of database
print("\n2. Creating mock analyzer with generated data...")

# We'll create a simplified version that doesn't need database
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Create output directories
plots_dir = output_dir / 'plots'
plots_dir.mkdir(exist_ok=True)

# Generate some example visualizations
print("\n3. Generating example visualizations...")

# Statistical Overview
print("\n   a. Statistical Overview...")
metrics = ['ensemble_score', 'traditional_score', 'deep_learning_score', 'processing_time_ms']
stats_data = []
for metric in metrics:
    values = data[metric].dropna()
    stats_data.append({
        'Metric': metric.replace('_', ' ').title(),
        'Mean': f"{values.mean():.4f}",
        'Median': f"{values.median():.4f}",
        'Std': f"{values.std():.4f}",
        'Min': f"{values.min():.4f}",
        'Max': f"{values.max():.4f}"
    })

stats_df = pd.DataFrame(stats_data)
print(stats_df.to_string(index=False))
stats_df.to_csv(output_dir / 'statistical_overview.csv', index=False)

# Score Distribution
print("\n   b. Score Distribution Plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

score_data = pd.DataFrame({
    'Ensemble': data['ensemble_score'],
    'Traditional': data['traditional_score'],
    'Deep Learning': data['deep_learning_score']
})
score_data_melted = score_data.melt(var_name='Method', value_name='Score')

sns.violinplot(data=score_data_melted, x='Method', y='Score', ax=axes[0])
axes[0].set_title('Score Distribution - Violin Plot', fontsize=14, fontweight='bold')

sns.boxplot(data=score_data_melted, x='Method', y='Score', ax=axes[1])
axes[1].set_title('Score Distribution - Box Plot', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("     ✓ Saved score_distributions.png")

# Correlation Heatmap
print("\n   c. Correlation Heatmap...")
scores = data[['ensemble_score', 'traditional_score', 'deep_learning_score']].copy()
scores.columns = ['Ensemble', 'Traditional', 'Deep Learning']
plcc = scores.corr(method='pearson')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(plcc, annot=True, fmt='.4f', cmap='coolwarm', center=0, 
           vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Pearson Linear Correlation (PLCC)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("     ✓ Saved correlation_heatmap.png")

# Feature Importance
print("\n   d. Feature Importance...")
features = ['laplacian_variance', 'rms_contrast', 'noise_estimate', 
           'mscn_std', 'gradient_energy', 'entropy', 'tenengrad']

correlations = {}
for feature in features:
    corr = data[feature].corr(data['ensemble_score'])
    correlations[feature] = corr

feature_names = [f.replace('_', ' ').title() for f in features]
corr_values = [correlations[f] for f in features]

fig, ax = plt.subplots(figsize=(10, 6))
colors_bars = ['green' if v > 0 else 'red' for v in corr_values]
ax.barh(feature_names, corr_values, color=colors_bars, alpha=0.7)
ax.set_xlabel('Correlation with Ensemble Score', fontsize=12)
ax.set_title('Feature Contribution to Quality Score', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("     ✓ Saved feature_importance.png")

# Ensemble Weights
print("\n   e. Ensemble Weights...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data['ensemble_weights_traditional'], bins=30, alpha=0.7, 
            color='blue', label='Traditional', edgecolor='black')
axes[0].hist(data['ensemble_weights_dl'], bins=30, alpha=0.7,
            color='orange', label='Deep Learning', edgecolor='black')
axes[0].set_xlabel('Weight', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Ensemble Weight Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

avg_trad = data['ensemble_weights_traditional'].mean()
avg_dl = data['ensemble_weights_dl'].mean()
axes[1].pie([avg_trad, avg_dl], 
           labels=['Traditional', 'Deep Learning'],
           colors=['blue', 'orange'],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Average Ensemble Weights', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(plots_dir / 'ensemble_weights.png', dpi=300, bbox_inches='tight')
plt.close()
print("     ✓ Saved ensemble_weights.png")

print("\n" + "=" * 80)
print("✅ DEMO COMPLETE!")
print("=" * 80)
print(f"\n📁 Results saved in: {output_dir}")
print(f"📊 Plots saved in: {plots_dir}")
print(f"\n📄 Files generated:")
print(f"   • mock_data.csv - Generated mock data")
print(f"   • statistical_overview.csv - Statistics table")
print(f"   • plots/score_distributions.png")
print(f"   • plots/correlation_heatmap.png")
print(f"   • plots/feature_importance.png")
print(f"   • plots/ensemble_weights.png")

print(f"\n💡 To run the full analysis with a real database:")
print(f"   python3 evaluation/results_analysis_v2.py")
print("\n" + "=" * 80)
