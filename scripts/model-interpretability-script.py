#!/usr/bin/env python3
"""
Model Interpretability Check Script for SR 11-7 Compliance
Analyzes model complexity and generates interpretability reports
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Model Interpretability Analysis')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID from registry')
    parser.add_argument('--model-type', type=str, required=True, 
                       choices=['linear', 'tree_based', 'neural_network', 'ensemble'],
                       help='Type of model for analysis')
    return parser.parse_args()

def load_model_and_data(model_id):
    """Load model and validation data from Domino Model Registry"""
    # In production, this would connect to Domino Model Registry
    # For demo, we'll create synthetic data
    print(f"Loading model {model_id} from registry...")
    
    # Synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create synthetic model predictions
    weights = np.random.randn(n_features)
    y_pred = 1 / (1 + np.exp(-X @ weights + np.random.randn(n_samples) * 0.1))
    y_true = (y_pred + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)
    
    return X, y_true, y_pred, feature_names, weights

def calculate_model_complexity_metrics(X, model_type):
    """Calculate model complexity metrics"""
    metrics = {
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'feature_correlation': np.mean(np.abs(np.corrcoef(X.T)[np.triu_indices(X.shape[1], k=1)])),
        'timestamp': datetime.now().isoformat()
    }
    
    # Model-specific complexity metrics
    if model_type == 'linear':
        metrics['complexity_score'] = 0.2
        metrics['interpretability_rating'] = 'High'
        metrics['explanation_method'] = 'Coefficients'
    elif model_type == 'tree_based':
        metrics['complexity_score'] = 0.5
        metrics['interpretability_rating'] = 'Medium'
        metrics['explanation_method'] = 'Feature Importance'
    elif model_type == 'neural_network':
        metrics['complexity_score'] = 0.9
        metrics['interpretability_rating'] = 'Low'
        metrics['explanation_method'] = 'SHAP/LIME'
    else:  # ensemble
        metrics['complexity_score'] = 0.7
        metrics['interpretability_rating'] = 'Medium-Low'
        metrics['explanation_method'] = 'Aggregated SHAP'
    
    return metrics

def generate_feature_importance_plot(X, y_pred, feature_names, model_type):
    """Generate feature importance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Interpretability Analysis - {model_type.replace("_", " ").title()}', fontsize=16)
    
    # 1. Feature Importance
    ax = axes[0, 0]
    # Synthetic importance scores
    importance_scores = np.random.exponential(0.5, len(feature_names))
    importance_scores = importance_scores / importance_scores.sum()
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True).tail(10)
    
    ax.barh(importance_df['feature'], importance_df['importance'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Feature Importances')
    ax.grid(True, alpha=0.3)
    
    # 2. Feature Correlation Heatmap
    ax = axes[0, 1]
    corr_matrix = np.corrcoef(X[:, :10].T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                xticklabels=feature_names[:10], yticklabels=feature_names[:10], ax=ax)
    ax.set_title('Feature Correlation Matrix (Top 10)')
    
    # 3. Partial Dependence Plot (for top feature)
    ax = axes[1, 0]
    top_feature_idx = np.argmax(importance_scores)
    feature_values = np.linspace(X[:, top_feature_idx].min(), X[:, top_feature_idx].max(), 100)
    partial_dependence = 0.5 + 0.3 * np.sin(feature_values)  # Synthetic PDP
    
    ax.plot(feature_values, partial_dependence, linewidth=2)
    ax.fill_between(feature_values, partial_dependence - 0.1, partial_dependence + 0.1, alpha=0.3)
    ax.set_xlabel(feature_names[top_feature_idx])
    ax.set_ylabel('Partial Dependence')
    ax.set_title(f'Partial Dependence Plot - {feature_names[top_feature_idx]}')
    ax.grid(True, alpha=0.3)
    
    # 4. Model Performance Distribution
    ax = axes[1, 1]
    ax.hist(y_pred, bins=30, alpha=0.7, label='Predictions', density=True)
    ax.axvline(y_pred.mean(), color='red', linestyle='--', label=f'Mean: {y_pred.mean():.3f}')
    ax.axvline(np.median(y_pred), color='green', linestyle='--', label=f'Median: {np.median(y_pred):.3f}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_interpretability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_shap_analysis(X, y_pred, feature_names):
    """Generate SHAP-based explanations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SHAP-based Model Explanations', fontsize=16)
    
    # Synthetic SHAP values for demonstration
    shap_values = X * np.random.randn(X.shape[1]) * 0.1
    
    # 1. SHAP Summary Plot
    ax = axes[0, 0]
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(feature_importance)[-10:]
    
    for i, idx in enumerate(top_features_idx):
        y_pos = i
        values = shap_values[:100, idx]
        x_values = X[:100, idx]
        scatter = ax.scatter(values, [y_pos] * len(values), c=x_values, 
                           cmap='RdBu', alpha=0.6, s=20)
    
    ax.set_yticks(range(len(top_features_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_features_idx])
    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Summary Plot (Top 10 Features)')
    ax.grid(True, alpha=0.3)
    
    # 2. SHAP Waterfall for Single Prediction
    ax = axes[0, 1]
    sample_idx = 0
    shap_sample = shap_values[sample_idx, :]
    top_features_sample = np.argsort(np.abs(shap_sample))[-10:]
    
    base_value = 0.5
    cumsum = base_value
    positions = []
    
    ax.barh(0, base_value, color='gray', alpha=0.5, label='Base Value')
    
    for i, idx in enumerate(top_features_sample):
        positions.append(cumsum)
        ax.barh(i+1, shap_sample[idx], left=cumsum, 
               color='red' if shap_sample[idx] < 0 else 'blue', alpha=0.7)
        cumsum += shap_sample[idx]
    
    ax.set_yticks(range(len(top_features_sample) + 1))
    ax.set_yticklabels(['Base'] + [feature_names[i] for i in top_features_sample])
    ax.set_xlabel('SHAP Value')
    ax.set_title('SHAP Waterfall - Single Prediction')
    ax.grid(True, alpha=0.3)
    
    # 3. Global Feature Importance via SHAP
    ax = axes[1, 0]
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True).tail(15)
    
    ax.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Global Feature Importance')
    ax.grid(True, alpha=0.3)
    
    # 4. SHAP Dependence Plot
    ax = axes[1, 1]
    top_feature = np.argmax(mean_abs_shap)
    ax.scatter(X[:, top_feature], shap_values[:, top_feature], 
              alpha=0.5, c=X[:, (top_feature + 1) % X.shape[1]], cmap='viridis')
    ax.set_xlabel(feature_names[top_feature])
    ax.set_ylabel('SHAP Value')
    ax.set_title(f'SHAP Dependence - {feature_names[top_feature]}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_interpretability_report(metrics, model_id, model_type):
    """Generate comprehensive interpretability report"""
    report = {
        'model_id': model_id,
        'model_type': model_type,
        'analysis_timestamp': datetime.now().isoformat(),
        'interpretability_metrics': metrics,
        'recommendations': []
    }
    
    # Add recommendations based on complexity
    if metrics['complexity_score'] > 0.7:
        report['recommendations'].extend([
            'High model complexity detected - consider using simpler models for critical decisions',
            'Implement comprehensive SHAP/LIME analysis for all predictions',
            'Establish human review process for high-impact decisions',
            'Document all model assumptions and limitations clearly'
        ])
    elif metrics['complexity_score'] > 0.4:
        report['recommendations'].extend([
            'Medium complexity - ensure feature importance is well documented',
            'Consider partial dependence plots for key features',
            'Implement monitoring for feature drift'
        ])
    else:
        report['recommendations'].extend([
            'Low complexity model - coefficients provide good interpretability',
            'Document linear assumptions and verify they hold in production',
            'Monitor for non-linear patterns in residuals'
        ])
    
    # Add specific guidance for model types
    if model_type == 'neural_network':
        report['additional_guidance'] = {
            'attention_mechanisms': 'Consider implementing attention visualization',
            'layer_analysis': 'Use layer-wise relevance propagation',
            'adversarial': 'Test for adversarial vulnerabilities'
        }
    
    # Write JSON report
    with open('interpretability_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate text summary
    with open('interpretability_summary.txt', 'w') as f:
        f.write(f"Model Interpretability Analysis Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Model ID: {model_id}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Key Findings:\n")
        f.write(f"- Complexity Score: {metrics['complexity_score']:.2f}\n")
        f.write(f"- Interpretability Rating: {metrics['interpretability_rating']}\n")
        f.write(f"- Recommended Explanation Method: {metrics['explanation_method']}\n\n")
        f.write(f"Recommendations:\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"Starting interpretability analysis for model {args.model_id}")
    
    # Load model and data
    X, y_true, y_pred, feature_names, weights = load_model_and_data(args.model_id)
    
    # Calculate complexity metrics
    metrics = calculate_model_complexity_metrics(X, args.model_type)
    
    # Generate visualizations
    print("Generating feature importance analysis...")
    generate_feature_importance_plot(X, y_pred, feature_names, args.model_type)
    
    print("Generating SHAP analysis...")
    generate_shap_analysis(X, y_pred, feature_names)
    
    # Generate report
    print("Generating interpretability report...")
    generate_interpretability_report(metrics, args.model_id, args.model_type)
    
    print("Analysis complete. Generated files:")
    print("- model_interpretability_analysis.png")
    print("- shap_analysis.png")
    print("- interpretability_report.json")
    print("- interpretability_summary.txt")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())