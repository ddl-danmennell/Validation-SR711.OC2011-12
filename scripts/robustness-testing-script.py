#!/usr/bin/env python3
"""
Model Robustness and Stability Testing Script for SR 11-7 Compliance
Tests model stability under various perturbations and conditions
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Model Robustness Testing')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID from registry')
    parser.add_argument('--test-types', type=str, required=True,
                       help='Comma-separated list of test types')
    parser.add_argument('--perturbation-level', type=str, required=True,
                       choices=['low', 'medium', 'high'],
                       help='Level of perturbation to apply')
    return parser.parse_args()

def load_model_and_baseline_data(model_id):
    """Load model and baseline test data"""
    print(f"Loading model {model_id} and baseline data...")
    
    # Generate synthetic baseline data
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Generate synthetic model behavior
    weights = np.random.randn(n_features) * 0.5
    y_pred_proba = 1 / (1 + np.exp(-(X @ weights + np.random.randn(n_samples) * 0.1)))
    y_true = (y_pred_proba + np.random.randn(n_samples) * 0.1 > 0.5).astype(int)
    
    baseline_performance = {
        'accuracy': accuracy_score(y_true, (y_pred_proba > 0.5).astype(int)),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }
    
    return X, y_true, y_pred_proba, feature_names, weights, baseline_performance

def apply_feature_perturbation(X, perturbation_level):
    """Apply feature perturbations to test data"""
    X_perturbed = X.copy()
    
    noise_levels = {
        'low': 0.1,
        'medium': 0.3,
        'high': 0.5
    }
    
    noise_scale = noise_levels[perturbation_level]
    
    # Add Gaussian noise
    noise = np.random.randn(*X.shape) * noise_scale
    X_perturbed += noise
    
    return X_perturbed

def test_temporal_stability(X, weights, n_time_periods=12):
    """Test model stability over time with drift"""
    results = []
    
    for t in range(n_time_periods):
        # Simulate temporal drift
        drift_factor = 0.02 * t
        X_drift = X + np.random.randn(*X.shape) * drift_factor
        
        # Add systematic drift to some features
        drift_features = np.random.choice(X.shape[1], size=3, replace=False)
        X_drift[:, drift_features] += 0.1 * t
        
        # Calculate predictions with drift
        y_pred_proba = 1 / (1 + np.exp(-(X_drift @ weights)))
        
        # Add time-based noise
        y_pred_proba += np.random.randn(len(y_pred_proba)) * 0.05 * (1 + t/10)
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        results.append({
            'time_period': t,
            'mean_prediction': y_pred_proba.mean(),
            'std_prediction': y_pred_proba.std(),
            'drift_magnitude': drift_factor
        })
    
    return pd.DataFrame(results)

def test_outlier_sensitivity(X, weights, contamination_rate=0.1):
    """Test model sensitivity to outliers"""
    results = {}
    
    # Test different types of outliers
    outlier_types = ['uniform', 'feature_specific', 'adversarial']
    
    for outlier_type in outlier_types:
        X_contaminated = X.copy()
        n_outliers = int(len(X) * contamination_rate)
        outlier_indices = np.random.choice(len(X), size=n_outliers, replace=False)
        
        if outlier_type == 'uniform':
            # Add uniform outliers
            X_contaminated[outlier_indices] = np.random.uniform(-5, 5, 
                                                               (n_outliers, X.shape[1]))
        elif outlier_type == 'feature_specific':
            # Add outliers to specific features
            outlier_features = np.random.choice(X.shape[1], size=3, replace=False)
            X_contaminated[outlier_indices, outlier_features] = np.random.uniform(-10, 10, 
                                                                                 (n_outliers, 3))
        else:  # adversarial
            # Add adversarial perturbations
            gradient = weights / np.linalg.norm(weights)
            X_contaminated[outlier_indices] += gradient * 2
        
        # Calculate predictions
        y_pred_clean = 1 / (1 + np.exp(-(X @ weights)))
        y_pred_contaminated = 1 / (1 + np.exp(-(X_contaminated @ weights)))
        
        # Calculate sensitivity metrics
        results[outlier_type] = {
            'mean_shift': np.abs(y_pred_contaminated.mean() - y_pred_clean.mean()),
            'max_individual_change': np.max(np.abs(y_pred_contaminated - y_pred_clean)),
            'affected_predictions': np.sum(np.abs(y_pred_contaminated - y_pred_clean) > 0.1) / len(X)
        }
    
    return results

def test_missing_data_robustness(X, weights):
    """Test model robustness to missing data"""
    missing_rates = [0.05, 0.1, 0.2, 0.3, 0.5]
    results = []
    
    for missing_rate in missing_rates:
        X_missing = X.copy()
        
        # Create missing data mask
        mask = np.random.random(X.shape) < missing_rate
        
        # Test different imputation strategies
        strategies = ['mean', 'zero', 'forward_fill']
        
        for strategy in strategies:
            X_imputed = X_missing.copy()
            
            if strategy == 'mean':
                # Mean imputation
                col_means = np.nanmean(X_missing, axis=0)
                for col in range(X.shape[1]):
                    X_imputed[mask[:, col], col] = col_means[col]
            elif strategy == 'zero':
                # Zero imputation
                X_imputed[mask] = 0
            else:  # forward_fill
                # Simple forward fill
                for col in range(X.shape[1]):
                    col_data = X_imputed[:, col]
                    missing_idx = np.where(mask[:, col])[0]
                    for idx in missing_idx:
                        if idx > 0:
                            col_data[idx] = col_data[idx-1]
                        else:
                            col_data[idx] = np.mean(col_data[~mask[:, col]])
            
            # Calculate predictions
            y_pred = 1 / (1 + np.exp(-(X_imputed @ weights)))
            y_pred_clean = 1 / (1 + np.exp(-(X @ weights)))
            
            results.append({
                'missing_rate': missing_rate,
                'imputation_strategy': strategy,
                'mean_absolute_error': np.mean(np.abs(y_pred - y_pred_clean)),
                'correlation': np.corrcoef(y_pred, y_pred_clean)[0, 1]
            })
    
    return pd.DataFrame(results)

def generate_robustness_plots(test_results, perturbation_level):
    """Generate comprehensive robustness visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Model Robustness Analysis - {perturbation_level.title()} Perturbation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Feature Perturbation Impact
    ax = axes[0, 0]
    if 'feature_perturbation' in test_results:
        perturb_results = test_results['feature_perturbation']
        features = list(perturb_results.keys())[:10]  # Top 10 features
        impacts = [perturb_results[f]['impact'] for f in features]
        
        bars = ax.bar(range(len(features)), impacts, color='skyblue')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Prediction Change')
        ax.set_title('Feature Perturbation Impact')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([f'F{i}' for i in range(len(features))])
        
        # Add threshold line
        ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
    
    # 2. Temporal Stability
    ax = axes[0, 1]
    if 'temporal_stability' in test_results:
        temporal_df = test_results['temporal_stability']
        ax.plot(temporal_df['time_period'], temporal_df['mean_prediction'], 
                marker='o', linewidth=2, label='Mean Prediction')
        ax.fill_between(temporal_df['time_period'], 
                       temporal_df['mean_prediction'] - temporal_df['std_prediction'],
                       temporal_df['mean_prediction'] + temporal_df['std_prediction'],
                       alpha=0.3, label='±1 Std Dev')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Prediction Statistics')
        ax.set_title('Temporal Stability Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Outlier Sensitivity
    ax = axes[1, 0]
    if 'outlier_sensitivity' in test_results:
        outlier_results = test_results['outlier_sensitivity']
        outlier_types = list(outlier_results.keys())
        metrics = ['mean_shift', 'max_individual_change', 'affected_predictions']
        
        x = np.arange(len(outlier_types))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [outlier_results[ot][metric] for ot in outlier_types]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Outlier Type')
        ax.set_ylabel('Impact Measure')
        ax.set_title('Outlier Sensitivity Analysis')
        ax.set_xticks(x + width)
        ax.set_xticklabels(outlier_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Missing Data Robustness
    ax = axes[1, 1]
    if 'missing_data' in test_results:
        missing_df = test_results['missing_data']
        
        for strategy in missing_df['imputation_strategy'].unique():
            strategy_data = missing_df[missing_df['imputation_strategy'] == strategy]
            ax.plot(strategy_data['missing_rate'], 
                   1 - strategy_data['mean_absolute_error'],  # Convert to robustness score
                   marker='o', linewidth=2, label=strategy)
        
        ax.set_xlabel('Missing Data Rate')
        ax.set_ylabel('Robustness Score')
        ax.set_title('Missing Data Robustness')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('robustness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_stability_heatmap(X, weights, feature_names):
    """Generate feature interaction stability heatmap"""
    n_features = min(15, len(feature_names))  # Limit to 15 features
    
    # Calculate pairwise feature interaction effects
    interaction_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                # Perturb both features
                X_perturbed = X.copy()
                X_perturbed[:, i] += np.random.randn(len(X)) * 0.5
                X_perturbed[:, j] += np.random.randn(len(X)) * 0.5
                
                # Calculate prediction change
                y_original = 1 / (1 + np.exp(-(X @ weights)))
                y_perturbed = 1 / (1 + np.exp(-(X_perturbed @ weights)))
                
                interaction_matrix[i, j] = np.mean(np.abs(y_perturbed - y_original))
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, 
                xticklabels=feature_names[:n_features],
                yticklabels=feature_names[:n_features],
                cmap='YlOrRd', 
                annot=True, 
                fmt='.3f',
                cbar_kws={'label': 'Prediction Change'})
    plt.title('Feature Interaction Stability Matrix')
    plt.tight_layout()
    plt.savefig('feature_interaction_stability.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_robustness_report(model_id, test_types, test_results, baseline_performance):
    """Generate comprehensive robustness report"""
    report = {
        'model_id': model_id,
        'test_date': datetime.now().isoformat(),
        'test_types': test_types,
        'baseline_performance': baseline_performance,
        'robustness_scores': {},
        'findings': [],
        'recommendations': [],
        'overall_robustness_score': 0
    }
    
    # Calculate robustness scores for each test
    scores = []
    
    if 'feature_perturbation' in test_results:
        # Score based on average impact
        avg_impact = np.mean([v['impact'] for v in test_results['feature_perturbation'].values()])
        score = max(0, 1 - avg_impact)
        scores.append(score)
        report['robustness_scores']['feature_perturbation'] = score
        
        if score < 0.8:
            report['findings'].append('Model shows sensitivity to feature perturbations')
            report['recommendations'].append('Consider regularization or ensemble methods')
    
    if 'temporal_stability' in test_results:
        # Score based on drift over time
        temporal_df = test_results['temporal_stability']
        drift = temporal_df['mean_prediction'].std()
        score = max(0, 1 - drift * 10)  # Scale drift to score
        scores.append(score)
        report['robustness_scores']['temporal_stability'] = score
        
        if score < 0.8:
            report['findings'].append('Model predictions drift significantly over time')
            report['recommendations'].append('Implement drift detection and retraining triggers')
    
    if 'outlier_sensitivity' in test_results:
        # Score based on outlier impact
        max_impact = max(v['mean_shift'] for v in test_results['outlier_sensitivity'].values())
        score = max(0, 1 - max_impact * 5)  # Scale impact to score
        scores.append(score)
        report['robustness_scores']['outlier_sensitivity'] = score
        
        if score < 0.8:
            report['findings'].append('Model vulnerable to outlier contamination')
            report['recommendations'].append('Implement outlier detection and robust training methods')
    
    if 'missing_data' in test_results:
        # Score based on performance with missing data
        missing_df = test_results['missing_data']
        worst_case = missing_df[missing_df['missing_rate'] == 0.3]['correlation'].min()
        score = max(0, worst_case)
        scores.append(score)
        report['robustness_scores']['missing_data'] = score
        
        if score < 0.8:
            report['findings'].append('Model performance degrades with missing data')
            report['recommendations'].append('Develop robust imputation strategies')
    
    # Calculate overall robustness score
    if scores:
        report['overall_robustness_score'] = np.mean(scores)
    
    # Determine overall status
    if report['overall_robustness_score'] >= 0.8:
        report['robustness_status'] = 'ROBUST'
        report['recommendations'].append('Model demonstrates good robustness - continue monitoring')
    elif report['overall_robustness_score'] >= 0.6:
        report['robustness_status'] = 'MODERATE'
        report['recommendations'].append('Model shows moderate robustness - address identified weaknesses')
    else:
        report['robustness_status'] = 'FRAGILE'
        report['recommendations'].append('Model lacks robustness - significant improvements needed')
    
    return report

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Parse test types
    test_types = [t.strip() for t in args.test_types.split(',')]
    
    print(f"Starting robustness testing for model {args.model_id}")
    print(f"Test types: {test_types}")
    print(f"Perturbation level: {args.perturbation_level}")
    
    # Load model and data
    X, y_true, y_pred_proba, feature_names, weights, baseline_performance = \
        load_model_and_baseline_data(args.model_id)
    
    test_results = {}
    
    # Run selected tests
    if 'feature_perturbation' in test_types:
        print("Running feature perturbation tests...")
        X_perturbed = apply_feature_perturbation(X, args.perturbation_level)
        
        # Calculate impact per feature
        feature_impacts = {}
        for i, feature in enumerate(feature_names):
            X_single_perturb = X.copy()
            X_single_perturb[:, i] = X_perturbed[:, i]
            
            y_original = 1 / (1 + np.exp(-(X @ weights)))
            y_perturbed = 1 / (1 + np.exp(-(X_single_perturb @ weights)))
            
            feature_impacts[feature] = {
                'impact': np.mean(np.abs(y_perturbed - y_original)),
                'max_change': np.max(np.abs(y_perturbed - y_original))
            }
        
        test_results['feature_perturbation'] = feature_impacts
    
    if 'temporal_stability' in test_types:
        print("Running temporal stability tests...")
        temporal_results = test_temporal_stability(X, weights)
        test_results['temporal_stability'] = temporal_results
    
    if 'outlier_sensitivity' in test_types:
        print("Running outlier sensitivity tests...")
        outlier_results = test_outlier_sensitivity(X, weights)
        test_results['outlier_sensitivity'] = outlier_results
    
    if 'missing_data' in test_types:
        print("Running missing data robustness tests...")
        missing_results = test_missing_data_robustness(X, weights)
        test_results['missing_data'] = missing_results
    
    # Generate visualizations
    print("Generating robustness visualizations...")
    generate_robustness_plots(test_results, args.perturbation_level)
    generate_stability_heatmap(X, weights, feature_names)
    
    # Generate report
    print("Generating robustness report...")
    report = generate_robustness_report(args.model_id, test_types, test_results, 
                                      baseline_performance)
    
    # Save reports
    with open('robustness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    with open('robustness_summary.txt', 'w') as f:
        f.write("Model Robustness Testing Summary\n")
        f.write("================================\n\n")
        f.write(f"Model ID: {args.model_id}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Perturbation Level: {args.perturbation_level}\n")
        f.write(f"Overall Robustness Score: {report['overall_robustness_score']:.3f}\n")
        f.write(f"Robustness Status: {report['robustness_status']}\n\n")
        
        f.write("Test Results:\n")
        f.write("-" * 40 + "\n")
        for test, score in report['robustness_scores'].items():
            f.write(f"{test.replace('_', ' ').title()}: {score:.3f}\n")
        
        f.write("\nKey Findings:\n")
        f.write("-" * 40 + "\n")
        for i, finding in enumerate(report['findings'], 1):
            f.write(f"{i}. {finding}\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 40 + "\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    print("\nTesting complete. Generated files:")
    print("- robustness_analysis.png")
    print("- feature_interaction_stability.png")
    print("- robustness_report.json")
    print("- robustness_summary.txt")
    
    print(f"\nOverall Robustness Score: {report['overall_robustness_score']:.3f}")
    print(f"Status: {report['robustness_status']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())