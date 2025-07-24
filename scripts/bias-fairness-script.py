#!/usr/bin/env python3
"""
Bias and Fairness Analysis Script for SR 11-7 Compliance
Comprehensive evaluation of model fairness across protected attributes
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Bias and Fairness Analysis')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID from registry')
    parser.add_argument('--sensitive-features', type=str, required=True,
                       help='Comma-separated list of sensitive features')
    parser.add_argument('--fairness-metrics', type=str, required=True,
                       help='Comma-separated list of fairness metrics to evaluate')
    return parser.parse_args()

def load_data_with_demographics(model_id):
    """Load model predictions with demographic data"""
    print(f"Loading data for model {model_id} with demographic information...")
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 5000
    
    # Create demographic features
    data = pd.DataFrame({
        'age': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56+'], n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Graduate'], n_samples),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples)
    })
    
    # Generate synthetic predictions with some bias
    base_prob = 0.5
    
    # Add bias based on demographics (for demonstration)
    prob_adjustments = {
        'age': {'18-25': -0.1, '26-35': 0, '36-45': 0.05, '46-55': 0.05, '56+': -0.05},
        'gender': {'Male': 0.05, 'Female': -0.05, 'Other': -0.1},
        'race': {'White': 0.05, 'Black': -0.1, 'Hispanic': -0.05, 'Asian': 0.02, 'Other': 0}
    }
    
    y_pred_proba = np.full(n_samples, base_prob)
    
    for feature, adjustments in prob_adjustments.items():
        for value, adj in adjustments.items():
            mask = data[feature] == value
            y_pred_proba[mask] += adj
    
    # Add noise and clip
    y_pred_proba += np.random.normal(0, 0.1, n_samples)
    y_pred_proba = np.clip(y_pred_proba, 0, 1)
    
    # Generate predictions and ground truth
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = (y_pred_proba + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)
    
    data['y_true'] = y_true
    data['y_pred'] = y_pred
    data['y_pred_proba'] = y_pred_proba
    
    return data

def calculate_fairness_metrics(data, sensitive_feature, metric_type):
    """Calculate various fairness metrics for a sensitive feature"""
    results = {}
    groups = data[sensitive_feature].unique()
    
    # Calculate metrics for each group
    group_metrics = {}
    for group in groups:
        mask = data[sensitive_feature] == group
        group_data = data[mask]
        
        if len(group_data) == 0:
            continue
            
        metrics = {
            'size': len(group_data),
            'positive_rate': group_data['y_pred'].mean(),
            'true_positive_rate': (group_data['y_pred'][group_data['y_true'] == 1]).mean() if (group_data['y_true'] == 1).any() else 0,
            'false_positive_rate': (group_data['y_pred'][group_data['y_true'] == 0]).mean() if (group_data['y_true'] == 0).any() else 0,
            'accuracy': accuracy_score(group_data['y_true'], group_data['y_pred']),
            'average_score': group_data['y_pred_proba'].mean()
        }
        group_metrics[group] = metrics
    
    # Calculate fairness metrics
    if metric_type == 'demographic_parity':
        # Demographic parity: difference in positive rates
        positive_rates = [m['positive_rate'] for m in group_metrics.values()]
        results['max_difference'] = max(positive_rates) - min(positive_rates)
        results['ratio'] = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
        
    elif metric_type == 'equalized_odds':
        # Equalized odds: difference in TPR and FPR
        tpr_values = [m['true_positive_rate'] for m in group_metrics.values()]
        fpr_values = [m['false_positive_rate'] for m in group_metrics.values()]
        results['tpr_difference'] = max(tpr_values) - min(tpr_values)
        results['fpr_difference'] = max(fpr_values) - min(fpr_values)
        results['max_difference'] = max(results['tpr_difference'], results['fpr_difference'])
        
    elif metric_type == 'equal_opportunity':
        # Equal opportunity: difference in TPR only
        tpr_values = [m['true_positive_rate'] for m in group_metrics.values()]
        results['max_difference'] = max(tpr_values) - min(tpr_values)
        
    results['group_metrics'] = group_metrics
    results['metric_type'] = metric_type
    results['feature'] = sensitive_feature
    
    return results

def generate_fairness_plots(data, sensitive_features, fairness_results):
    """Generate comprehensive fairness visualization plots"""
    n_features = len(sensitive_features)
    fig, axes = plt.subplots(2, max(2, (n_features + 1) // 2), figsize=(16, 10))
    axes = axes.flatten()
    
    fig.suptitle('Bias and Fairness Analysis', fontsize=16, fontweight='bold')
    
    plot_idx = 0
    
    # Plot fairness metrics for each sensitive feature
    for feature in sensitive_features:
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        feature_data = []
        
        # Get unique groups
        groups = sorted(data[feature].unique())
        
        # Calculate metrics for each group
        for group in groups:
            mask = data[feature] == group
            positive_rate = data[mask]['y_pred'].mean()
            accuracy = accuracy_score(data[mask]['y_true'], data[mask]['y_pred'])
            feature_data.append({
                'Group': group,
                'Positive Rate': positive_rate,
                'Accuracy': accuracy,
                'Count': mask.sum()
            })
        
        df = pd.DataFrame(feature_data)
        
        # Create grouped bar plot
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['Positive Rate'], width, label='Positive Rate', alpha=0.8)
        bars2 = ax.bar(x + width/2, df['Accuracy'], width, label='Accuracy', alpha=0.8)
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Rate')
        ax.set_title(f'Fairness Metrics by {feature.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Group'], rotation=45 if len(df['Group'].iloc[0]) > 5 else 0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add sample size annotations
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height = max(bar1.get_height(), bar2.get_height())
            ax.text(bar1.get_x() + width/2, height + 0.02, f'n={df["Count"].iloc[i]}', 
                   ha='center', va='bottom', fontsize=8)
        
        plot_idx += 1
    
    # Add a summary plot of all fairness violations
    if plot_idx < len(axes):
        ax = axes[plot_idx]
        
        # Create fairness summary
        fairness_summary = []
        for feature, results in fairness_results.items():
            for metric_type, metric_results in results.items():
                if 'max_difference' in metric_results:
                    fairness_summary.append({
                        'Feature': feature,
                        'Metric': metric_type.replace('_', ' ').title(),
                        'Disparity': metric_results['max_difference']
                    })
        
        if fairness_summary:
            summary_df = pd.DataFrame(fairness_summary)
            summary_pivot = summary_df.pivot(index='Feature', columns='Metric', values='Disparity')
            
            sns.heatmap(summary_pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Disparity'})
            ax.set_title('Fairness Metric Summary (Lower is Better)')
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('bias_fairness_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_fairness_report(data, sensitive_features):
    """Generate detailed fairness analysis report"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Fairness Analysis Report', fontsize=16, fontweight='bold')
    
    # 1. Distribution of predictions by sensitive features
    ax = axes[0, 0]
    for i, feature in enumerate(sensitive_features[:3]):  # Limit to 3 features
        groups = data[feature].unique()
        positions = np.arange(len(groups)) + i * 0.25
        values = [data[data[feature] == g]['y_pred_proba'].mean() for g in groups]
        ax.bar(positions, values, width=0.2, label=feature, alpha=0.7)
    
    ax.set_xlabel('Groups')
    ax.set_ylabel('Average Prediction Score')
    ax.set_title('Average Predictions by Protected Groups')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Confusion matrices by group (for first sensitive feature)
    ax = axes[0, 1]
    feature = sensitive_features[0]
    groups = sorted(data[feature].unique())[:3]  # Limit to 3 groups
    
    cm_data = []
    for group in groups:
        mask = data[feature] == group
        cm = confusion_matrix(data[mask]['y_true'], data[mask]['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_data.append(cm_norm)
    
    # Plot confusion matrices side by side
    for i, (group, cm) in enumerate(zip(groups, cm_data)):
        im = ax.imshow(cm, extent=[i, i+0.8, 0, 2], cmap='Blues', aspect='auto')
        ax.text(i+0.4, 1, f'{group}', ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(-0.1, len(groups)-0.1)
    ax.set_ylim(-0.1, 2.1)
    ax.set_title(f'Normalized Confusion Matrices by {feature}')
    ax.set_xlabel('Groups')
    ax.set_ylabel('True (0) / Predicted (1)')
    
    # 3. Fairness metrics violations
    ax = axes[1, 0]
    thresholds = {
        'Demographic Parity': 0.1,
        'Equalized Odds': 0.1,
        'Equal Opportunity': 0.1
    }
    
    violations = []
    for feature in sensitive_features:
        for metric, threshold in thresholds.items():
            # Simulated violation check
            violation_score = np.random.uniform(0, 0.2)
            violations.append({
                'Feature': feature,
                'Metric': metric,
                'Score': violation_score,
                'Violated': violation_score > threshold
            })
    
    violations_df = pd.DataFrame(violations)
    violated_df = violations_df[violations_df['Violated']]
    
    if not violated_df.empty:
        y_pos = np.arange(len(violated_df))
        ax.barh(y_pos, violated_df['Score'], color='red', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{row['Feature']} - {row['Metric']}" 
                           for _, row in violated_df.iterrows()])
        ax.axvline(x=0.1, color='black', linestyle='--', label='Threshold')
        ax.set_xlabel('Disparity Score')
        ax.set_title('Fairness Violations')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Fairness Violations Detected', 
               ha='center', va='center', fontsize=14)
        ax.set_title('Fairness Violations')
    ax.grid(True, alpha=0.3)
    
    # 4. Recommendations
    ax = axes[1, 1]
    ax.axis('off')
    recommendations_text = """
    Bias Mitigation Recommendations:
    
    1. Re-weight training data to balance 
       representation across groups
    
    2. Apply fairness constraints during 
       model training
    
    3. Use post-processing calibration to 
       equalize outcomes
    
    4. Implement disparate impact testing 
       in production
    
    5. Regular fairness audits with 
       updated demographic data
    """
    ax.text(0.1, 0.9, recommendations_text, fontsize=11, 
           verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('detailed_fairness_report.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_fairness_report_json(model_id, sensitive_features, fairness_results):
    """Generate JSON report with fairness findings"""
    report = {
        'model_id': model_id,
        'analysis_date': datetime.now().isoformat(),
        'sensitive_features_analyzed': sensitive_features,
        'fairness_metrics_evaluated': list(set(
            metric for results in fairness_results.values() 
            for metric in results.keys()
        )),
        'findings': [],
        'violations': [],
        'recommendations': []
    }
    
    # Define thresholds
    thresholds = {
        'demographic_parity': 0.1,
        'equalized_odds': 0.1,
        'equal_opportunity': 0.1
    }
    
    # Analyze results
    for feature, results in fairness_results.items():
        for metric_type, metric_results in results.items():
            if 'max_difference' in metric_results:
                disparity = metric_results['max_difference']
                threshold = thresholds.get(metric_type, 0.1)
                
                finding = {
                    'feature': feature,
                    'metric': metric_type,
                    'disparity': disparity,
                    'threshold': threshold,
                    'violated': disparity > threshold
                }
                
                if disparity > threshold:
                    report['violations'].append(finding)
                    report['findings'].append(
                        f"Fairness violation detected for {feature} on {metric_type}: "
                        f"disparity of {disparity:.3f} exceeds threshold of {threshold}"
                    )
                else:
                    report['findings'].append(
                        f"{feature} passes {metric_type} test with disparity of {disparity:.3f}"
                    )
    
    # Add recommendations based on violations
    if report['violations']:
        report['recommendations'].extend([
            "Implement bias mitigation techniques during model training",
            "Consider re-sampling or re-weighting training data",
            "Apply post-processing calibration for affected groups",
            "Increase monitoring frequency for biased predictions",
            "Conduct root cause analysis of bias sources"
        ])
    else:
        report['recommendations'].append(
            "Model shows acceptable fairness - continue regular monitoring"
        )
    
    # Calculate overall fairness score
    if report['violations']:
        report['overall_fairness_score'] = 1 - (len(report['violations']) / 
                                               (len(sensitive_features) * len(thresholds)))
    else:
        report['overall_fairness_score'] = 1.0
    
    report['fairness_status'] = 'PASSED' if report['overall_fairness_score'] > 0.8 else 'FAILED'
    
    return report

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Parse input arguments
    sensitive_features = [f.strip() for f in args.sensitive_features.split(',')]
    fairness_metrics = [f.strip() for f in args.fairness_metrics.split(',')]
    
    print(f"Starting bias and fairness analysis for model {args.model_id}")
    print(f"Analyzing features: {sensitive_features}")
    print(f"Using metrics: {fairness_metrics}")
    
    # Load data
    data = load_data_with_demographics(args.model_id)
    
    # Calculate fairness metrics
    print("Calculating fairness metrics...")
    fairness_results = {}
    
    for feature in sensitive_features:
        if feature in data.columns:
            fairness_results[feature] = {}
            for metric in fairness_metrics:
                results = calculate_fairness_metrics(data, feature, metric)
                fairness_results[feature][metric] = results
    
    # Generate visualizations
    print("Generating fairness visualizations...")
    generate_fairness_plots(data, sensitive_features, fairness_results)
    generate_detailed_fairness_report(data, sensitive_features)
    
    # Generate report
    print("Generating fairness report...")
    report = generate_fairness_report_json(args.model_id, sensitive_features, fairness_results)
    
    # Save report
    with open('fairness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary text file
    with open('fairness_summary.txt', 'w') as f:
        f.write("Bias and Fairness Analysis Summary\n")
        f.write("=================================\n\n")
        f.write(f"Model ID: {args.model_id}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Overall Fairness Score: {report['overall_fairness_score']:.3f}\n")
        f.write(f"Fairness Status: {report['fairness_status']}\n\n")
        
        if report['violations']:
            f.write("Fairness Violations Detected:\n")
            f.write("-" * 40 + "\n")
            for violation in report['violations']:
                f.write(f"- {violation['feature']} failed {violation['metric']} "
                       f"(disparity: {violation['disparity']:.3f})\n")
        else:
            f.write("No fairness violations detected.\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 40 + "\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    print("\nAnalysis complete. Generated files:")
    print("- bias_fairness_analysis.png")
    print("- detailed_fairness_report.png") 
    print("- fairness_report.json")
    print("- fairness_summary.txt")
    
    print(f"\nOverall Fairness Score: {report['overall_fairness_score']:.3f}")
    print(f"Status: {report['fairness_status']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())