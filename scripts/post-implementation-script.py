#!/usr/bin/env python3
"""
Post-Implementation Review Script for SR 11-7 Compliance
Analyzes model performance after deployment to production
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Post-Implementation Review')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID from registry')
    parser.add_argument('--review-period', type=str, required=True,
                       choices=['30_days', '60_days', '90_days', '180_days'],
                       help='Period to review')
    parser.add_argument('--compare-to-validation', type=str, default='true',
                       choices=['true', 'false'], help='Compare to validation results')
    return parser.parse_args()

def load_production_data(model_id, review_period):
    """Load production performance data for the review period"""
    print(f"Loading production data for model {model_id} over {review_period}...")
    
    # Parse review period
    days = int(review_period.split('_')[0])
    
    # Generate synthetic production data
    np.random.seed(42)
    
    # Create daily performance data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Simulate production metrics with some degradation over time
    base_performance = 0.85
    degradation_rate = 0.0001  # Small daily degradation
    noise_level = 0.02
    
    daily_metrics = []
    for i, date in enumerate(dates):
        # Add weekly seasonality
        weekly_effect = 0.02 * np.sin(2 * np.pi * i / 7)
        
        # Calculate metrics with degradation and noise
        performance = base_performance - (i * degradation_rate) + weekly_effect
        
        metrics = {
            'date': date,
            'accuracy': performance + np.random.normal(0, noise_level),
            'precision': performance + 0.02 + np.random.normal(0, noise_level),
            'recall': performance - 0.05 + np.random.normal(0, noise_level),
            'f1_score': performance - 0.02 + np.random.normal(0, noise_level),
            'auc_roc': performance + 0.05 + np.random.normal(0, noise_level),
            'daily_volume': np.random.poisson(1000) + 500,
            'avg_inference_time_ms': 50 + np.random.exponential(10),
            'error_rate': 0.02 + np.random.exponential(0.01)
        }
        daily_metrics.append(metrics)
    
    production_df = pd.DataFrame(daily_metrics)
    
    # Generate prediction distribution data
    n_predictions = days * 1000
    predictions = np.random.beta(2, 2, n_predictions)
    actuals = (predictions + np.random.normal(0, 0.1, n_predictions) > 0.5).astype(int)
    
    return production_df, predictions, actuals

def load_validation_baseline(model_id):
    """Load validation baseline metrics for comparison"""
    # Simulated validation metrics
    validation_metrics = {
        'accuracy': 0.87,
        'precision': 0.88,
        'recall': 0.82,
        'f1_score': 0.85,
        'auc_roc': 0.91,
        'validation_date': (datetime.now() - timedelta(days=120)).isoformat()
    }
    return validation_metrics

def analyze_performance_trends(production_df):
    """Analyze performance trends over time"""
    # Calculate rolling averages
    window_size = 7
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        production_df[f'{metric}_rolling'] = production_df[metric].rolling(window=window_size).mean()
    
    # Detect significant changes
    metrics_analysis = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        initial_performance = production_df[metric].iloc[:7].mean()
        recent_performance = production_df[metric].iloc[-7:].mean()
        
        change = recent_performance - initial_performance
        pct_change = (change / initial_performance) * 100
        
        # Perform simple trend test
        x = np.arange(len(production_df))
        y = production_df[metric].values
        slope, intercept = np.polyfit(x, y, 1)
        
        metrics_analysis[metric] = {
            'initial': initial_performance,
            'recent': recent_performance,
            'change': change,
            'pct_change': pct_change,
            'trend_slope': slope,
            'trend_direction': 'declining' if slope < -0.0001 else 'stable' if abs(slope) <= 0.0001 else 'improving'
        }
    
    return metrics_analysis

def generate_performance_comparison_plots(production_df, validation_metrics, review_period):
    """Generate comprehensive performance comparison plots"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    days = int(review_period.split('_')[0])
    fig.suptitle(f'Post-Implementation Review - {days} Days in Production', 
                 fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics Over Time
    ax = fig.add_subplot(gs[0, :2])
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for metric, color in zip(metrics, colors):
        ax.plot(production_df['date'], production_df[metric], alpha=0.3, color=color)
        ax.plot(production_df['date'], production_df[f'{metric}_rolling'], 
               label=metric.replace('_', ' ').title(), linewidth=2, color=color)
        
        # Add validation baseline
        if validation_metrics and metric in validation_metrics:
            ax.axhline(y=validation_metrics[metric], color=color, linestyle='--', 
                      alpha=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Metrics Over Time (with 7-day rolling average)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Daily Volume and Error Rate
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = ax1.twinx()
    
    ax1.bar(production_df['date'], production_df['daily_volume'], 
            alpha=0.5, color='skyblue', label='Daily Volume')
    ax2.plot(production_df['date'], production_df['error_rate'], 
            color='red', linewidth=2, label='Error Rate')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Volume', color='skyblue')
    ax2.set_ylabel('Error Rate', color='red')
    ax1.set_title('Volume and Error Rate')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 3. Performance Distribution Comparison
    ax = fig.add_subplot(gs[1, 0])
    
    # Create box plots for each metric
    metrics_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        metrics_data.append(production_df[metric].values)
    
    bp = ax.boxplot(metrics_data, labels=[m.replace('_', ' ').title() for m in metrics], 
                    patch_artist=True)
    
    # Color boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add validation points
    if validation_metrics:
        for i, metric in enumerate(metrics):
            if metric in validation_metrics:
                ax.scatter(i+1, validation_metrics[metric], color='red', s=100, 
                          marker='*', label='Validation' if i == 0 else '')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Performance Distribution vs Validation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Inference Time Analysis
    ax = fig.add_subplot(gs[1, 1])
    
    ax.hist(production_df['avg_inference_time_ms'], bins=30, alpha=0.7, 
            color='green', edgecolor='black')
    ax.axvline(production_df['avg_inference_time_ms'].mean(), color='red', 
              linestyle='--', linewidth=2, label=f"Mean: {production_df['avg_inference_time_ms'].mean():.1f}ms")
    ax.axvline(production_df['avg_inference_time_ms'].quantile(0.95), color='orange', 
              linestyle='--', linewidth=2, label=f"95th %ile: {production_df['avg_inference_time_ms'].quantile(0.95):.1f}ms")
    
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Weekly Performance Heatmap
    ax = fig.add_subplot(gs[1, 2])
    
    # Create weekly aggregation
    production_df['week'] = production_df['date'].dt.isocalendar().week
    weekly_metrics = production_df.groupby('week')[metrics].mean()
    
    sns.heatmap(weekly_metrics.T, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.85, ax=ax, cbar_kws={'label': 'Metric Value'})
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Metric')
    ax.set_title('Weekly Performance Heatmap')
    
    # 6. Cumulative Performance Degradation
    ax = fig.add_subplot(gs[2, :])
    
    for metric in ['accuracy', 'precision', 'recall']:
        if validation_metrics and metric in validation_metrics:
            baseline = validation_metrics[metric]
            cumulative_diff = (production_df[metric] - baseline).cumsum()
            ax.plot(production_df['date'], cumulative_diff, 
                   label=metric.replace('_', ' ').title(), linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax.fill_between(production_df['date'], -5, 5, alpha=0.2, color='gray', 
                   label='Acceptable Range')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Difference from Validation')
    ax.set_title('Cumulative Performance Drift from Validation Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('post_implementation_review.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_analysis_plots(production_df, predictions, actuals):
    """Generate additional detailed analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Detailed Post-Implementation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Prediction Distribution Over Time
    ax = axes[0, 0]
    
    # Sample predictions for different time periods
    n_samples = len(predictions)
    period_size = n_samples // 4
    
    periods = ['Month 1', 'Month 2', 'Month 3', 'Recent']
    for i, period in enumerate(periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else n_samples
        
        ax.hist(predictions[start_idx:end_idx], bins=30, alpha=0.5, 
               label=period, density=True)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error Analysis by Prediction Confidence
    ax = axes[0, 1]
    
    # Bin predictions by confidence
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    prediction_bins = pd.cut(predictions, bins=bins, labels=bin_labels)
    
    # Calculate error rate for each bin
    error_rates = []
    for bin_label in bin_labels:
        mask = prediction_bins == bin_label
        if mask.sum() > 0:
            errors = (predictions[mask] > 0.5) != actuals[mask]
            error_rates.append(errors.mean())
        else:
            error_rates.append(0)
    
    ax.bar(bin_labels, error_rates, color='coral', alpha=0.7)
    ax.set_xlabel('Prediction Confidence Range')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate by Prediction Confidence')
    ax.grid(True, alpha=0.3)
    
    # 3. Daily Performance Variance
    ax = axes[1, 0]
    
    # Calculate daily variance for each metric
    metrics = ['accuracy', 'precision', 'recall']
    daily_variance = production_df[metrics].rolling(window=7).std()
    
    for metric in metrics:
        ax.plot(production_df['date'], daily_variance[metric], 
               label=metric.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('7-Day Rolling Standard Deviation')
    ax.set_title('Performance Stability (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 4. Alert Summary
    ax = axes[1, 1]
    
    # Simulate alert data
    alert_types = ['Performance Warning', 'Data Drift', 'High Latency', 'Error Spike', 'Volume Anomaly']
    alert_counts = np.random.poisson(5, len(alert_types))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(alert_types)))
    ax.pie(alert_counts, labels=alert_types, colors=colors, autopct='%1.0f%%', startangle=90)
    ax.set_title('Production Alerts Distribution')
    
    plt.tight_layout()
    plt.savefig('detailed_post_implementation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_post_implementation_report(model_id, review_period, production_df, 
                                      metrics_analysis, validation_metrics):
    """Generate comprehensive post-implementation report"""
    report = {
        'model_id': model_id,
        'review_period': review_period,
        'review_date': datetime.now().isoformat(),
        'deployment_date': (datetime.now() - timedelta(days=int(review_period.split('_')[0]))).isoformat(),
        'summary_statistics': {
            'total_predictions': int(production_df['daily_volume'].sum()),
            'average_daily_volume': int(production_df['daily_volume'].mean()),
            'uptime_percentage': 99.5,  # Simulated
            'average_inference_time_ms': float(production_df['avg_inference_time_ms'].mean()),
            'p95_inference_time_ms': float(production_df['avg_inference_time_ms'].quantile(0.95))
        },
        'performance_summary': {},
        'trends': metrics_analysis,
        'comparison_to_validation': {},
        'issues_identified': [],
        'recommendations': [],
        'overall_assessment': ''
    }
    
    # Calculate performance summary
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
        report['performance_summary'][metric] = {
            'mean': float(production_df[metric].mean()),
            'std': float(production_df[metric].std()),
            'min': float(production_df[metric].min()),
            'max': float(production_df[metric].max()),
            'recent_7_days': float(production_df[metric].iloc[-7:].mean())
        }
    
    # Compare to validation
    if validation_metrics:
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            if metric in validation_metrics:
                prod_mean = production_df[metric].mean()
                val_baseline = validation_metrics[metric]
                diff = prod_mean - val_baseline
                pct_diff = (diff / val_baseline) * 100
                
                report['comparison_to_validation'][metric] = {
                    'validation_baseline': val_baseline,
                    'production_mean': prod_mean,
                    'difference': diff,
                    'percent_change': pct_diff,
                    'status': 'acceptable' if abs(pct_diff) < 5 else 'degraded' if pct_diff < -5 else 'improved'
                }
    
    # Identify issues
    for metric, analysis in metrics_analysis.items():
        if analysis['trend_direction'] == 'declining' and abs(analysis['pct_change']) > 5:
            report['issues_identified'].append(
                f"{metric} showing declining trend: {analysis['pct_change']:.1f}% decrease"
            )
        
        if report['comparison_to_validation'].get(metric, {}).get('status') == 'degraded':
            report['issues_identified'].append(
                f"{metric} below validation baseline by {abs(report['comparison_to_validation'][metric]['percent_change']):.1f}%"
            )
    
    # Add recommendations based on findings
    if report['issues_identified']:
        report['recommendations'].extend([
            "Investigate root causes of performance degradation",
            "Consider model retraining with recent data",
            "Increase monitoring frequency for degraded metrics",
            "Implement A/B testing with updated model version"
        ])
    else:
        report['recommendations'].append("Continue regular monitoring schedule")
    
    # Determine overall assessment
    degraded_metrics = sum(1 for m in report['comparison_to_validation'].values() 
                          if m.get('status') == 'degraded')
    
    if degraded_metrics == 0:
        report['overall_assessment'] = 'EXCELLENT - Model performing at or above validation levels'
    elif degraded_metrics <= 2:
        report['overall_assessment'] = 'GOOD - Minor degradation observed, within acceptable limits'
    elif degraded_metrics <= 3:
        report['overall_assessment'] = 'FAIR - Moderate degradation requiring attention'
    else:
        report['overall_assessment'] = 'POOR - Significant degradation requiring immediate action'
    
    return report

def generate_pdf_summary(report):
    """Generate executive summary PDF"""
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages('post_implementation_summary.pdf') as pdf:
        # Create summary page
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.95, 'Post-Implementation Review Summary', 
                ha='center', fontsize=18, fontweight='bold')
        fig.text(0.5, 0.90, f"Model ID: {report['model_id']}", 
                ha='center', fontsize=12)
        fig.text(0.5, 0.87, f"Review Period: {report['review_period'].replace('_', ' ')}", 
                ha='center', fontsize=12)
        
        # Overall assessment
        fig.text(0.1, 0.80, 'Overall Assessment:', fontsize=14, fontweight='bold')
        fig.text(0.1, 0.76, report['overall_assessment'], fontsize=12, wrap=True)
        
        # Key metrics
        fig.text(0.1, 0.68, 'Key Performance Metrics:', fontsize=14, fontweight='bold')
        y_pos = 0.64
        for metric, values in list(report['performance_summary'].items())[:5]:
            fig.text(0.1, y_pos, f"• {metric.replace('_', ' ').title()}: {values['mean']:.3f} (±{values['std']:.3f})", 
                    fontsize=11)
            y_pos -= 0.04
        
        # Issues and recommendations
        fig.text(0.1, 0.40, 'Key Findings:', fontsize=14, fontweight='bold')
        y_pos = 0.36
        for issue in report['issues_identified'][:5]:
            fig.text(0.1, y_pos, f"• {issue}", fontsize=11, wrap=True)
            y_pos -= 0.04
        
        fig.text(0.1, 0.20, 'Recommendations:', fontsize=14, fontweight='bold')
        y_pos = 0.16
        for rec in report['recommendations'][:4]:
            fig.text(0.1, y_pos, f"• {rec}", fontsize=11, wrap=True)
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"Starting post-implementation review for model {args.model_id}")
    print(f"Review period: {args.review_period}")
    
    # Load production data
    production_df, predictions, actuals = load_production_data(args.model_id, args.review_period)
    
    # Load validation baseline if requested
    validation_metrics = None
    if args.compare_to_validation == 'true':
        validation_metrics = load_validation_baseline(args.model_id)
        print("Loaded validation baseline for comparison")
    
    # Analyze performance trends
    print("Analyzing performance trends...")
    metrics_analysis = analyze_performance_trends(production_df)
    
    # Generate visualizations
    print("Generating performance comparison plots...")
    generate_performance_comparison_plots(production_df, validation_metrics, args.review_period)
    
    print("Generating detailed analysis plots...")
    generate_detailed_analysis_plots(production_df, predictions, actuals)
    
    # Generate report
    print("Generating post-implementation report...")
    report = generate_post_implementation_report(
        args.model_id, args.review_period, production_df, 
        metrics_analysis, validation_metrics
    )
    
    # Save report
    with open('post_implementation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    with open('post_implementation_summary.txt', 'w') as f:
        f.write("Post-Implementation Review Summary\n")
        f.write("=================================\n\n")
        f.write(f"Model ID: {args.model_id}\n")
        f.write(f"Review Period: {args.review_period.replace('_', ' ')}\n")
        f.write(f"Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Overall Assessment: {report['overall_assessment']}\n\n")
        
        f.write("Performance Summary:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<15} {'Mean':<8} {'Std':<8} {'vs Valid':<10}\n")
        f.write("-" * 50 + "\n")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
            mean_val = report['performance_summary'][metric]['mean']
            std_val = report['performance_summary'][metric]['std']
            
            if metric in report['comparison_to_validation']:
                vs_valid = f"{report['comparison_to_validation'][metric]['percent_change']:+.1f}%"
            else:
                vs_valid = "N/A"
            
            f.write(f"{metric:<15} {mean_val:<8.3f} {std_val:<8.3f} {vs_valid:<10}\n")
        
        f.write("\nKey Findings:\n")
        f.write("-" * 50 + "\n")
        if report['issues_identified']:
            for i, issue in enumerate(report['issues_identified'], 1):
                f.write(f"{i}. {issue}\n")
        else:
            f.write("No significant issues identified.\n")
        
        f.write("\nRecommendations:\n")
        f.write("-" * 50 + "\n")
        for i, rec in enumerate(report['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
    
    # Generate PDF summary
    print("Generating PDF summary...")
    generate_pdf_summary(report)
    
    print("\nPost-implementation review complete. Generated files:")
    print("- post_implementation_review.png")
    print("- detailed_post_implementation_analysis.png")
    print("- post_implementation_report.json")
    print("- post_implementation_summary.txt")
    print("- post_implementation_summary.pdf")
    
    print(f"\n{report['overall_assessment']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())