#!/usr/bin/env python3
"""
Generate Monitoring Configuration Script for SR 11-7 Compliance
Creates monitoring configuration files for production model deployment
"""

import argparse
import json
import yaml
import sys
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Monitoring Configuration')
    parser.add_argument('--model-id', type=str, required=True, help='Model ID from registry')
    parser.add_argument('--monitoring-frequency', type=str, required=True,
                       choices=['realtime', 'daily', 'weekly', 'monthly'],
                       help='Frequency of monitoring checks')
    parser.add_argument('--alert-channels', type=str, required=True,
                       help='Comma-separated list of alert channels')
    return parser.parse_args()

def generate_monitoring_config(model_id, frequency, alert_channels):
    """Generate comprehensive monitoring configuration"""
    
    # Parse alert channels
    channels = [ch.strip() for ch in alert_channels.split(',')]
    
    # Define monitoring intervals
    intervals = {
        'realtime': '*/5 * * * *',  # Every 5 minutes
        'daily': '0 2 * * *',       # 2 AM daily
        'weekly': '0 2 * * 1',      # 2 AM Monday
        'monthly': '0 2 1 * *'      # 2 AM first day of month
    }
    
    config = {
        'model_id': model_id,
        'created_at': datetime.now().isoformat(),
        'monitoring_frequency': frequency,
        'cron_schedule': intervals[frequency],
        'alert_configuration': {
            'channels': channels,
            'escalation_policy': {
                'level_1': {
                    'threshold': 'warning',
                    'channels': channels[:1],
                    'delay_minutes': 0
                },
                'level_2': {
                    'threshold': 'critical',
                    'channels': channels,
                    'delay_minutes': 15
                },
                'level_3': {
                    'threshold': 'severe',
                    'channels': channels + ['pagerduty'],
                    'delay_minutes': 30
                }
            }
        },
        'metrics': {
            'performance_metrics': {
                'accuracy': {
                    'enabled': True,
                    'warning_threshold': 0.85,
                    'critical_threshold': 0.80,
                    'severe_threshold': 0.75,
                    'aggregation': 'mean',
                    'window': '1h' if frequency == 'realtime' else '24h'
                },
                'precision': {
                    'enabled': True,
                    'warning_threshold': 0.80,
                    'critical_threshold': 0.75,
                    'severe_threshold': 0.70,
                    'aggregation': 'mean',
                    'window': '1h' if frequency == 'realtime' else '24h'
                },
                'recall': {
                    'enabled': True,
                    'warning_threshold': 0.75,
                    'critical_threshold': 0.70,
                    'severe_threshold': 0.65,
                    'aggregation': 'mean',
                    'window': '1h' if frequency == 'realtime' else '24h'
                },
                'f1_score': {
                    'enabled': True,
                    'warning_threshold': 0.78,
                    'critical_threshold': 0.73,
                    'severe_threshold': 0.68,
                    'aggregation': 'mean',
                    'window': '1h' if frequency == 'realtime' else '24h'
                },
                'auc_roc': {
                    'enabled': True,
                    'warning_threshold': 0.70,
                    'critical_threshold': 0.65,
                    'severe_threshold': 0.60,
                    'aggregation': 'mean',
                    'window': '1h' if frequency == 'realtime' else '24h'
                }
            },
            'data_quality_metrics': {
                'missing_feature_rate': {
                    'enabled': True,
                    'warning_threshold': 0.05,
                    'critical_threshold': 0.10,
                    'severe_threshold': 0.20,
                    'aggregation': 'max',
                    'window': '1h'
                },
                'feature_out_of_range_rate': {
                    'enabled': True,
                    'warning_threshold': 0.05,
                    'critical_threshold': 0.10,
                    'severe_threshold': 0.15,
                    'aggregation': 'max',
                    'window': '1h'
                },
                'null_prediction_rate': {
                    'enabled': True,
                    'warning_threshold': 0.01,
                    'critical_threshold': 0.05,
                    'severe_threshold': 0.10,
                    'aggregation': 'max',
                    'window': '1h'
                }
            },
            'drift_metrics': {
                'prediction_drift': {
                    'enabled': True,
                    'method': 'kolmogorov_smirnov',
                    'warning_threshold': 0.1,
                    'critical_threshold': 0.2,
                    'severe_threshold': 0.3,
                    'reference_window': '7d',
                    'comparison_window': '1d'
                },
                'feature_drift': {
                    'enabled': True,
                    'method': 'population_stability_index',
                    'warning_threshold': 0.1,
                    'critical_threshold': 0.25,
                    'severe_threshold': 0.5,
                    'reference_window': '30d',
                    'comparison_window': '1d',
                    'top_k_features': 10
                },
                'concept_drift': {
                    'enabled': True,
                    'method': 'ddm',  # Drift Detection Method
                    'warning_level': 2.0,
                    'drift_level': 3.0,
                    'min_samples': 100
                }
            },
            'business_metrics': {
                'false_positive_cost': {
                    'enabled': True,
                    'warning_threshold': 10000,
                    'critical_threshold': 50000,
                    'severe_threshold': 100000,
                    'aggregation': 'sum',
                    'window': '24h',
                    'unit': 'USD'
                },
                'processing_time_p99': {
                    'enabled': True,
                    'warning_threshold': 500,
                    'critical_threshold': 1000,
                    'severe_threshold': 2000,
                    'aggregation': 'p99',
                    'window': '5m',
                    'unit': 'ms'
                },
                'daily_prediction_volume': {
                    'enabled': True,
                    'warning_threshold_low': 1000,
                    'warning_threshold_high': 100000,
                    'critical_threshold_low': 500,
                    'critical_threshold_high': 200000,
                    'aggregation': 'count',
                    'window': '24h'
                }
            },
            'fairness_metrics': {
                'demographic_parity_difference': {
                    'enabled': True,
                    'protected_attributes': ['age', 'gender', 'race'],
                    'warning_threshold': 0.05,
                    'critical_threshold': 0.10,
                    'severe_threshold': 0.15,
                    'evaluation_frequency': 'weekly'
                },
                'equal_opportunity_difference': {
                    'enabled': True,
                    'protected_attributes': ['age', 'gender', 'race'],
                    'warning_threshold': 0.05,
                    'critical_threshold': 0.10,
                    'severe_threshold': 0.15,
                    'evaluation_frequency': 'weekly'
                }
            }
        },
        'monitoring_dashboards': {
            'executive_dashboard': {
                'url': f'/dashboards/model/{model_id}/executive',
                'refresh_rate': '5m',
                'panels': [
                    'overall_health_score',
                    'key_performance_metrics',
                    'drift_indicators',
                    'business_impact'
                ]
            },
            'technical_dashboard': {
                'url': f'/dashboards/model/{model_id}/technical',
                'refresh_rate': '1m' if frequency == 'realtime' else '5m',
                'panels': [
                    'detailed_performance_metrics',
                    'feature_importance_changes',
                    'prediction_distributions',
                    'error_analysis',
                    'system_performance'
                ]
            },
            'fairness_dashboard': {
                'url': f'/dashboards/model/{model_id}/fairness',
                'refresh_rate': '1h',
                'panels': [
                    'group_performance_comparison',
                    'fairness_metrics_trends',
                    'disparate_impact_analysis'
                ]
            }
        },
        'data_retention': {
            'raw_predictions': '90d',
            'aggregated_metrics': '2y',
            'alert_history': '1y',
            'model_artifacts': 'indefinite'
        },
        'integration_endpoints': {
            'metrics_api': f'/api/v1/models/{model_id}/metrics',
            'predictions_api': f'/api/v1/models/{model_id}/predictions',
            'alerts_webhook': f'/api/v1/models/{model_id}/alerts',
            'health_check': f'/api/v1/models/{model_id}/health'
        }
    }
    
    # Add specific configurations based on frequency
    if frequency == 'realtime':
        config['streaming_configuration'] = {
            'kafka_topic': f'model-{model_id}-predictions',
            'batch_size': 100,
            'buffer_timeout_ms': 1000
        }
    
    return config

def generate_alert_rules(config):
    """Generate Prometheus-style alert rules"""
    rules = {
        'groups': [
            {
                'name': f"model_{config['model_id']}_alerts",
                'interval': '1m',
                'rules': []
            }
        ]
    }
    
    # Generate rules for each metric
    for metric_category, metrics in config['metrics'].items():
        for metric_name, metric_config in metrics.items():
            if isinstance(metric_config, dict) and metric_config.get('enabled'):
                # Warning alert
                if 'warning_threshold' in metric_config:
                    rules['groups'][0]['rules'].append({
                        'alert': f"{metric_name}_warning",
                        'expr': f"model_metric_{metric_name} < {metric_config['warning_threshold']}",
                        'for': '5m',
                        'labels': {
                            'severity': 'warning',
                            'model_id': config['model_id'],
                            'metric_category': metric_category
                        },
                        'annotations': {
                            'summary': f"{metric_name} below warning threshold",
                            'description': f"{metric_name} is {{{{ $value }}}} which is below threshold {metric_config['warning_threshold']}"
                        }
                    })
                
                # Critical alert
                if 'critical_threshold' in metric_config:
                    rules['groups'][0]['rules'].append({
                        'alert': f"{metric_name}_critical",
                        'expr': f"model_metric_{metric_name} < {metric_config['critical_threshold']}",
                        'for': '3m',
                        'labels': {
                            'severity': 'critical',
                            'model_id': config['model_id'],
                            'metric_category': metric_category
                        },
                        'annotations': {
                            'summary': f"{metric_name} below critical threshold",
                            'description': f"{metric_name} is {{{{ $value }}}} which is below threshold {metric_config['critical_threshold']}"
                        }
                    })
    
    return rules

def generate_grafana_dashboard(config):
    """Generate Grafana dashboard configuration"""
    dashboard = {
        'dashboard': {
            'title': f"Model {config['model_id']} Monitoring Dashboard",
            'uid': f"model-{config['model_id']}",
            'tags': ['model-monitoring', 'ml-ops', config['model_id']],
            'timezone': 'browser',
            'panels': [],
            'time': {
                'from': 'now-6h',
                'to': 'now'
            },
            'refresh': '10s' if config['monitoring_frequency'] == 'realtime' else '1m'
        }
    }
    
    # Add panels for different metrics
    panel_id = 1
    row_y = 0
    
    # Performance metrics row
    for i, (metric_name, metric_config) in enumerate(config['metrics']['performance_metrics'].items()):
        if metric_config.get('enabled'):
            dashboard['dashboard']['panels'].append({
                'id': panel_id,
                'type': 'graph',
                'title': metric_name.replace('_', ' ').title(),
                'gridPos': {'h': 8, 'w': 6, 'x': (i % 4) * 6, 'y': row_y + (i // 4) * 8},
                'targets': [{
                    'expr': f'model_metric_{metric_name}{{model_id="{config["model_id"]}"}}',
                    'refId': 'A'
                }],
                'thresholds': [
                    {'value': metric_config.get('critical_threshold', 0), 'color': 'red'},
                    {'value': metric_config.get('warning_threshold', 0), 'color': 'yellow'}
                ]
            })
            panel_id += 1
    
    return dashboard

def main():
    """Main execution function"""
    args = parse_arguments()
    
    print(f"Generating monitoring configuration for model {args.model_id}")
    print(f"Monitoring frequency: {args.monitoring_frequency}")
    print(f"Alert channels: {args.alert_channels}")
    
    # Generate main configuration
    config = generate_monitoring_config(args.model_id, args.monitoring_frequency, 
                                      args.alert_channels)
    
    # Save main configuration as JSON
    with open('monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save as YAML for Kubernetes/Docker deployments
    with open('monitoring_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Generate alert rules
    alert_rules = generate_alert_rules(config)
    with open('alert_rules.yaml', 'w') as f:
        yaml.dump(alert_rules, f, default_flow_style=False)
    
    # Generate Grafana dashboard
    dashboard = generate_grafana_dashboard(config)
    with open('grafana_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("\nMonitoring configuration generated successfully!")
    print("Generated files:")
    print("- monitoring_config.json")
    print("- monitoring_config.yaml")
    print("- alert_rules.yaml")
    print("- grafana_dashboard.json")
    
    print(f"\nCron schedule: {config['cron_schedule']}")
    print(f"Alert channels: {', '.join(config['alert_configuration']['channels'])}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())