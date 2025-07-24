# FRB SR 11-7 / OCC 2011-12 Model Validation Policy for Domino Data Lab

## Overview

This repository contains a comprehensive model validation governance policy designed for Domino Data Lab that ensures compliance with Federal Reserve SR 11-7 and OCC 2011-12 regulatory requirements for model risk management.

## Contents

### 1. Policy File
- **`FRB_SR_11-7_Model_Validation_Policy.yml`** - Main governance policy file defining the complete model validation workflow

### 2. Python Scripts for Automated Validation
Located in the `/scripts` directory:
- **`model_interpretability_check.py`** - Analyzes model complexity and generates interpretability reports
- **`performance_validation.py`** - Comprehensive model performance validation suite
- **`bias_fairness_analysis.py`** - Evaluates model fairness across protected attributes
- **`robustness_testing.py`** - Tests model stability under various conditions
- **`generate_monitoring_config.py`** - Creates monitoring configuration for production
- **`post_implementation_review.py`** - Analyzes model performance after deployment

### 3. Documentation
- **`README.md`** - This file

## Policy Structure

The policy implements a comprehensive 5-stage validation process:

### Stage 1: Model Development and Documentation
- Business purpose definition
- Model design and methodology documentation
- Data quality assessment
- Development testing

### Stage 2: Independent Model Validation
- Conceptual soundness review
- Performance validation
- Bias and fairness assessment
- Robustness testing
- Limitations and risk assessment

### Stage 3: Governance and Compliance Review
- Documentation completeness
- Regulatory compliance assessment
- IT security review
- Legal review (when applicable)

### Stage 4: Production Readiness and Deployment
- Implementation planning
- Monitoring configuration
- Revalidation scheduling

### Stage 5: Post-Implementation Review
- Initial performance review
- Outcomes analysis
- Ongoing monitoring

## Key Features

### Risk-Based Approach
The policy uses a classification system to determine validation requirements based on:
- Model complexity (Low/Medium/High)
- Business impact (Low/Medium/High)
- Resulting in appropriate validation depth and approval requirements

### Automated Validation Scripts
Each script generates comprehensive reports including:
- **Visualizations**: PNG files with charts, graphs, and heatmaps
- **JSON Reports**: Structured findings and metrics
- **Text Summaries**: Human-readable summaries
- **PDF Reports**: Executive-level documentation

### Integration with Domino Features
- Model Registry integration for tracking model versions
- Automated scripted checks for validation
- Role-based access control (RBAC) for approvals
- Conditional visibility of evidence fields
- File upload capabilities for documentation

## Installation and Setup

### Prerequisites
- Domino Data Lab environment
- Python 3.8+ with required packages:
  ```
  numpy
  pandas
  scikit-learn
  matplotlib
  seaborn
  shap
  lime
  pyyaml
  ```

### Setup Instructions

1. **Upload Policy to Domino**
   ```bash
   domino upload FRB_SR_11-7_Model_Validation_Policy.yml
   ```

2. **Configure Scripts Directory**
   - Create `/scripts` directory in your Domino project
   - Upload all Python scripts to this directory
   - Ensure scripts have execute permissions

3. **Configure Approver Groups**
   - Set up the approval groups defined in `Approvers.md`:
     - modeling-review
     - modeling-practitioners
     - modeling-leadership
     - it-review
     - it-leadership
     - infosec-review
     - infosec-leadership
     - legal-review
     - legal-leadership
     - lob-leadership

4. **Create Model Registry Entry**
   - Register your model in Domino Model Registry
   - Define performance metrics thresholds
   - Configure model metadata

## Usage

### Starting a New Validation

1. **Create Governance Item**
   ```
   Navigate to Governance > Create New Item
   Select "FRB SR 11-7 Model Validation Policy"
   ```

2. **Complete Evidence Collection**
   - Fill in all required fields for each stage
   - Upload supporting documentation
   - Register model in Model Registry

3. **Run Automated Validations**
   - Scripts will execute automatically at appropriate stages
   - Review generated reports and visualizations
   - Address any findings or violations

4. **Obtain Approvals**
   - Submit for approval at each stage gate
   - Approvers receive notifications automatically
   - Track approval status in Domino

### Monitoring Production Models

1. **Configure Monitoring**
   ```bash
   python /scripts/generate_monitoring_config.py \
     --model-id YOUR_MODEL_ID \
     --monitoring-frequency daily \
     --alert-channels "email,slack"
   ```

2. **Schedule Post-Implementation Reviews**
   ```bash
   python /scripts/post_implementation_review.py \
     --model-id YOUR_MODEL_ID \
     --review-period 90_days \
     --compare-to-validation true
   ```

## Validation Scripts Reference

### model_interpretability_check.py
**Purpose**: Analyzes model complexity and interpretability
**Outputs**:
- `model_interpretability_analysis.png` - Feature importance and complexity visualizations
- `shap_analysis.png` - SHAP-based explanations
- `interpretability_report.json` - Detailed findings
- `interpretability_summary.txt` - Summary report

### performance_validation.py
**Purpose**: Comprehensive performance testing
**Outputs**:
- `performance_validation_plots.png` - ROC curves, calibration plots, etc.
- `validation_report.json` - Performance metrics and findings
- `validation_summary.txt` - Summary report
- `validation_report.pdf` - PDF report (optional)

### bias_fairness_analysis.py
**Purpose**: Detects bias across protected attributes
**Outputs**:
- `bias_fairness_analysis.png` - Fairness metrics by group
- `detailed_fairness_report.png` - Detailed analysis
- `fairness_report.json` - Violations and metrics
- `fairness_summary.txt` - Summary report

### robustness_testing.py
**Purpose**: Tests model stability and robustness
**Outputs**:
- `robustness_analysis.png` - Stability test results
- `feature_interaction_stability.png` - Feature interaction heatmap
- `robustness_report.json` - Robustness scores
- `robustness_summary.txt` - Summary report

## Compliance Mapping

This policy addresses key SR 11-7 requirements:

| SR 11-7 Section | Policy Implementation |
|-----------------|----------------------|
| Model Development | Stage 1: Comprehensive documentation and testing |
| Model Validation | Stage 2: Independent validation with three core elements |
| Conceptual Soundness | Automated interpretability analysis |
| Ongoing Monitoring | Stage 4: Monitoring configuration and Stage 5: Reviews |
| Outcomes Analysis | Post-implementation review with backtesting |
| Governance | Stage 3: Multi-level approval workflow |
| Documentation | Required at each stage with templates |

## Best Practices

1. **Documentation**
   - Maintain comprehensive documentation throughout
   - Use provided templates for consistency
   - Store all artifacts in Domino for audit trail

2. **Independence**
   - Ensure validators are independent from developers
   - Use separate Domino projects for validation
   - Document any potential conflicts of interest

3. **Monitoring**
   - Set up automated monitoring before production
   - Review alerts and metrics regularly
   - Schedule periodic revalidation

4. **Risk Management**
   - Classify models appropriately by risk
   - Apply additional scrutiny to high-risk models
   - Document all limitations and assumptions

## Support and Maintenance

### Updating the Policy
1. Modify the YAML file as needed
2. Test changes in a development environment
3. Update version control
4. Notify all stakeholders of changes

### Adding Custom Validations
1. Create new Python scripts following the existing pattern
2. Add script references to the policy YAML
3. Define inputs and outputs clearly
4. Test thoroughly before deployment

### Troubleshooting
- Check Domino logs for script execution errors
- Verify all required packages are installed
- Ensure file paths are correct
- Confirm approval groups are properly configured

## Version History

- **v1.0.0** - Initial release with SR 11-7 compliance
- Comprehensive 5-stage validation workflow
- Six automated validation scripts
- Full integration with Domino Data Lab

## License

This policy template is provided as-is for use with Domino Data Lab. Customize as needed for your organization's specific requirements while maintaining compliance with applicable regulations.

## Contact

For questions or support regarding this policy:
- Domino Data Lab Support
- Your organization's Model Risk Management team
- Compliance department for regulatory questions