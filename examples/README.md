# Examples and Demonstrations

This directory contains example scripts and demonstrations for the Universal Study P-Value Explorer.

## Demo Scripts

### `/demos/`

**demo_chi_square.py**
- Demonstrates chi-square test of independence
- Shows both 2x2 and larger contingency tables
- Includes clinical trial and survey analysis examples
- Usage: `python examples/demos/demo_chi_square.py`

**demo_correlation_analysis.py**
- Shows Pearson and Spearman correlation analysis
- Covers linear and monotonic relationships
- Includes real-world scenarios from clinical, healthcare, psychology domains
- Usage: `python examples/demos/demo_correlation_analysis.py`

**demo_one_way_anova.py**
- Demonstrates one-way ANOVA for multiple group comparisons
- Shows effect size calculations (eta-squared)
- Includes clinical trial and educational examples
- Usage: `python examples/demos/demo_one_way_anova.py`

## Running Examples

All demo scripts can be run from the project root directory:

```bash
# Run individual demos
python examples/demos/demo_chi_square.py
python examples/demos/demo_correlation_analysis.py
python examples/demos/demo_one_way_anova.py

# Or run all demos
python -c "
import subprocess
import os
demos = ['demo_chi_square.py', 'demo_correlation_analysis.py', 'demo_one_way_anova.py']
for demo in demos:
    print(f'\\n=== Running {demo} ===')
    subprocess.run(['python', f'examples/demos/{demo}'])
"
```

## What Each Demo Shows

### Statistical Test Implementations
- Factory pattern usage for test discovery
- Parameter validation and error handling
- Effect size calculations and interpretations
- Real-world application scenarios

### Integration Examples
- How to use the statistical tests programmatically
- Proper parameter setup for different study types
- Result interpretation and reporting
- Power analysis and sample size considerations

### Best Practices
- Input data validation
- Error handling strategies
- Result presentation formats
- Documentation and reproducibility

These examples serve as both documentation and testing for the statistical implementations in the Universal Study P-Value Explorer.