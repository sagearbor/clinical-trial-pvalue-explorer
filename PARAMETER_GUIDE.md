# Statistical Test Parameter Guide

## Overview
The system now provides appropriate parameter inputs for each statistical test, with an **"🔧 Advanced Parameters"** toggle for statistical experts.

## Parameter Sets by Test Type

### 1. **Mixed Effects Model (MMRM)**
**Basic Parameters:**
- Total N (subjects)
- Number of timepoints
- Number of groups  
- Effect size
- Expected dropout rate

**Advanced Parameters (Stats Experts):**
- Within-subject correlation
- Covariance structure (Unstructured, Compound Symmetry, AR(1), Toeplitz)
- Random slopes inclusion
- Number of covariates

### 2. **Logistic Regression**
**Basic Parameters:**
- Total N
- Number of predictors
- Baseline event rate
- Target odds ratio
- Pseudo R²

**Advanced Parameters:**
- Interaction terms
- Max VIF (multicollinearity)
- Link function (Logit, Probit, Complementary log-log)
- Regularization (None, L1/Lasso, L2/Ridge, Elastic Net)

### 3. **Cox Proportional Hazards**
**Basic Parameters:**
- Total N
- Expected events
- Median follow-up (months)
- Target hazard ratio
- Number of covariates

**Advanced Parameters:**
- Censoring rate
- Proportional hazards assumption check
- Time-varying covariates
- Competing risks

### 4. **Chi-Square Test**
**Basic Parameters:**
- Total N
- Number of groups
- Number of categories
- Cramér's V (effect size)

**Advanced Parameters:**
- Minimum expected frequency
- Yates continuity correction

### 5. **Non-Parametric Tests** (Mann-Whitney, Kruskal-Wallis, Wilcoxon)
**Basic Parameters:**
- Total N
- Number of groups (for Kruskal-Wallis)
- Effect size (probability of superiority)

**Advanced Parameters:**
- Method for handling ties
- Continuity correction

### 6. **Two-Sample T-Test**
**Basic Parameters:**
- Total N
- Cohen's d

**Advanced Parameters:**
- Alpha level
- Equal variance assumption

## User Experience Features

### For Non-Statisticians:
- **Clean interface** by default with only essential parameters
- **AI suggestions** from Azure OpenAI
- **Sensible defaults** for all parameters
- **Clear labels** without jargon

### For Statisticians:
- **"🔧 Advanced Parameters" toggle** reveals expert options
- **All relevant parameters** for each test type
- **Industry-standard options** (covariance structures, link functions, etc.)
- **Fine control** over assumptions and corrections

## How It Works:

1. **Select a test** from the dropdown (23 available)
2. **Basic parameters** appear automatically
3. **Toggle "Advanced Parameters"** if you need expert control
4. **Click "📊 Calculate"** to run the analysis
5. **Results show** P-value, Power, and Sample Size

## Example Use Cases:

### Clinical Trial Statistician:
- Selects "Mixed Effects Model"
- Enables Advanced Parameters
- Sets covariance structure to "AR(1)" for time-series correlation
- Includes random slopes for patient-specific trajectories
- Adjusts for 5 covariates (age, sex, BMI, baseline, site)

### Research Scientist (Non-Stats):
- Enters study description
- AI suggests "Mixed Effects Model"
- Uses default parameters (200 subjects, 4 timepoints)
- Gets immediate power calculation
- No need to understand covariance structures

### Epidemiologist:
- Selects "Cox Proportional Hazards"
- Enables Advanced Parameters
- Accounts for competing risks
- Sets 30% censoring rate
- Includes time-varying covariates

## Implementation Status:
✅ Parameter forms for all major test types
✅ Advanced mode toggle
✅ Appropriate defaults
✅ Clean separation of basic/advanced
✅ Calculations integrated

## Future Enhancements:
- Save parameter presets
- Import parameters from protocol
- Export parameter settings
- Parameter validation rules
- Context-sensitive help