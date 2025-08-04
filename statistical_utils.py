"""
Statistical calculation utilities for p-value and power analysis.
Consolidates statistical functions used across the application.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Optional


def validate_statistical_inputs(N_total: float, cohens_d: float) -> Optional[str]:
    """
    Validate inputs for statistical calculations.
    
    Args:
        N_total: Total sample size
        cohens_d: Cohen's d effect size
        
    Returns:
        Error message if validation fails, None if inputs are valid
    """
    if not isinstance(N_total, (int, float)) or N_total <= 2:
        return "Total N must be a number greater than 2."
    if not isinstance(cohens_d, (int, float)):
        return "Cohen's d must be a number."
    
    N_total_int = int(N_total)
    if N_total_int <= 2:
        return "Total N must be greater than 2 for valid degrees of freedom."
    
    n_per_group = N_total_int / 2.0
    if n_per_group <= 1:
        return "Sample size per group (N_total / 2) must be greater than 1."
    
    return None


def calculate_p_value_from_N_d(N_total: float, cohens_d: float) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate the two-sided p-value for a two-sample t-test.
    
    Args:
        N_total: Total sample size (assumed equally split into two groups)
        cohens_d: Observed Cohen's d effect size
        
    Returns:
        Tuple of (p_value, error_message). If calculation succeeds, 
        returns (p_value, None). If fails, returns (None, error_message).
    """
    # Validate inputs
    error_msg = validate_statistical_inputs(N_total, cohens_d)
    if error_msg:
        return None, error_msg
    
    N_total_int = int(N_total)
    n_per_group = N_total_int / 2.0
    
    # Handle zero effect size
    if cohens_d == 0:
        return 1.0, "With Cohen's d = 0, the t-statistic is 0, leading to a p-value of 1.0 (no effect observed)."
    
    try:
        # Calculate t-statistic
        if (n_per_group / 2.0) <= 0:
            return None, "Invalid internal calculation for t-statistic (sqrt of non-positive)."
        t_statistic = cohens_d * np.sqrt(n_per_group / 2.0)
        
        # Calculate degrees of freedom
        df = N_total_int - 2
        if df <= 0:
            return None, "Degrees of freedom (N_total - 2) must be positive."
        
        # Calculate two-sided p-value
        p_value = (1 - stats.t.cdf(abs(t_statistic), df)) * 2
        
        return p_value, None
        
    except FloatingPointError:
        return None, "Numerical instability in t-statistic calculation."
    except Exception as e:
        return None, f"Error during SciPy p-value calculation: {str(e)}"


def calculate_power_from_N_d(N_total: float, cohens_d: float, alpha: float = 0.05) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate statistical power for a two-sample t-test with equal group sizes.
    
    Args:
        N_total: Total sample size
        cohens_d: Cohen's d effect size
        alpha: Significance level (default 0.05)
        
    Returns:
        Tuple of (power, error_message). If calculation succeeds, 
        returns (power, None). If fails, returns (None, error_message).
    """
    # Validate inputs
    error_msg = validate_statistical_inputs(N_total, cohens_d)
    if error_msg:
        return None, error_msg
    
    N_total_int = int(N_total)
    n_per_group = N_total_int / 2.0
    df = N_total_int - 2
    
    try:
        # Calculate non-centrality parameter
        ncp = cohens_d * np.sqrt(n_per_group / 2.0)
        
        # Calculate critical t-value
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        
        # Calculate power using non-central t-distribution
        beta = stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp)
        power = 1 - beta
        
        return power, None
        
    except Exception as e:
        return None, f"Error during power calculation: {str(e)}"