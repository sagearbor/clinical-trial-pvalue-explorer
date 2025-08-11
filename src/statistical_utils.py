"""
Statistical calculation utilities for p-value and power analysis.
Consolidates statistical functions used across the application.
Now also includes perform_statistical_calculations for API integration.
"""
import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, Any

def validate_statistical_inputs(N_total: float, cohens_d: float) -> Optional[str]:
    try:
        N_total_val = float(N_total)
        cohens_d_val = float(cohens_d)
    except (TypeError, ValueError):
        return "Invalid input types - N_total and Cohen's d must be numeric."
    if N_total_val <= 2:
        return "Total N must be a number greater than 2."
    if not np.isfinite(cohens_d_val):
        return "Cohen's d must be a finite number."
    N_total_int = int(N_total_val)
    if N_total_int <= 2:
        return "Total N must be greater than 2 for valid degrees of freedom."
    n_per_group = N_total_int / 2.0
    if n_per_group <= 1:
        return "Sample size per group (N_total / 2) must be greater than 1."
    return None

def calculate_p_value_from_N_d(N_total: float, cohens_d: float) -> Tuple[Optional[float], Optional[str]]:
    error_msg = validate_statistical_inputs(N_total, cohens_d)
    if error_msg:
        return None, error_msg
    N_total_int = int(N_total)
    n_per_group = N_total_int / 2.0
    if cohens_d == 0:
        return 1.0, "With Cohen's d = 0, p-value = 1.0 (no effect)."
    try:
        if (n_per_group / 2.0) <= 0:
            return None, "Invalid internal calculation for t-statistic."
        t_statistic = cohens_d * np.sqrt(n_per_group / 2.0)
        df = N_total_int - 2
        if df <= 0:
            return None, "Degrees of freedom (N_total - 2) must be positive."
        p_value = (1 - stats.t.cdf(abs(t_statistic), df)) * 2
        return p_value, None
    except FloatingPointError:
        return None, "Numerical instability in t-statistic calculation."
    except Exception as e:
        return None, f"Error during SciPy p-value calculation: {str(e)}"

def calculate_power_from_N_d(N_total: float, cohens_d: float, alpha: float = 0.05) -> Tuple[Optional[float], Optional[str]]:
    error_msg = validate_statistical_inputs(N_total, cohens_d)
    if error_msg:
        return None, error_msg
    N_total_int = int(N_total)
    n_per_group = N_total_int / 2.0
    df = N_total_int - 2
    try:
        ncp = cohens_d * np.sqrt(n_per_group / 2.0)
        t_crit = stats.t.ppf(1 - alpha / 2, df)
        beta = stats.nct.cdf(t_crit, df, ncp) - stats.nct.cdf(-t_crit, df, ncp)
        power = 1 - beta
        return power, None
    except Exception as e:
        return None, f"Error during power calculation: {str(e)}"

def perform_statistical_calculations(test_type: str, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrapper to integrate with API — currently supports two_sample_t_test."""
    parameters = parameters or {}
    result = {
        "statistical_test_used": test_type,
        "calculation_error": None,
        "calculated_p_value": None,
        "calculated_power": None,
    }
    try:
        if test_type == "two_sample_t_test":
            # Require explicit parameters for backwards compatibility
            if not parameters or ("total_n" not in parameters) or ("cohens_d" not in parameters and "effect_size_value" not in parameters):
                result["calculation_error"] = "Missing required parameters: total_n and cohens_d (or effect_size_value)."
                return result

            total_n = parameters.get("total_n")
            effect = parameters.get("cohens_d", parameters.get("effect_size_value"))

            p_val, p_err = calculate_p_value_from_N_d(total_n, effect)
            power, pw_err = calculate_power_from_N_d(total_n, effect, parameters.get("alpha", 0.05))
            result["calculated_p_value"] = p_val
            result["calculated_power"] = power
            if p_err or pw_err:
                result["calculation_error"] = ", ".join(filter(None, [p_err, pw_err]))
        else:
            result["calculation_error"] = f"Test type '{test_type}' not implemented."
    except Exception as e:
        result["calculation_error"] = str(e)
    return result
