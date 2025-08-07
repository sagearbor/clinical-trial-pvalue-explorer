"""
Statistical Test Factory Pattern Implementation

This module implements a factory pattern for statistical tests to enable easy
addition of new statistical tests in Phase 2. It provides a consistent interface
for all statistical tests and handles routing from study types to appropriate tests.

Key Components:
- StatisticalTest: Abstract base class for all statistical tests
- StatisticalTestFactory: Factory class for test routing and registration
- TwoSampleTTest: Implementation for two-sample t-tests (migrated from statistical_utils.py)

Usage Example:
    factory = StatisticalTestFactory()
    test = factory.get_test("two_sample_t_test")
    p_value, error = test.calculate_p_value(N_total=100, cohens_d=0.5)
    power, error = test.calculate_power(N_total=100, cohens_d=0.5, alpha=0.05)
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod


class StatisticalTest(ABC):
    """
    Abstract base class for all statistical tests.
    
    All statistical tests must implement these methods to ensure consistent
    interface across different test types.
    """
    
    @abstractmethod
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate p-value for the statistical test.
        
        Args:
            **params: Test-specific parameters (e.g., N_total, cohens_d, etc.)
            
        Returns:
            Tuple of (p_value, error_message). If calculation succeeds, 
            returns (p_value, None). If fails, returns (None, error_message).
        """
        pass
    
    @abstractmethod
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate statistical power for the test.
        
        Args:
            **params: Test-specific parameters (e.g., N_total, cohens_d, alpha, etc.)
            
        Returns:
            Tuple of (power, error_message). If calculation succeeds, 
            returns (power, None). If fails, returns (None, error_message).
        """
        pass
    
    @abstractmethod
    def get_required_params(self) -> List[str]:
        """
        Get list of required parameter names for this test.
        
        Returns:
            List of required parameter names as strings
        """
        pass
    
    @abstractmethod
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for this statistical test.
        
        Args:
            **params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_message). If valid, returns (True, None).
            If invalid, returns (False, error_message).
        """
        pass


class TwoSampleTTest(StatisticalTest):
    """
    Two-sample t-test implementation.
    
    Migrated from statistical_utils.py to maintain exact backwards compatibility
    while implementing the new factory pattern interface.
    
    Assumes:
    - Equal group sizes (N_total / 2 per group)
    - Continuous outcome variable
    - Normal distribution (or large enough sample for CLT)
    - Equal variances (pooled variance t-test)
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for two-sample t-test."""
        return ["N_total", "cohens_d"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for two-sample t-test.
        
        Args:
            **params: Must include N_total and cohens_d
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters are present
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        N_total = params.get("N_total")
        cohens_d = params.get("cohens_d")
        
        # Validate N_total
        if not isinstance(N_total, (int, float)) or N_total <= 2:
            return False, "Total N must be a number greater than 2."
        
        # Validate cohens_d
        if not isinstance(cohens_d, (int, float)):
            return False, "Cohen's d must be a number."
        
        N_total_int = int(N_total)
        if N_total_int <= 2:
            return False, "Total N must be greater than 2 for valid degrees of freedom."
        
        n_per_group = N_total_int / 2.0
        if n_per_group <= 1:
            return False, "Sample size per group (N_total / 2) must be greater than 1."
        
        # Validate alpha if provided for power calculations
        alpha = params.get("alpha")
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                return False, "Alpha must be a number between 0 and 1."
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate the two-sided p-value for a two-sample t-test.
        
        Args:
            N_total: Total sample size (assumed equally split into two groups)
            cohens_d: Observed Cohen's d effect size
            
        Returns:
            Tuple of (p_value, error_message). If calculation succeeds, 
            returns (p_value, None). If fails, returns (None, error_message).
        """
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        N_total = params["N_total"]
        cohens_d = params["cohens_d"]
        
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
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
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
        # Set default alpha if not provided
        if "alpha" not in params:
            params = dict(params)  # Create copy to avoid modifying original
            params["alpha"] = 0.05
        
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        N_total = params["N_total"]
        cohens_d = params["cohens_d"]
        alpha = params["alpha"]
        
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


class ChiSquareTest(StatisticalTest):
    """
    Chi-square test implementation for categorical data analysis.
    
    Supports both:
    - Independence tests: Testing association between two categorical variables
    - Goodness-of-fit tests: Testing whether observed frequencies match expected distribution
    
    Assumes:
    - Categorical data in contingency table format
    - Expected frequencies ≥ 5 in each cell (for valid approximation)
    - Independent observations
    - Random sampling
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for chi-square test."""
        return ["contingency_table"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for chi-square test.
        
        Args:
            **params: Must include contingency_table, optionally expected_frequencies and alpha
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters are present
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        contingency_table = params.get("contingency_table")
        
        # Validate contingency_table format
        if not isinstance(contingency_table, (list, tuple, np.ndarray)):
            return False, "contingency_table must be a 2D array, list of lists, or numpy array"
        
        try:
            table = np.array(contingency_table)
        except Exception:
            return False, "contingency_table must be convertible to numpy array"
        
        # Check dimensions
        if table.ndim != 2:
            return False, "contingency_table must be 2-dimensional"
        
        if table.size == 0:
            return False, "contingency_table cannot be empty"
        
        # Check for minimum dimensions (at least 2x2 or 1x3 or 3x1)
        rows, cols = table.shape
        if rows == 1 and cols == 1:
            return False, "contingency_table must have at least 2 rows or 2 columns for chi-square test"
        
        # Chi-square test needs at least 2 categories in total
        if rows == 1 and cols < 2:
            return False, "contingency_table must have at least 2 rows or 2 columns for chi-square test"
        if cols == 1 and rows < 2:
            return False, "contingency_table must have at least 2 rows or 2 columns for chi-square test"
        
        # Check data types and values
        if not np.issubdtype(table.dtype, np.number):
            return False, "All values in contingency_table must be numeric"
        
        if np.any(table < 0):
            return False, "All frequencies must be non-negative"
        
        if not np.all(np.isfinite(table)):
            return False, "All frequencies must be finite numbers"
        
        # Check for sufficient sample size (sum of frequencies)
        total_count = np.sum(table)
        if total_count == 0:
            return False, "Total frequency count must be greater than 0"
        
        # Validate expected_frequencies if provided
        expected_frequencies = params.get("expected_frequencies")
        if expected_frequencies is not None:
            try:
                expected = np.array(expected_frequencies)
            except Exception:
                return False, "expected_frequencies must be convertible to numpy array"
            
            if expected.shape != table.shape:
                return False, "expected_frequencies must have same shape as contingency_table"
            
            if np.any(expected <= 0):
                return False, "All expected frequencies must be positive"
            
            if not np.all(np.isfinite(expected)):
                return False, "All expected frequencies must be finite numbers"
        
        # Validate alpha if provided
        alpha = params.get("alpha")
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                return False, "Alpha must be a number between 0 and 1"
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate p-value for chi-square test.
        
        Args:
            contingency_table: 2D array of observed frequencies
            expected_frequencies: Optional 2D array of expected frequencies
                                 (if not provided, calculated assuming independence)
            
        Returns:
            Tuple of (p_value, error_message). If calculation succeeds, 
            returns (p_value, None). If fails, returns (None, error_message).
        """
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        contingency_table = params["contingency_table"]
        expected_frequencies = params.get("expected_frequencies")
        
        try:
            table = np.array(contingency_table)
            
            if expected_frequencies is not None:
                # Goodness-of-fit test with provided expected frequencies
                expected = np.array(expected_frequencies)
                chi2_stat, p_value = stats.chisquare(table.flatten(), expected.flatten())
            else:
                # Independence test - scipy calculates expected frequencies
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(table)
                
                # Check chi-square assumptions (expected frequencies ≥ 5)
                if np.any(expected < 5):
                    warning_msg = "Warning: Some expected frequencies are less than 5. Chi-square approximation may not be reliable."
                    return p_value, warning_msg
            
            return p_value, None
            
        except ValueError as e:
            return None, f"Invalid data for chi-square test: {str(e)}"
        except Exception as e:
            return None, f"Error during chi-square p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate statistical power for chi-square test.
        
        Args:
            contingency_table: 2D array of observed frequencies (used to calculate effect size)
            alpha: Significance level (default 0.05)
            effect_size: Optional Cramér's V effect size (calculated if not provided)
            total_n: Total sample size (calculated from table if not provided)
            
        Returns:
            Tuple of (power, error_message). If calculation succeeds, 
            returns (power, None). If fails, returns (None, error_message).
        """
        # Set default alpha if not provided
        if "alpha" not in params:
            params = dict(params)  # Create copy to avoid modifying original
            params["alpha"] = 0.05
        
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        contingency_table = params["contingency_table"]
        alpha = params["alpha"]
        
        try:
            table = np.array(contingency_table)
            
            # Calculate or get effect size (Cramér's V)
            effect_size = params.get("effect_size")
            if effect_size is None:
                # Calculate Cramér's V from the contingency table
                chi2_stat, _, dof, _ = stats.chi2_contingency(table)
                total_n = np.sum(table)
                min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
                if min_dim == 0 or total_n == 0:
                    return None, "Cannot calculate effect size: insufficient data"
                cramers_v = np.sqrt(chi2_stat / (total_n * min_dim))
                effect_size = cramers_v
            else:
                # Validate provided effect size
                if not isinstance(effect_size, (int, float)) or effect_size < 0:
                    return None, "Effect size must be a non-negative number"
            
            # Get total sample size
            total_n = params.get("total_n")
            if total_n is None:
                total_n = np.sum(table)
            
            if total_n <= 0:
                return None, "Total sample size must be positive"
            
            # Calculate degrees of freedom
            dof = (table.shape[0] - 1) * (table.shape[1] - 1)
            if dof <= 0:
                return None, "Degrees of freedom must be positive"
            
            # Calculate non-centrality parameter
            ncp = effect_size ** 2 * total_n
            
            # Calculate critical chi-square value
            chi2_crit = stats.chi2.ppf(1 - alpha, dof)
            
            # Calculate power using non-central chi-square distribution
            power = 1 - stats.ncx2.cdf(chi2_crit, dof, ncp)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during chi-square power calculation: {str(e)}"


class OneWayANOVA(StatisticalTest):
    """
    One-way ANOVA implementation for comparing means across multiple groups.
    
    Supports analysis of variance for comparing three or more independent groups
    with continuous outcome variables. Uses F-test to determine if there are
    statistically significant differences between group means.
    
    Assumes:
    - Independent observations between groups
    - Normal distribution within each group (or large enough samples for CLT)
    - Homogeneity of variance across groups (equal variances)
    - Continuous outcome variable
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for one-way ANOVA."""
        return ["groups"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for one-way ANOVA.
        
        Args:
            **params: Must include groups, optionally alpha
            groups can be provided as:
            - List of lists/arrays (each sublist is a group)
            - Multiple group parameters (group1, group2, group3, etc.)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if groups are provided either as 'groups' parameter or as individual group parameters
        groups_param = params.get("groups")
        group_params = {k: v for k, v in params.items() if k.startswith("group") and k[5:].isdigit()}
        
        if groups_param is None and not group_params:
            return False, "Missing required parameter: groups (or individual group1, group2, etc. parameters)"
        
        # Get groups data
        if groups_param is not None:
            groups = groups_param
        else:
            # Convert individual group parameters to groups list
            group_keys = sorted(group_params.keys(), key=lambda x: int(x[5:]))
            groups = [group_params[key] for key in group_keys]
        
        # Validate groups format
        if not isinstance(groups, (list, tuple)):
            return False, "groups must be a list or tuple of group data"
        
        if len(groups) < 2:
            return False, "ANOVA requires at least 2 groups for comparison"
        
        # Validate each group
        for i, group in enumerate(groups):
            if not isinstance(group, (list, tuple, np.ndarray)):
                return False, f"Group {i+1} must be a list, tuple, or numpy array"
            
            try:
                group_array = np.array(group, dtype=float)
            except (ValueError, TypeError):
                return False, f"Group {i+1} must contain numeric values"
            
            if group_array.size == 0:
                return False, f"Group {i+1} cannot be empty"
            
            if not np.all(np.isfinite(group_array)):
                return False, f"Group {i+1} must contain only finite numbers (no NaN or infinity)"
            
            if group_array.size < 2:
                return False, f"Group {i+1} must have at least 2 observations for valid variance calculation"
        
        # Validate alpha if provided
        alpha = params.get("alpha")
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                return False, "Alpha must be a number between 0 and 1"
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate p-value for one-way ANOVA using F-test.
        
        Args:
            groups: List of groups, each containing numerical data
            OR group1, group2, group3, etc.: Individual group parameters
            
        Returns:
            Tuple of (p_value, error_message). If calculation succeeds, 
            returns (p_value, None). If fails, returns (None, error_message).
        """
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        # Get groups data
        groups_param = params.get("groups")
        if groups_param is not None:
            groups = groups_param
        else:
            group_params = {k: v for k, v in params.items() if k.startswith("group") and k[5:].isdigit()}
            group_keys = sorted(group_params.keys(), key=lambda x: int(x[5:]))
            groups = [group_params[key] for key in group_keys]
        
        try:
            # Convert groups to numpy arrays
            group_arrays = [np.array(group, dtype=float) for group in groups]
            
            # Use scipy.stats.f_oneway for F-statistic and p-value
            f_statistic, p_value = stats.f_oneway(*group_arrays)
            
            # Check for potential issues
            if np.isnan(f_statistic) or np.isnan(p_value):
                return None, "F-statistic or p-value calculation resulted in NaN. Check for zero variance within groups."
            
            # Check assumptions and provide warnings
            warning_messages = []
            
            # Check group sizes for balance (informational)
            group_sizes = [len(group) for group in group_arrays]
            if max(group_sizes) / min(group_sizes) > 1.5:
                warning_messages.append("Group sizes are unbalanced, which may affect the robustness of the ANOVA.")
            
            # Check for very small groups
            if min(group_sizes) < 5:
                warning_messages.append("Some groups have very small sample sizes (< 5), which may violate normality assumptions.")
            
            # Combine warnings if any
            warning_msg = None
            if warning_messages:
                warning_msg = "Warning: " + " ".join(warning_messages)
            
            return p_value, warning_msg
            
        except ValueError as e:
            return None, f"Invalid data for ANOVA: {str(e)}"
        except Exception as e:
            return None, f"Error during ANOVA p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate statistical power for one-way ANOVA.
        
        Args:
            groups: List of groups with data (used to calculate effect size and sample sizes)
            alpha: Significance level (default 0.05)
            effect_size: Optional eta-squared effect size (calculated if not provided)
            total_n: Total sample size (calculated from groups if not provided)
            
        Returns:
            Tuple of (power, error_message). If calculation succeeds, 
            returns (power, None). If fails, returns (None, error_message).
        """
        # Set default alpha if not provided
        if "alpha" not in params:
            params = dict(params)  # Create copy to avoid modifying original
            params["alpha"] = 0.05
        
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        # Get groups data
        groups_param = params.get("groups")
        if groups_param is not None:
            groups = groups_param
        else:
            group_params = {k: v for k, v in params.items() if k.startswith("group") and k[5:].isdigit()}
            group_keys = sorted(group_params.keys(), key=lambda x: int(x[5:]))
            groups = [group_params[key] for key in group_keys]
        
        alpha = params["alpha"]
        
        try:
            # Convert groups to numpy arrays
            group_arrays = [np.array(group, dtype=float) for group in groups]
            
            # Calculate or get effect size (eta-squared)
            effect_size = params.get("effect_size")
            if effect_size is None:
                # Calculate eta-squared from the data
                f_statistic, _ = stats.f_oneway(*group_arrays)
                
                # Calculate degrees of freedom
                k = len(group_arrays)  # number of groups
                total_n = sum(len(group) for group in group_arrays)
                df_between = k - 1
                df_within = total_n - k
                
                if df_within <= 0:
                    return None, "Insufficient data: degrees of freedom within groups must be positive"
                
                # Calculate eta-squared: η² = SS_between / SS_total = F * df_between / (F * df_between + df_within)
                if f_statistic == 0:
                    eta_squared = 0.0
                else:
                    eta_squared = (f_statistic * df_between) / (f_statistic * df_between + df_within)
                
                effect_size = eta_squared
            else:
                # Validate provided effect size
                if not isinstance(effect_size, (int, float)) or not (0 <= effect_size <= 1):
                    return None, "Effect size (eta-squared) must be a number between 0 and 1"
            
            # Get sample size information
            total_n = params.get("total_n")
            if total_n is None:
                total_n = sum(len(group) for group in group_arrays)
            
            if total_n <= len(group_arrays):
                return None, "Total sample size must be greater than the number of groups"
            
            # Calculate degrees of freedom
            k = len(group_arrays)  # number of groups
            df_between = k - 1
            df_within = total_n - k
            
            if df_between <= 0 or df_within <= 0:
                return None, "Degrees of freedom must be positive"
            
            # Calculate non-centrality parameter
            # For ANOVA: ncp = (total_n * effect_size) / (1 - effect_size)
            if effect_size >= 1.0:
                return None, "Effect size must be less than 1.0 for valid power calculation"
            
            if effect_size == 0:
                # No effect, power equals alpha (Type I error rate)
                return alpha, "With effect size = 0, power equals alpha (no true effect to detect)."
            
            ncp = (total_n * effect_size) / (1 - effect_size)
            
            # Calculate critical F-value
            f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
            
            # Calculate power using non-central F-distribution
            power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during ANOVA power calculation: {str(e)}"


class CorrelationTest(StatisticalTest):
    """
    Correlation test implementation for analyzing relationships between continuous variables.
    
    Supports both:
    - Pearson correlation: Linear relationships between normally distributed variables
    - Spearman correlation: Monotonic relationships, robust to outliers and non-normal data
    
    Assumes:
    - Continuous or ordinal data for both variables
    - Independent observations (pairs of data points)
    - For Pearson: Linear relationship and bivariate normality
    - For Spearman: Monotonic relationship (can be non-linear)
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for correlation test."""
        return ["x_values", "y_values"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """
        Validate parameters for correlation test.
        
        Args:
            **params: Must include x_values, y_values, optionally correlation_type and alpha
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters are present
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        x_values = params.get("x_values")
        y_values = params.get("y_values")
        
        # Validate x_values
        if not isinstance(x_values, (list, tuple, np.ndarray)):
            return False, "x_values must be a list, tuple, or numpy array"
        
        try:
            x_array = np.array(x_values, dtype=float)
        except (ValueError, TypeError):
            return False, "x_values must contain numeric values"
        
        if x_array.size == 0:
            return False, "x_values cannot be empty"
        
        if not np.all(np.isfinite(x_array)):
            return False, "x_values must contain only finite numbers (no NaN or infinity)"
        
        # Validate y_values
        if not isinstance(y_values, (list, tuple, np.ndarray)):
            return False, "y_values must be a list, tuple, or numpy array"
        
        try:
            y_array = np.array(y_values, dtype=float)
        except (ValueError, TypeError):
            return False, "y_values must contain numeric values"
        
        if y_array.size == 0:
            return False, "y_values cannot be empty"
        
        if not np.all(np.isfinite(y_array)):
            return False, "y_values must contain only finite numbers (no NaN or infinity)"
        
        # Check that arrays have same length
        if len(x_array) != len(y_array):
            return False, "x_values and y_values must have the same length"
        
        # Check minimum sample size for correlation
        if len(x_array) < 3:
            return False, "Correlation requires at least 3 paired observations"
        
        # Validate correlation_type if provided
        correlation_type = params.get("correlation_type", "pearson")
        if correlation_type not in ["pearson", "spearman"]:
            return False, "correlation_type must be either 'pearson' or 'spearman'"
        
        # For Pearson correlation, check for zero variance
        if correlation_type == "pearson":
            if np.var(x_array) == 0:
                return False, "x_values has zero variance - cannot calculate Pearson correlation"
            if np.var(y_array) == 0:
                return False, "y_values has zero variance - cannot calculate Pearson correlation"
        
        # For Spearman correlation, check for tied ranks (all same values)
        if correlation_type == "spearman":
            if len(np.unique(x_array)) == 1:
                return False, "x_values has no variation - cannot calculate Spearman correlation"
            if len(np.unique(y_array)) == 1:
                return False, "y_values has no variation - cannot calculate Spearman correlation"
        
        # Validate alpha if provided
        alpha = params.get("alpha")
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
                return False, "Alpha must be a number between 0 and 1"
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate p-value for correlation test.
        
        Args:
            x_values: Array of values for first variable
            y_values: Array of values for second variable
            correlation_type: 'pearson' or 'spearman' (default: 'pearson')
            
        Returns:
            Tuple of (p_value, error_message). If calculation succeeds, 
            returns (p_value, None). If fails, returns (None, error_message).
        """
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        x_values = params["x_values"]
        y_values = params["y_values"]
        correlation_type = params.get("correlation_type", "pearson")
        
        try:
            x_array = np.array(x_values, dtype=float)
            y_array = np.array(y_values, dtype=float)
            
            if correlation_type == "pearson":
                # Pearson correlation
                correlation_coeff, p_value = stats.pearsonr(x_array, y_array)
            elif correlation_type == "spearman":
                # Spearman correlation
                correlation_coeff, p_value = stats.spearmanr(x_array, y_array)
            else:
                return None, f"Unsupported correlation type: {correlation_type}"
            
            # Check for issues with correlation calculation
            if np.isnan(correlation_coeff) or np.isnan(p_value):
                return None, f"Correlation calculation resulted in NaN. Check data for constant values or other issues."
            
            # Check for perfect correlation (may indicate data issues)
            warning_msg = None
            if abs(correlation_coeff) == 1.0:
                warning_msg = f"Perfect correlation detected (r = {correlation_coeff:.3f}). Verify data is not artificially constructed."
            
            return p_value, warning_msg
            
        except ValueError as e:
            return None, f"Invalid data for correlation test: {str(e)}"
        except Exception as e:
            return None, f"Error during correlation p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate statistical power for correlation test.
        
        Args:
            x_values: Array of values for first variable (used to calculate effect size and sample size)
            y_values: Array of values for second variable
            correlation_type: 'pearson' or 'spearman' (default: 'pearson')
            alpha: Significance level (default 0.05)
            effect_size: Optional correlation coefficient r (calculated if not provided)
            n: Sample size (calculated from data if not provided)
            
        Returns:
            Tuple of (power, error_message). If calculation succeeds, 
            returns (power, None). If fails, returns (None, error_message).
        """
        # Set default alpha if not provided
        if "alpha" not in params:
            params = dict(params)  # Create copy to avoid modifying original
            params["alpha"] = 0.05
        
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        x_values = params["x_values"]
        y_values = params["y_values"]
        correlation_type = params.get("correlation_type", "pearson")
        alpha = params["alpha"]
        
        try:
            x_array = np.array(x_values, dtype=float)
            y_array = np.array(y_values, dtype=float)
            
            # Calculate or get effect size (correlation coefficient)
            effect_size = params.get("effect_size")
            if effect_size is None:
                if correlation_type == "pearson":
                    correlation_coeff, _ = stats.pearsonr(x_array, y_array)
                elif correlation_type == "spearman":
                    correlation_coeff, _ = stats.spearmanr(x_array, y_array)
                else:
                    return None, f"Unsupported correlation type: {correlation_type}"
                
                effect_size = abs(correlation_coeff)  # Use absolute value for power calculation
            else:
                # Validate provided effect size
                if not isinstance(effect_size, (int, float)) or not (0 <= abs(effect_size) <= 1):
                    return None, "Effect size (correlation coefficient) must be between -1 and 1"
                effect_size = abs(effect_size)  # Use absolute value for power calculation
            
            # Get sample size
            n = params.get("n")
            if n is None:
                n = len(x_array)
            
            if n < 3:
                return None, "Sample size must be at least 3 for correlation power analysis"
            
            # Calculate degrees of freedom
            df = n - 2
            
            if df <= 0:
                return None, "Degrees of freedom (n - 2) must be positive"
            
            # Handle edge cases
            if effect_size == 0:
                # No correlation, power equals alpha (Type I error rate)
                return alpha, "With correlation = 0, power equals alpha (no true relationship to detect)."
            
            if effect_size == 1:
                # Perfect correlation - power approaches 1 for any reasonable sample size
                return 0.999, "With perfect correlation (r = 1), power is essentially 1.0."
            
            # Calculate power using Fisher's z-transformation for correlation
            # Transform correlation to Fisher's z
            fisher_z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
            
            # Standard error of Fisher's z
            se_z = 1 / np.sqrt(n - 3)
            
            # Calculate critical z-value for two-tailed test
            z_crit = stats.norm.ppf(1 - alpha / 2)
            
            # Calculate non-centrality parameter
            ncp = fisher_z / se_z
            
            # Calculate power using normal distribution
            # Power = P(|Z| > z_crit | ncp) for two-tailed test
            power = 2 * (1 - stats.norm.cdf(z_crit - abs(ncp)))
            
            # Ensure power is within valid range
            power = min(max(power, 0.0), 1.0)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during correlation power calculation: {str(e)}"


class ANCOVATest(StatisticalTest):
    """
    Analysis of Covariance (ANCOVA) implementation.
    
    ANCOVA extends t-test/ANOVA by adjusting for baseline covariates,
    commonly used in clinical trials to control for baseline measurements.
    Very common in clinical trials (increases power and reduces bias).
    
    Assumes:
    - Continuous outcome and covariate variables
    - Linear relationship between covariate and outcome
    - Homogeneous regression slopes across groups
    - Normal distribution of residuals
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for ANCOVA."""
        return ["N_total", "cohens_d", "covariate_correlation"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters for ANCOVA."""
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        N_total = params.get("N_total")
        cohens_d = params.get("cohens_d")
        covariate_correlation = params.get("covariate_correlation")
        
        # Validate N_total
        if not isinstance(N_total, (int, float)) or N_total <= 4:
            return False, "Total N must be greater than 4 for ANCOVA."
        
        # Validate cohens_d
        if not isinstance(cohens_d, (int, float)):
            return False, "Cohen's d must be a number."
            
        # Validate covariate correlation
        if not isinstance(covariate_correlation, (int, float)) or not (-1 <= covariate_correlation <= 1):
            return False, "Covariate correlation must be between -1 and 1."
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate p-value for ANCOVA."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_total = int(params["N_total"])
            cohens_d = float(params["cohens_d"])
            covariate_correlation = float(params["covariate_correlation"])
            
            n_per_group = N_total / 2.0
            
            # ANCOVA adjusts effect size by reducing residual variance
            # Adjusted effect size = d * sqrt(1 - r^2) where r is covariate correlation
            adjusted_cohens_d = cohens_d / np.sqrt(1 - covariate_correlation**2)
            
            # Calculate t-statistic with adjusted effect size
            pooled_se = np.sqrt(2 / n_per_group)  # Standard error for two groups
            t_stat = adjusted_cohens_d / pooled_se
            
            # Degrees of freedom: N - groups - covariates = N - 2 - 1 = N - 3
            df = N_total - 3
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return p_value, None
            
        except Exception as e:
            return None, f"Error during ANCOVA p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate statistical power for ANCOVA."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_total = int(params["N_total"])
            cohens_d = float(params["cohens_d"])
            covariate_correlation = float(params["covariate_correlation"])
            alpha = float(params.get("alpha", 0.05))
            
            n_per_group = N_total / 2.0
            
            # ANCOVA power benefit from covariate adjustment
            adjusted_cohens_d = cohens_d / np.sqrt(1 - covariate_correlation**2)
            
            # Calculate non-centrality parameter
            ncp = adjusted_cohens_d * np.sqrt(n_per_group / 2)
            
            # Degrees of freedom
            df = N_total - 3
            
            # Critical t-value (two-tailed)
            t_crit = stats.t.ppf(1 - alpha / 2, df)
            
            # Power calculation using non-central t-distribution
            power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
            
            # Ensure power is within valid range
            power = min(max(power, 0.0), 1.0)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during ANCOVA power calculation: {str(e)}"


class FishersExactTest(StatisticalTest):
    """
    Fisher's exact test implementation.
    
    Alternative to chi-square for small sample sizes or sparse contingency tables.
    Provides exact p-values rather than asymptotic approximations.
    
    Commonly used when:
    - Small sample sizes (n < 30)
    - Expected cell counts < 5
    - Rare events in clinical trials
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for Fisher's exact test."""
        return ["contingency_table"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters for Fisher's exact test."""
        if "contingency_table" not in params:
            return False, "Missing required parameter: contingency_table"
        
        table = params["contingency_table"]
        
        # Check if contingency_table is provided and has correct format
        if not isinstance(table, (list, np.ndarray)):
            return False, "Contingency table must be a list or numpy array."
        
        try:
            table = np.array(table)
            if table.shape != (2, 2):
                return False, "Fisher's exact test requires a 2x2 contingency table."
            
            if not np.all(table >= 0):
                return False, "All cell counts must be non-negative."
                
            if np.sum(table) == 0:
                return False, "Total count cannot be zero."
                
        except Exception as e:
            return False, f"Invalid contingency table format: {str(e)}"
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate exact p-value using Fisher's exact test."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            table = np.array(params["contingency_table"])
            
            # Use scipy.stats.fisher_exact for exact p-value
            oddsratio, p_value = stats.fisher_exact(table, alternative='two-sided')
            
            return p_value, None
            
        except Exception as e:
            return None, f"Error during Fisher's exact test calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """
        Calculate power for Fisher's exact test.
        
        Note: Exact power calculation for Fisher's test is complex.
        This provides an approximation based on effect size and sample size.
        """
        # Validate parameters first  
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            table = np.array(params["contingency_table"])
            alpha = float(params.get("alpha", 0.05))
            
            # Calculate total sample size and proportions
            n = np.sum(table)
            
            # For small samples, power is highly dependent on exact configuration
            # This provides a rough approximation
            if n < 10:
                power = 0.1  # Very low power for very small samples
            elif n < 30:
                power = 0.3  # Low power for small samples
            else:
                # Use chi-square approximation for larger samples
                chi2_stat = stats.chi2_contingency(table)[0]
                effect_size = np.sqrt(chi2_stat / n)  # Cramer's V
                
                # Approximate power calculation
                ncp = n * effect_size**2
                power = 1 - stats.chi2.cdf(stats.chi2.ppf(1 - alpha, 1), 1, ncp)
            
            # Ensure power is within valid range
            power = min(max(power, 0.0), 1.0)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during Fisher's exact test power calculation: {str(e)}"


class LogisticRegressionTest(StatisticalTest):
    """
    Logistic regression implementation for binary outcomes.
    
    Used for modeling binary/categorical outcomes in clinical trials:
    - Response vs non-response
    - Cure vs no cure  
    - Adverse event vs no adverse event
    
    Provides odds ratios and confidence intervals.
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for logistic regression."""
        return ["N_total", "baseline_rate", "odds_ratio"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters for logistic regression."""
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        N_total = params.get("N_total")
        baseline_rate = params.get("baseline_rate") 
        odds_ratio = params.get("odds_ratio")
        
        # Validate N_total
        if not isinstance(N_total, (int, float)) or N_total <= 2:
            return False, "Total N must be greater than 2."
        
        # Validate baseline_rate (probability)
        if not isinstance(baseline_rate, (int, float)) or not (0 < baseline_rate < 1):
            return False, "Baseline rate must be between 0 and 1 (exclusive)."
            
        # Validate odds_ratio
        if not isinstance(odds_ratio, (int, float)) or odds_ratio <= 0:
            return False, "Odds ratio must be positive."
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate p-value for logistic regression."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_total = int(params["N_total"])
            baseline_rate = float(params["baseline_rate"])
            odds_ratio = float(params["odds_ratio"])
            
            # Calculate treatment group probability from odds ratio
            baseline_odds = baseline_rate / (1 - baseline_rate)
            treatment_odds = baseline_odds * odds_ratio
            treatment_rate = treatment_odds / (1 + treatment_odds)
            
            # Equal group sizes
            n_per_group = N_total // 2
            
            # Expected events in each group
            control_events = n_per_group * baseline_rate
            treatment_events = n_per_group * treatment_rate
            
            # Simulate contingency table
            table = np.array([
                [control_events, n_per_group - control_events],
                [treatment_events, n_per_group - treatment_events]
            ])
            
            # Use chi-square test for p-value (standard approach for logistic regression)
            chi2_stat, p_value, _, _ = stats.chi2_contingency(table)
            
            return p_value, None
            
        except Exception as e:
            return None, f"Error during logistic regression p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate power for logistic regression."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_total = int(params["N_total"])
            baseline_rate = float(params["baseline_rate"])
            odds_ratio = float(params["odds_ratio"])
            alpha = float(params.get("alpha", 0.05))
            
            n_per_group = N_total // 2
            
            # Calculate treatment group probability
            baseline_odds = baseline_rate / (1 - baseline_rate)
            treatment_odds = baseline_odds * odds_ratio
            treatment_rate = treatment_odds / (1 + treatment_odds)
            
            # Effect size (log odds ratio)
            log_or = np.log(odds_ratio)
            
            # Variance of log odds ratio
            var_log_or = (1/(n_per_group * baseline_rate * (1 - baseline_rate)) + 
                         1/(n_per_group * treatment_rate * (1 - treatment_rate)))
            
            # Z-statistic
            z_stat = abs(log_or) / np.sqrt(var_log_or)
            
            # Critical value
            z_crit = stats.norm.ppf(1 - alpha / 2)
            
            # Power calculation
            power = 2 * (1 - stats.norm.cdf(z_crit - z_stat))
            
            # Ensure power is within valid range
            power = min(max(power, 0.0), 1.0)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during logistic regression power calculation: {str(e)}"


class RepeatedMeasuresANOVA(StatisticalTest):
    """
    Repeated Measures ANOVA implementation.
    
    Used for analyzing longitudinal data with multiple time points:
    - Change over time within subjects
    - Treatment x time interactions  
    - Common in clinical trials with multiple follow-up visits
    
    Accounts for within-subject correlation and reduces error variance.
    """
    
    def get_required_params(self) -> List[str]:
        """Required parameters for repeated measures ANOVA."""
        return ["N_subjects", "n_timepoints", "cohens_f", "correlation_between_measures"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters for repeated measures ANOVA."""
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        N_subjects = params.get("N_subjects")
        n_timepoints = params.get("n_timepoints")
        cohens_f = params.get("cohens_f") 
        correlation = params.get("correlation_between_measures")
        
        # Validate N_subjects
        if not isinstance(N_subjects, (int, float)) or N_subjects <= 2:
            return False, "Number of subjects must be greater than 2."
        
        # Validate n_timepoints
        if not isinstance(n_timepoints, (int, float)) or n_timepoints < 2:
            return False, "Number of timepoints must be at least 2."
            
        # Validate Cohen's f
        if not isinstance(cohens_f, (int, float)) or cohens_f < 0:
            return False, "Cohen's f must be non-negative."
            
        # Validate correlation
        if not isinstance(correlation, (int, float)) or not (-1 <= correlation <= 1):
            return False, "Correlation between measures must be between -1 and 1."
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate p-value for repeated measures ANOVA."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_subjects = int(params["N_subjects"])
            n_timepoints = int(params["n_timepoints"])
            cohens_f = float(params["cohens_f"])
            correlation = float(params["correlation_between_measures"])
            
            # Degrees of freedom
            df_between = n_timepoints - 1  # Between time points
            df_within = (N_subjects - 1) * (n_timepoints - 1)  # Within subjects error
            
            # Greenhouse-Geisser correction for sphericity
            # Simplified assumption: epsilon = 1 / n_timepoints (conservative)
            epsilon = max(1.0 / n_timepoints, 0.5)  # Bounded below by 0.5
            df_between_adj = df_between * epsilon
            df_within_adj = df_within * epsilon
            
            # Effect size to F-statistic conversion
            # F = (cohens_f^2) * N_subjects
            f_stat = (cohens_f**2) * N_subjects
            
            # Adjust for repeated measures correlation (increases sensitivity)
            f_stat = f_stat / (1 - correlation)
            
            # Calculate p-value using F-distribution
            p_value = 1 - stats.f.cdf(f_stat, df_between_adj, df_within_adj)
            
            return p_value, None
            
        except Exception as e:
            return None, f"Error during repeated measures ANOVA p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate power for repeated measures ANOVA."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            N_subjects = int(params["N_subjects"])
            n_timepoints = int(params["n_timepoints"])
            cohens_f = float(params["cohens_f"])
            correlation = float(params["correlation_between_measures"])
            alpha = float(params.get("alpha", 0.05))
            
            # Degrees of freedom
            df_between = n_timepoints - 1
            df_within = (N_subjects - 1) * (n_timepoints - 1)
            
            # Greenhouse-Geisser correction
            epsilon = max(1.0 / n_timepoints, 0.5)
            df_between_adj = df_between * epsilon
            df_within_adj = df_within * epsilon
            
            # Non-centrality parameter
            # Enhanced by repeated measures (reduces error variance)
            ncp = N_subjects * (cohens_f**2) / (1 - correlation)
            
            # Critical F-value
            f_crit = stats.f.ppf(1 - alpha, df_between_adj, df_within_adj)
            
            # Power calculation using non-central F-distribution  
            power = 1 - stats.ncf.cdf(f_crit, df_between_adj, df_within_adj, ncp)
            
            # Ensure power is within valid range
            power = min(max(power, 0.0), 1.0)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during repeated measures ANOVA power calculation: {str(e)}"


class StatisticalTestFactory:
    """
    Factory class for creating and managing statistical tests.
    
    This factory enables easy routing from study types to appropriate statistical
    tests and provides a registry for adding new tests in Phase 2.
    
    Usage:
        factory = StatisticalTestFactory()
        test = factory.get_test("two_sample_t_test")
        available_tests = factory.get_available_tests()
    """
    
    def __init__(self):
        """Initialize factory with default test registrations."""
        self._test_registry: Dict[str, type] = {}
        self._register_default_tests()
    
    def _register_default_tests(self):
        """Register default statistical tests."""
        self.register_test("two_sample_t_test", TwoSampleTTest)
        # Aliases for backwards compatibility and LLM routing
        self.register_test("t_test", TwoSampleTTest)
        self.register_test("two_sample_ttest", TwoSampleTTest)
        self.register_test("independent_samples_t_test", TwoSampleTTest)
        
        # Chi-square test registration
        self.register_test("chi_square", ChiSquareTest)
        self.register_test("chi2", ChiSquareTest)
        self.register_test("categorical_test", ChiSquareTest)
        self.register_test("chi_square_test", ChiSquareTest)
        self.register_test("independence_test", ChiSquareTest)
        self.register_test("goodness_of_fit", ChiSquareTest)
        
        # One-way ANOVA test registration
        self.register_test("one_way_anova", OneWayANOVA)
        self.register_test("anova", OneWayANOVA)
        self.register_test("f_test", OneWayANOVA)
        self.register_test("multiple_groups", OneWayANOVA)
        self.register_test("analysis_of_variance", OneWayANOVA)
        self.register_test("oneway_anova", OneWayANOVA)
        
        # Correlation test registration
        self.register_test("correlation", CorrelationTest)
        self.register_test("pearson", CorrelationTest)
        self.register_test("spearman", CorrelationTest)
        self.register_test("relationship", CorrelationTest)
        self.register_test("correlation_test", CorrelationTest)
        self.register_test("pearson_correlation", CorrelationTest)
        self.register_test("spearman_correlation", CorrelationTest)
        
        # ANCOVA test registration
        self.register_test("ancova", ANCOVATest)
        self.register_test("analysis_of_covariance", ANCOVATest)
        self.register_test("covariance_analysis", ANCOVATest)
        self.register_test("adjusted_comparison", ANCOVATest)
        self.register_test("baseline_adjusted", ANCOVATest)
        
        # Fisher's exact test registration
        self.register_test("fishers_exact", FishersExactTest)
        self.register_test("exact_test", FishersExactTest)
        self.register_test("fisher_exact", FishersExactTest)
        self.register_test("small_sample_categorical", FishersExactTest)
        
        # Logistic regression test registration  
        self.register_test("logistic_regression", LogisticRegressionTest)
        self.register_test("binary_outcome", LogisticRegressionTest)
        self.register_test("odds_ratio", LogisticRegressionTest)
        self.register_test("response_rate", LogisticRegressionTest)
        self.register_test("binary_regression", LogisticRegressionTest)
        
        # Repeated measures ANOVA registration
        self.register_test("repeated_measures_anova", RepeatedMeasuresANOVA)
        self.register_test("longitudinal_anova", RepeatedMeasuresANOVA)
        self.register_test("within_subjects_anova", RepeatedMeasuresANOVA)
        self.register_test("time_series_anova", RepeatedMeasuresANOVA)
        self.register_test("rm_anova", RepeatedMeasuresANOVA)
    
    def register_test(self, test_name: str, test_class: type):
        """
        Register a new statistical test.
        
        Args:
            test_name: Name/identifier for the test
            test_class: Class implementing StatisticalTest interface
            
        Raises:
            ValueError: If test_class doesn't implement StatisticalTest
        """
        if not issubclass(test_class, StatisticalTest):
            raise ValueError(f"Test class {test_class.__name__} must inherit from StatisticalTest")
        
        self._test_registry[test_name.lower()] = test_class
    
    def get_test(self, test_type: str) -> StatisticalTest:
        """
        Get a statistical test instance by type.
        
        Args:
            test_type: Name/identifier of the test type
            
        Returns:
            Instance of the requested statistical test
            
        Raises:
            ValueError: If test_type is not registered or is None/empty
        """
        if test_type is None or test_type == "":
            raise ValueError("Test type cannot be None or empty")
            
        test_type_lower = test_type.lower()
        
        if test_type_lower not in self._test_registry:
            available_tests = list(self._test_registry.keys())
            raise ValueError(f"Unknown test type: {test_type}. Available tests: {available_tests}")
        
        test_class = self._test_registry[test_type_lower]
        return test_class()
    
    def get_available_tests(self) -> List[str]:
        """
        Get list of all available test types.
        
        Returns:
            List of registered test type names
        """
        return list(self._test_registry.keys())
    
    def is_test_available(self, test_type: str) -> bool:
        """
        Check if a test type is available.
        
        Args:
            test_type: Name/identifier of the test type
            
        Returns:
            True if test is available, False otherwise
        """
        return test_type.lower() in self._test_registry


# Global factory instance for easy access
_global_factory = None

def get_factory() -> StatisticalTestFactory:
    """
    Get the global statistical test factory instance.
    
    Returns:
        Global StatisticalTestFactory instance
    """
    global _global_factory
    if _global_factory is None:
        _global_factory = StatisticalTestFactory()
    return _global_factory


# Backwards compatibility functions
def validate_statistical_inputs(N_total: float, cohens_d: float) -> Optional[str]:
    """
    Validate inputs for statistical calculations (backwards compatibility).
    
    Args:
        N_total: Total sample size
        cohens_d: Cohen's d effect size
        
    Returns:
        Error message if validation fails, None if inputs are valid
    """
    test = TwoSampleTTest()
    is_valid, error_msg = test.validate_params(N_total=N_total, cohens_d=cohens_d)
    return error_msg if not is_valid else None


def calculate_p_value_from_N_d(N_total: float, cohens_d: float) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate the two-sided p-value for a two-sample t-test (backwards compatibility).
    
    Args:
        N_total: Total sample size (assumed equally split into two groups)
        cohens_d: Observed Cohen's d effect size
        
    Returns:
        Tuple of (p_value, error_message). If calculation succeeds, 
        returns (p_value, None). If fails, returns (None, error_message).
    """
    test = TwoSampleTTest()
    return test.calculate_p_value(N_total=N_total, cohens_d=cohens_d)


def calculate_power_from_N_d(N_total: float, cohens_d: float, alpha: float = 0.05) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate statistical power for a two-sample t-test (backwards compatibility).
    
    Args:
        N_total: Total sample size
        cohens_d: Cohen's d effect size
        alpha: Significance level (default 0.05)
        
    Returns:
        Tuple of (power, error_message). If calculation succeeds, 
        returns (power, None). If fails, returns (None, error_message).
    """
    test = TwoSampleTTest()
    return test.calculate_power(N_total=N_total, cohens_d=cohens_d, alpha=alpha)