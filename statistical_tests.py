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