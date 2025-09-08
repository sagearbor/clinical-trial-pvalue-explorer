"""Non-parametric statistical tests implementation."""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod


class NonParametricTest(ABC):
    """Abstract base class for non-parametric tests."""
    
    @abstractmethod
    def calculate(self, **params) -> Dict[str, Any]:
        """Calculate test statistic and p-value."""
        pass
    
    @abstractmethod
    def get_required_parameters(self) -> List[str]:
        """Get list of required parameters."""
        pass
    
    @abstractmethod
    def validate_parameters(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate input parameters."""
        pass
    
    @abstractmethod
    def calculate_effect_size(self, **params) -> float:
        """Calculate appropriate effect size measure."""
        pass


class MannWhitneyUTest(NonParametricTest):
    """
    Mann-Whitney U test (Wilcoxon rank-sum test) for two independent samples.
    
    Non-parametric alternative to two-sample t-test.
    """
    
    def calculate(self, **params) -> Dict[str, Any]:
        """
        Calculate Mann-Whitney U test.
        
        Parameters:
            data1: First sample data (or n1 if simulating)
            data2: Second sample data (or n2 if simulating)
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        valid, error = self.validate_parameters(**params)
        if not valid:
            return {"error": error}
        
        if 'data1' in params and 'data2' in params:
            # Actual data provided
            data1 = np.array(params['data1'])
            data2 = np.array(params['data2'])
        else:
            # Simulate data based on parameters
            n1 = params.get('n1', params.get('n_total', 100) // 2)
            n2 = params.get('n2', params.get('n_total', 100) // 2)
            effect_size = params.get('effect_size', 0.5)
            
            # Generate data with specified effect
            data1 = np.random.normal(0, 1, n1)
            data2 = np.random.normal(effect_size, 1, n2)
        
        alternative = params.get('alternative', 'two-sided')
        
        # Perform Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            data1, data2,
            alternative=alternative,
            method='auto'
        )
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        rank_biserial = 1 - (2 * statistic) / (n1 * n2)
        
        # Calculate common language effect size
        cles = statistic / (n1 * n2)
        
        return {
            "test_name": "Mann-Whitney U Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(rank_biserial),
            "rank_biserial": float(rank_biserial),
            "common_language_effect": float(cles),
            "n1": n1,
            "n2": n2,
            "alternative": alternative,
            "interpretation": self._interpret_results(p_value, rank_biserial)
        }
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return ["n_total or (n1, n2) or (data1, data2)"]
    
    def validate_parameters(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        # Check if we have data or sample sizes
        has_data = 'data1' in params and 'data2' in params
        has_sizes = any(k in params for k in ['n_total', 'n1', 'n2'])
        
        if not has_data and not has_sizes:
            return False, "Must provide either data arrays or sample sizes"
        
        if has_data:
            if len(params['data1']) < 2 or len(params['data2']) < 2:
                return False, "Each group must have at least 2 observations"
        
        return True, None
    
    def calculate_effect_size(self, **params) -> float:
        """Calculate rank-biserial correlation as effect size."""
        results = self.calculate(**params)
        return results.get('rank_biserial', 0.0)
    
    def calculate_power(
        self,
        n1: int,
        n2: int,
        effect_size: float,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Calculate statistical power for Mann-Whitney U test.
        
        Uses asymptotic approximation.
        """
        n = n1 + n2
        n_eff = n1 * n2 / n
        
        # Convert effect size to standardized mean difference
        # Approximation: rank-biserial ≈ 2*Φ(d/√2) - 1
        d = effect_size * np.sqrt(2)
        
        # Calculate non-centrality parameter
        nc = d * np.sqrt(n_eff / 2)
        
        # Determine critical value
        if alternative == 'two-sided':
            critical_z = stats.norm.ppf(1 - alpha/2)
        else:
            critical_z = stats.norm.ppf(1 - alpha)
        
        # Calculate power
        power = 1 - stats.norm.cdf(critical_z - nc)
        if alternative == 'two-sided':
            power += stats.norm.cdf(-critical_z - nc)
        
        return float(power)
    
    def _interpret_results(self, p_value: float, effect_size: float) -> str:
        """Interpret test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            magnitude = "negligible"
        elif abs_effect < 0.3:
            magnitude = "small"
        elif abs_effect < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"Result is {sig} (p={p_value:.4f}) with {magnitude} effect size (r={effect_size:.3f})"


class KruskalWallisTest(NonParametricTest):
    """
    Kruskal-Wallis H test for multiple independent samples.
    
    Non-parametric alternative to one-way ANOVA.
    """
    
    def calculate(self, **params) -> Dict[str, Any]:
        """
        Calculate Kruskal-Wallis test.
        
        Parameters:
            groups: List of group data arrays
            n_groups: Number of groups (if simulating)
            n_per_group: Sample size per group (if simulating)
            
        Returns:
            Dictionary with test results
        """
        valid, error = self.validate_parameters(**params)
        if not valid:
            return {"error": error}
        
        if 'groups' in params:
            # Actual data provided
            groups = [np.array(g) for g in params['groups']]
        else:
            # Simulate data
            n_groups = params.get('n_groups', 3)
            n_per_group = params.get('n_per_group', params.get('n_total', 90) // n_groups)
            effect_size = params.get('effect_size', 0.25)
            
            # Generate data with increasing means
            groups = []
            for i in range(n_groups):
                mean = i * effect_size
                groups.append(np.random.normal(mean, 1, n_per_group))
        
        # Perform Kruskal-Wallis test
        statistic, p_value = stats.kruskal(*groups)
        
        # Calculate epsilon-squared effect size
        n_total = sum(len(g) for g in groups)
        epsilon_squared = (statistic - len(groups) + 1) / (n_total - len(groups))
        epsilon_squared = max(0, epsilon_squared)  # Ensure non-negative
        
        # Post-hoc analysis suggestion
        post_hoc = "Consider Dunn's test for pairwise comparisons" if p_value < 0.05 else None
        
        return {
            "test_name": "Kruskal-Wallis Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(epsilon_squared),
            "epsilon_squared": float(epsilon_squared),
            "n_groups": len(groups),
            "n_total": n_total,
            "df": len(groups) - 1,
            "post_hoc_suggestion": post_hoc,
            "interpretation": self._interpret_results(p_value, epsilon_squared, len(groups))
        }
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return ["groups or (n_groups and n_total/n_per_group)"]
    
    def validate_parameters(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        if 'groups' in params:
            groups = params['groups']
            if len(groups) < 2:
                return False, "Need at least 2 groups"
            if any(len(g) < 2 for g in groups):
                return False, "Each group must have at least 2 observations"
        else:
            if 'n_groups' not in params:
                return False, "Must specify number of groups"
            if params['n_groups'] < 2:
                return False, "Need at least 2 groups"
        
        return True, None
    
    def calculate_effect_size(self, **params) -> float:
        """Calculate epsilon-squared as effect size."""
        results = self.calculate(**params)
        return results.get('epsilon_squared', 0.0)
    
    def calculate_power(
        self,
        n_groups: int,
        n_per_group: int,
        effect_size: float,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate statistical power for Kruskal-Wallis test.
        
        Uses simulation-based approximation.
        """
        # Simplified power calculation using chi-square approximation
        n_total = n_groups * n_per_group
        df = n_groups - 1
        
        # Convert effect size to non-centrality parameter
        # Approximation based on epsilon-squared
        nc = effect_size * n_total
        
        # Critical value from chi-square distribution
        critical_chi2 = stats.chi2.ppf(1 - alpha, df)
        
        # Power from non-central chi-square
        power = 1 - stats.ncx2.cdf(critical_chi2, df, nc)
        
        return float(power)
    
    def _interpret_results(self, p_value: float, effect_size: float, n_groups: int) -> str:
        """Interpret test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        if effect_size < 0.01:
            magnitude = "negligible"
        elif effect_size < 0.06:
            magnitude = "small"
        elif effect_size < 0.14:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return f"Differences among {n_groups} groups are {sig} (p={p_value:.4f}) with {magnitude} effect (ε²={effect_size:.3f})"


class WilcoxonSignedRankTest(NonParametricTest):
    """
    Wilcoxon signed-rank test for paired samples.
    
    Non-parametric alternative to paired t-test.
    """
    
    def calculate(self, **params) -> Dict[str, Any]:
        """
        Calculate Wilcoxon signed-rank test.
        
        Parameters:
            data1: First paired sample
            data2: Second paired sample (or differences if only one provided)
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        valid, error = self.validate_parameters(**params)
        if not valid:
            return {"error": error}
        
        if 'differences' in params:
            differences = np.array(params['differences'])
        elif 'data1' in params and 'data2' in params:
            data1 = np.array(params['data1'])
            data2 = np.array(params['data2'])
            differences = data1 - data2
        else:
            # Simulate paired data
            n = params.get('n', 50)
            effect_size = params.get('effect_size', 0.5)
            differences = np.random.normal(effect_size, 1, n)
        
        alternative = params.get('alternative', 'two-sided')
        
        # Perform Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(
            differences,
            alternative=alternative,
            mode='auto'
        )
        
        # Calculate effect size (matched-pairs rank-biserial)
        n = len(differences)
        rank_sum = statistic
        max_rank_sum = n * (n + 1) / 2
        rank_biserial = 1 - (2 * rank_sum) / max_rank_sum
        
        # Calculate median difference
        median_diff = np.median(differences)
        
        # Hodges-Lehmann estimator (median of pairwise averages)
        hodges_lehmann = np.median([(differences[i] + differences[j])/2 
                                    for i in range(n) for j in range(i, n)])
        
        return {
            "test_name": "Wilcoxon Signed-Rank Test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(rank_biserial),
            "rank_biserial": float(rank_biserial),
            "median_difference": float(median_diff),
            "hodges_lehmann_estimate": float(hodges_lehmann),
            "n_pairs": n,
            "alternative": alternative,
            "interpretation": self._interpret_results(p_value, rank_biserial, median_diff)
        }
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return ["differences or (data1, data2) or n"]
    
    def validate_parameters(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        has_differences = 'differences' in params
        has_paired_data = 'data1' in params and 'data2' in params
        has_n = 'n' in params
        
        if not (has_differences or has_paired_data or has_n):
            return False, "Must provide differences, paired data, or sample size"
        
        if has_paired_data:
            if len(params['data1']) != len(params['data2']):
                return False, "Paired samples must have equal length"
            if len(params['data1']) < 3:
                return False, "Need at least 3 pairs"
        
        if has_differences:
            if len(params['differences']) < 3:
                return False, "Need at least 3 differences"
        
        return True, None
    
    def calculate_effect_size(self, **params) -> float:
        """Calculate rank-biserial correlation as effect size."""
        results = self.calculate(**params)
        return results.get('rank_biserial', 0.0)
    
    def calculate_power(
        self,
        n: int,
        effect_size: float,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Calculate statistical power for Wilcoxon signed-rank test.
        
        Uses normal approximation.
        """
        # Convert effect size to standardized mean difference
        d = effect_size
        
        # Calculate non-centrality parameter
        nc = d * np.sqrt(n)
        
        # Determine critical value
        if alternative == 'two-sided':
            critical_z = stats.norm.ppf(1 - alpha/2)
        else:
            critical_z = stats.norm.ppf(1 - alpha)
        
        # Calculate power
        power = 1 - stats.norm.cdf(critical_z - nc)
        if alternative == 'two-sided':
            power += stats.norm.cdf(-critical_z - nc)
        
        return float(power)
    
    def _interpret_results(self, p_value: float, effect_size: float, median_diff: float) -> str:
        """Interpret test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        abs_effect = abs(effect_size)
        if abs_effect < 0.1:
            magnitude = "negligible"
        elif abs_effect < 0.3:
            magnitude = "small"
        elif abs_effect < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        direction = "increase" if median_diff > 0 else "decrease"
        
        return f"Paired differences are {sig} (p={p_value:.4f}) with {magnitude} effect (r={effect_size:.3f}), median {direction} of {abs(median_diff):.3f}"


class SpearmanCorrelationTest(NonParametricTest):
    """
    Spearman's rank correlation test.
    
    Non-parametric alternative to Pearson correlation.
    """
    
    def calculate(self, **params) -> Dict[str, Any]:
        """
        Calculate Spearman's rank correlation.
        
        Parameters:
            x: First variable data
            y: Second variable data
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dictionary with test results
        """
        valid, error = self.validate_parameters(**params)
        if not valid:
            return {"error": error}
        
        if 'x' in params and 'y' in params:
            x = np.array(params['x'])
            y = np.array(params['y'])
        else:
            # Simulate correlated data
            n = params.get('n', 100)
            true_rho = params.get('correlation', 0.5)
            
            # Generate correlated ranks
            mean = [0, 0]
            cov = [[1, true_rho], [true_rho, 1]]
            data = np.random.multivariate_normal(mean, cov, n)
            x = stats.rankdata(data[:, 0])
            y = stats.rankdata(data[:, 1])
        
        alternative = params.get('alternative', 'two-sided')
        
        # Calculate Spearman correlation
        rho, p_value = stats.spearmanr(x, y, alternative=alternative)
        
        # Calculate confidence interval (Fisher transformation)
        n = len(x)
        z = np.arctanh(rho)
        se = 1 / np.sqrt(n - 3)
        z_ci_lower = z - 1.96 * se
        z_ci_upper = z + 1.96 * se
        ci_lower = np.tanh(z_ci_lower)
        ci_upper = np.tanh(z_ci_upper)
        
        # Calculate coefficient of determination
        r_squared = rho ** 2
        
        return {
            "test_name": "Spearman Rank Correlation",
            "statistic": float(rho),
            "p_value": float(p_value),
            "correlation": float(rho),
            "r_squared": float(r_squared),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n": n,
            "alternative": alternative,
            "interpretation": self._interpret_results(p_value, rho, r_squared)
        }
    
    def get_required_parameters(self) -> List[str]:
        """Get required parameters."""
        return ["(x, y) or (n and correlation)"]
    
    def validate_parameters(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate parameters."""
        has_data = 'x' in params and 'y' in params
        has_simulation = 'n' in params
        
        if not (has_data or has_simulation):
            return False, "Must provide data (x, y) or sample size n"
        
        if has_data:
            if len(params['x']) != len(params['y']):
                return False, "x and y must have equal length"
            if len(params['x']) < 3:
                return False, "Need at least 3 observations"
        
        return True, None
    
    def calculate_effect_size(self, **params) -> float:
        """Return correlation coefficient as effect size."""
        results = self.calculate(**params)
        return abs(results.get('correlation', 0.0))
    
    def calculate_power(
        self,
        n: int,
        correlation: float,
        alpha: float = 0.05,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Calculate statistical power for Spearman correlation test.
        
        Uses Fisher transformation approximation.
        """
        # Fisher transformation
        z_r = np.arctanh(correlation)
        
        # Standard error
        se = 1 / np.sqrt(n - 3)
        
        # Non-centrality parameter
        nc = z_r / se
        
        # Critical value
        if alternative == 'two-sided':
            critical_z = stats.norm.ppf(1 - alpha/2)
        else:
            critical_z = stats.norm.ppf(1 - alpha)
        
        # Calculate power
        power = 1 - stats.norm.cdf(critical_z - nc)
        if alternative == 'two-sided':
            power += stats.norm.cdf(-critical_z - nc)
        
        return float(power)
    
    def _interpret_results(self, p_value: float, rho: float, r_squared: float) -> str:
        """Interpret test results."""
        sig = "significant" if p_value < 0.05 else "not significant"
        
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            strength = "negligible"
        elif abs_rho < 0.3:
            strength = "weak"
        elif abs_rho < 0.5:
            strength = "moderate"
        elif abs_rho < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if rho > 0 else "negative"
        
        return f"{strength.capitalize()} {direction} correlation (ρ={rho:.3f}, p={p_value:.4f}), explaining {r_squared:.1%} of variance"


# Factory for non-parametric tests
class NonParametricTestFactory:
    """Factory for creating non-parametric test instances."""
    
    def __init__(self):
        """Initialize factory with available tests."""
        self.tests = {
            'mann_whitney': MannWhitneyUTest(),
            'wilcoxon_ranksum': MannWhitneyUTest(),  # Alias
            'kruskal_wallis': KruskalWallisTest(),
            'wilcoxon_signed': WilcoxonSignedRankTest(),
            'wilcoxon_paired': WilcoxonSignedRankTest(),  # Alias
            'spearman': SpearmanCorrelationTest(),
            'spearman_correlation': SpearmanCorrelationTest()  # Alias
        }
    
    def get_test(self, test_name: str) -> Optional[NonParametricTest]:
        """Get test instance by name."""
        return self.tests.get(test_name.lower())
    
    def list_tests(self) -> List[str]:
        """List available test names."""
        return list(self.tests.keys())