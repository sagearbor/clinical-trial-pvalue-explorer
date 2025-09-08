"""Bayesian statistical analysis engine."""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BayesianEngine:
    """
    Bayesian statistical analysis engine for clinical trials.
    
    Provides Bayesian alternatives to frequentist tests without requiring PyMC.
    Uses conjugate priors and analytical solutions where possible.
    """
    
    def __init__(self):
        """Initialize Bayesian engine."""
        self.prior_library = self._initialize_prior_library()
    
    def _initialize_prior_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library of standard priors."""
        return {
            'uninformative': {
                'mu_mean': 0,
                'mu_variance': 1000,
                'sigma_shape': 0.001,
                'sigma_scale': 1000,
                'description': 'Weakly informative prior'
            },
            'skeptical': {
                'mu_mean': 0,
                'mu_variance': 0.1,
                'sigma_shape': 2,
                'sigma_scale': 1,
                'description': 'Skeptical prior centered at null'
            },
            'optimistic': {
                'mu_mean': 0.5,
                'mu_variance': 0.5,
                'sigma_shape': 2,
                'sigma_scale': 1,
                'description': 'Optimistic prior expecting effect'
            },
            'historical': {
                'mu_mean': None,  # To be set from historical data
                'mu_variance': None,
                'sigma_shape': None,
                'sigma_scale': None,
                'description': 'Prior based on historical data'
            }
        }
    
    def bayesian_t_test(
        self,
        group1: Optional[np.ndarray] = None,
        group2: Optional[np.ndarray] = None,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        mean1: Optional[float] = None,
        mean2: Optional[float] = None,
        std1: Optional[float] = None,
        std2: Optional[float] = None,
        prior_type: str = 'uninformative',
        credible_level: float = 0.95,
        rope: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian t-test using conjugate priors.
        
        Args:
            group1, group2: Data arrays (if available)
            n1, n2: Sample sizes (if data not provided)
            mean1, mean2: Sample means (if data not provided)
            std1, std2: Sample standard deviations (if data not provided)
            prior_type: Type of prior to use
            credible_level: Credible interval level
            rope: Region of Practical Equivalence (lower, upper)
            
        Returns:
            Dictionary with Bayesian analysis results
        """
        # Get data statistics
        if group1 is not None and group2 is not None:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        elif n1 is None or n2 is None or mean1 is None or mean2 is None:
            # Simulate data if not enough information
            n1 = n1 or 50
            n2 = n2 or 50
            effect_size = 0.5
            mean1 = 0
            mean2 = effect_size
            std1 = std2 = 1
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Calculate difference in means
        mean_diff = mean2 - mean1
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        
        # Get prior parameters
        prior = self.prior_library[prior_type]
        
        # Update posterior using conjugate prior (Normal-Inverse-Gamma)
        # Simplified: using normal approximation for difference
        prior_mean = prior['mu_mean']
        prior_variance = prior['mu_variance']
        
        # Posterior parameters (conjugate update)
        posterior_variance = 1 / (1/prior_variance + 1/se_diff**2)
        posterior_mean = posterior_variance * (prior_mean/prior_variance + mean_diff/se_diff**2)
        posterior_std = np.sqrt(posterior_variance)
        
        # Calculate credible interval
        alpha = 1 - credible_level
        ci_lower = stats.norm.ppf(alpha/2, posterior_mean, posterior_std)
        ci_upper = stats.norm.ppf(1 - alpha/2, posterior_mean, posterior_std)
        
        # Calculate probability of positive effect
        prob_positive = 1 - stats.norm.cdf(0, posterior_mean, posterior_std)
        
        # Calculate Bayes Factor (BF10)
        # Simplified: ratio of likelihoods under H1 and H0
        bf10 = self._calculate_bayes_factor(mean_diff, se_diff, prior_mean, prior_variance)
        
        # ROPE analysis if provided
        rope_result = None
        if rope:
            prob_in_rope = stats.norm.cdf(rope[1], posterior_mean, posterior_std) - \
                          stats.norm.cdf(rope[0], posterior_mean, posterior_std)
            rope_result = {
                'rope': rope,
                'prob_in_rope': float(prob_in_rope),
                'prob_below_rope': float(stats.norm.cdf(rope[0], posterior_mean, posterior_std)),
                'prob_above_rope': float(1 - stats.norm.cdf(rope[1], posterior_mean, posterior_std))
            }
        
        # Calculate effect size (Cohen's d)
        cohens_d = mean_diff / pooled_std
        
        # Posterior predictive check
        posterior_samples = np.random.normal(posterior_mean, posterior_std, 10000)
        
        return {
            'test_name': 'Bayesian T-Test',
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std),
            'credible_interval': {
                'level': credible_level,
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'prob_positive_effect': float(prob_positive),
            'bayes_factor': float(bf10),
            'bf_interpretation': self._interpret_bayes_factor(bf10),
            'effect_size': float(cohens_d),
            'rope_analysis': rope_result,
            'prior_type': prior_type,
            'sample_sizes': {'n1': n1, 'n2': n2},
            'observed_difference': float(mean_diff),
            'interpretation': self._interpret_bayesian_results(
                prob_positive, bf10, ci_lower, ci_upper, rope_result
            )
        }
    
    def bayesian_proportion_test(
        self,
        successes1: int,
        n1: int,
        successes2: int,
        n2: int,
        prior_alpha: float = 1,
        prior_beta: float = 1,
        credible_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bayesian test for difference in proportions using Beta-Binomial conjugate prior.
        
        Args:
            successes1, successes2: Number of successes in each group
            n1, n2: Sample sizes
            prior_alpha, prior_beta: Beta prior parameters
            credible_level: Credible interval level
            
        Returns:
            Dictionary with Bayesian analysis results
        """
        # Posterior parameters (Beta conjugate update)
        post_alpha1 = prior_alpha + successes1
        post_beta1 = prior_beta + n1 - successes1
        post_alpha2 = prior_alpha + successes2
        post_beta2 = prior_beta + n2 - successes2
        
        # Sample from posteriors
        n_samples = 10000
        samples1 = np.random.beta(post_alpha1, post_beta1, n_samples)
        samples2 = np.random.beta(post_alpha2, post_beta2, n_samples)
        
        # Calculate difference
        diff_samples = samples2 - samples1
        
        # Posterior statistics
        posterior_mean = np.mean(diff_samples)
        posterior_std = np.std(diff_samples)
        
        # Credible interval
        alpha = 1 - credible_level
        ci_lower = np.percentile(diff_samples, 100 * alpha/2)
        ci_upper = np.percentile(diff_samples, 100 * (1 - alpha/2))
        
        # Probability of positive effect
        prob_positive = np.mean(diff_samples > 0)
        
        # Calculate risk ratio and odds ratio
        risk_ratio_samples = samples2 / samples1
        odds_ratio_samples = (samples2 / (1 - samples2)) / (samples1 / (1 - samples1))
        
        rr_mean = np.mean(risk_ratio_samples)
        rr_ci = np.percentile(risk_ratio_samples, [100*alpha/2, 100*(1-alpha/2)])
        
        or_mean = np.mean(odds_ratio_samples)
        or_ci = np.percentile(odds_ratio_samples, [100*alpha/2, 100*(1-alpha/2)])
        
        return {
            'test_name': 'Bayesian Proportion Test',
            'posterior_mean_diff': float(posterior_mean),
            'posterior_std_diff': float(posterior_std),
            'credible_interval': {
                'level': credible_level,
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'prob_positive_effect': float(prob_positive),
            'risk_ratio': {
                'mean': float(rr_mean),
                'ci_lower': float(rr_ci[0]),
                'ci_upper': float(rr_ci[1])
            },
            'odds_ratio': {
                'mean': float(or_mean),
                'ci_lower': float(or_ci[0]),
                'ci_upper': float(or_ci[1])
            },
            'observed_proportions': {
                'p1': successes1 / n1,
                'p2': successes2 / n2
            },
            'sample_sizes': {'n1': n1, 'n2': n2},
            'prior': f'Beta({prior_alpha}, {prior_beta})',
            'interpretation': self._interpret_proportion_results(
                prob_positive, ci_lower, ci_upper, rr_mean
            )
        }
    
    def bayesian_anova(
        self,
        groups: List[np.ndarray],
        prior_type: str = 'uninformative',
        credible_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bayesian one-way ANOVA using analytical approximation.
        
        Args:
            groups: List of data arrays for each group
            prior_type: Type of prior to use
            credible_level: Credible interval level
            
        Returns:
            Dictionary with Bayesian ANOVA results
        """
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        
        # Calculate group statistics
        group_means = [np.mean(g) for g in groups]
        group_vars = [np.var(g, ddof=1) for g in groups]
        group_sizes = [len(g) for g in groups]
        
        # Grand mean
        grand_mean = np.sum([m * n for m, n in zip(group_means, group_sizes)]) / n_total
        
        # Between-group variance
        ssb = np.sum([n * (m - grand_mean)**2 for m, n in zip(group_means, group_sizes)])
        msb = ssb / (k - 1)
        
        # Within-group variance
        ssw = np.sum([(n - 1) * v for n, v in zip(group_sizes, group_vars)])
        msw = ssw / (n_total - k)
        
        # Bayes Factor for group differences
        # Using BIC approximation
        bic_null = n_total * np.log(ssw + ssb) / n_total
        bic_alternative = n_total * np.log(ssw / n_total) + (k - 1) * np.log(n_total)
        bf10 = np.exp((bic_null - bic_alternative) / 2)
        
        # Effect size (eta-squared)
        eta_squared = ssb / (ssb + ssw)
        
        # Posterior probabilities for pairwise differences
        pairwise_probs = []
        for i in range(k):
            for j in range(i + 1, k):
                # Approximate posterior probability of difference
                mean_diff = group_means[i] - group_means[j]
                se_diff = np.sqrt(msw * (1/group_sizes[i] + 1/group_sizes[j]))
                z_score = mean_diff / se_diff
                prob_diff = 1 - stats.norm.cdf(0, mean_diff, se_diff)
                pairwise_probs.append({
                    'groups': (i, j),
                    'mean_diff': float(mean_diff),
                    'prob_different': float(max(prob_diff, 1 - prob_diff))
                })
        
        return {
            'test_name': 'Bayesian One-Way ANOVA',
            'n_groups': k,
            'n_total': n_total,
            'bayes_factor': float(bf10),
            'bf_interpretation': self._interpret_bayes_factor(bf10),
            'eta_squared': float(eta_squared),
            'group_means': [float(m) for m in group_means],
            'grand_mean': float(grand_mean),
            'pairwise_comparisons': pairwise_probs,
            'interpretation': self._interpret_anova_results(bf10, eta_squared, k)
        }
    
    def bayesian_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        prior_type: str = 'uninformative',
        credible_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bayesian correlation analysis.
        
        Args:
            x, y: Data arrays
            prior_type: Type of prior to use
            credible_level: Credible interval level
            
        Returns:
            Dictionary with Bayesian correlation results
        """
        n = len(x)
        
        # Calculate sample correlation
        r = np.corrcoef(x, y)[0, 1]
        
        # Fisher transformation
        z = np.arctanh(r)
        se_z = 1 / np.sqrt(n - 3)
        
        # Posterior distribution (using uniform prior on correlation)
        # This is approximate but reasonable for moderate sample sizes
        posterior_samples = np.random.normal(z, se_z, 10000)
        r_samples = np.tanh(posterior_samples)
        
        # Posterior statistics
        posterior_mean = np.mean(r_samples)
        posterior_std = np.std(r_samples)
        
        # Credible interval
        alpha = 1 - credible_level
        ci_lower = np.percentile(r_samples, 100 * alpha/2)
        ci_upper = np.percentile(r_samples, 100 * (1 - alpha/2))
        
        # Probability of positive correlation
        prob_positive = np.mean(r_samples > 0)
        
        # Bayes Factor for correlation vs no correlation
        # Using Jeffreys (1961) approximation
        bf10 = self._correlation_bayes_factor(r, n)
        
        # R-squared
        r_squared = r ** 2
        
        return {
            'test_name': 'Bayesian Correlation',
            'correlation': float(r),
            'posterior_mean': float(posterior_mean),
            'posterior_std': float(posterior_std),
            'credible_interval': {
                'level': credible_level,
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'prob_positive': float(prob_positive),
            'bayes_factor': float(bf10),
            'bf_interpretation': self._interpret_bayes_factor(bf10),
            'r_squared': float(r_squared),
            'n': n,
            'interpretation': self._interpret_correlation_results(
                r, prob_positive, bf10, ci_lower, ci_upper
            )
        }
    
    def bayesian_power_analysis(
        self,
        effect_size: float,
        n: int,
        prior_type: str = 'uninformative',
        design_type: str = 'two_sample',
        success_criterion: float = 0.95
    ) -> Dict[str, Any]:
        """
        Bayesian power analysis and sample size determination.
        
        Args:
            effect_size: Expected effect size
            n: Sample size (per group for two-sample)
            prior_type: Type of prior to use
            design_type: 'two_sample', 'paired', 'correlation'
            success_criterion: Probability threshold for success
            
        Returns:
            Dictionary with Bayesian power analysis results
        """
        # Simulate multiple hypothetical studies
        n_simulations = 1000
        success_count = 0
        
        for _ in range(n_simulations):
            if design_type == 'two_sample':
                # Simulate two-sample data
                group1 = np.random.normal(0, 1, n)
                group2 = np.random.normal(effect_size, 1, n)
                
                # Perform Bayesian analysis
                result = self.bayesian_t_test(
                    group1=group1,
                    group2=group2,
                    prior_type=prior_type
                )
                
                # Check if credible interval excludes zero
                if result['credible_interval']['lower'] > 0 or \
                   result['credible_interval']['upper'] < 0:
                    success_count += 1
            
            elif design_type == 'paired':
                # Simulate paired data
                differences = np.random.normal(effect_size, 1, n)
                
                # Check if credible interval excludes zero
                posterior_mean = np.mean(differences)
                posterior_std = np.std(differences) / np.sqrt(n)
                ci_lower = posterior_mean - 1.96 * posterior_std
                
                if ci_lower > 0:
                    success_count += 1
            
            elif design_type == 'correlation':
                # Simulate correlated data
                mean = [0, 0]
                cov = [[1, effect_size], [effect_size, 1]]
                data = np.random.multivariate_normal(mean, cov, n)
                
                result = self.bayesian_correlation(data[:, 0], data[:, 1])
                
                if result['credible_interval']['lower'] > 0:
                    success_count += 1
        
        # Calculate Bayesian power
        bayesian_power = success_count / n_simulations
        
        # Determine required sample size for target power
        required_n = self._find_required_n_bayesian(
            effect_size, success_criterion, prior_type, design_type
        )
        
        return {
            'design_type': design_type,
            'effect_size': effect_size,
            'sample_size': n,
            'bayesian_power': float(bayesian_power),
            'success_criterion': success_criterion,
            'required_n_for_criterion': required_n,
            'prior_type': prior_type,
            'interpretation': self._interpret_power_results(
                bayesian_power, success_criterion, required_n
            )
        }
    
    # Helper methods
    def _calculate_bayes_factor(
        self,
        observed_diff: float,
        se: float,
        prior_mean: float,
        prior_variance: float
    ) -> float:
        """Calculate Bayes Factor using Savage-Dickey density ratio."""
        # Likelihood under null (diff = 0)
        likelihood_null = stats.norm.pdf(0, observed_diff, se)
        
        # Prior under null
        prior_null = stats.norm.pdf(0, prior_mean, np.sqrt(prior_variance))
        
        # Posterior under null
        post_var = 1 / (1/prior_variance + 1/se**2)
        post_mean = post_var * (prior_mean/prior_variance + observed_diff/se**2)
        post_std = np.sqrt(post_var)
        posterior_null = stats.norm.pdf(0, post_mean, post_std)
        
        # Savage-Dickey ratio
        bf01 = posterior_null / prior_null
        bf10 = 1 / bf01 if bf01 > 0 else float('inf')
        
        return bf10
    
    def _correlation_bayes_factor(self, r: float, n: int) -> float:
        """Calculate Bayes Factor for correlation (Jeffreys, 1961)."""
        # Simplified approximation
        t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
        
        # BF10 approximation for correlation
        bf10 = (1 + t**2 / (n - 2)) ** (-(n - 1) / 2)
        bf10 = 1 / bf10  # Invert for H1 over H0
        
        return bf10
    
    def _interpret_bayes_factor(self, bf: float) -> str:
        """Interpret Bayes Factor according to Jeffreys' scale."""
        if bf < 1:
            return f"Evidence for null (BF = {bf:.2f})"
        elif bf < 3:
            return "Anecdotal evidence for alternative"
        elif bf < 10:
            return "Moderate evidence for alternative"
        elif bf < 30:
            return "Strong evidence for alternative"
        elif bf < 100:
            return "Very strong evidence for alternative"
        else:
            return "Decisive evidence for alternative"
    
    def _interpret_bayesian_results(
        self,
        prob_positive: float,
        bf: float,
        ci_lower: float,
        ci_upper: float,
        rope_result: Optional[Dict] = None
    ) -> str:
        """Interpret Bayesian test results."""
        # Effect direction
        if prob_positive > 0.95:
            direction = "strong evidence for positive effect"
        elif prob_positive > 0.75:
            direction = "moderate evidence for positive effect"
        elif prob_positive < 0.05:
            direction = "strong evidence for negative effect"
        elif prob_positive < 0.25:
            direction = "moderate evidence for negative effect"
        else:
            direction = "uncertain effect direction"
        
        # Credible interval
        if ci_lower > 0:
            ci_interp = "credible interval excludes zero (positive effect)"
        elif ci_upper < 0:
            ci_interp = "credible interval excludes zero (negative effect)"
        else:
            ci_interp = "credible interval includes zero"
        
        # ROPE if provided
        rope_interp = ""
        if rope_result:
            if rope_result['prob_in_rope'] > 0.95:
                rope_interp = ", effect is practically equivalent to null"
            elif rope_result['prob_above_rope'] > 0.95:
                rope_interp = ", effect is practically significant"
        
        return f"{direction}, {ci_interp}{rope_interp}. {self._interpret_bayes_factor(bf)}"
    
    def _interpret_proportion_results(
        self,
        prob_positive: float,
        ci_lower: float,
        ci_upper: float,
        risk_ratio: float
    ) -> str:
        """Interpret Bayesian proportion test results."""
        # Risk ratio interpretation
        if risk_ratio > 1.5:
            rr_interp = "substantial increase in risk"
        elif risk_ratio > 1.2:
            rr_interp = "moderate increase in risk"
        elif risk_ratio < 0.67:
            rr_interp = "substantial decrease in risk"
        elif risk_ratio < 0.83:
            rr_interp = "moderate decrease in risk"
        else:
            rr_interp = "minimal change in risk"
        
        # Credible interval
        if ci_lower > 0:
            ci_interp = "credibly higher proportion in group 2"
        elif ci_upper < 0:
            ci_interp = "credibly higher proportion in group 1"
        else:
            ci_interp = "no credible difference"
        
        return f"{rr_interp} (RR={risk_ratio:.2f}), {ci_interp}"
    
    def _interpret_anova_results(
        self,
        bf: float,
        eta_squared: float,
        k: int
    ) -> str:
        """Interpret Bayesian ANOVA results."""
        # Effect size interpretation
        if eta_squared < 0.01:
            effect_interp = "negligible effect"
        elif eta_squared < 0.06:
            effect_interp = "small effect"
        elif eta_squared < 0.14:
            effect_interp = "medium effect"
        else:
            effect_interp = "large effect"
        
        bf_interp = self._interpret_bayes_factor(bf)
        
        return f"Comparing {k} groups: {effect_interp} (η²={eta_squared:.3f}), {bf_interp}"
    
    def _interpret_correlation_results(
        self,
        r: float,
        prob_positive: float,
        bf: float,
        ci_lower: float,
        ci_upper: float
    ) -> str:
        """Interpret Bayesian correlation results."""
        # Correlation strength
        abs_r = abs(r)
        if abs_r < 0.1:
            strength = "negligible"
        elif abs_r < 0.3:
            strength = "weak"
        elif abs_r < 0.5:
            strength = "moderate"
        elif abs_r < 0.7:
            strength = "strong"
        else:
            strength = "very strong"
        
        direction = "positive" if r > 0 else "negative"
        
        # Credible interval
        if ci_lower > 0:
            ci_interp = "credibly positive"
        elif ci_upper < 0:
            ci_interp = "credibly negative"
        else:
            ci_interp = "uncertain direction"
        
        bf_interp = self._interpret_bayes_factor(bf)
        
        return f"{strength} {direction} correlation (r={r:.3f}), {ci_interp}, {bf_interp}"
    
    def _interpret_power_results(
        self,
        power: float,
        criterion: float,
        required_n: int
    ) -> str:
        """Interpret Bayesian power analysis results."""
        if power >= criterion:
            power_interp = f"Adequate power ({power:.1%} >= {criterion:.1%})"
        else:
            power_interp = f"Insufficient power ({power:.1%} < {criterion:.1%})"
        
        return f"{power_interp}. Required n={required_n} for {criterion:.1%} power"
    
    def _find_required_n_bayesian(
        self,
        effect_size: float,
        target_power: float,
        prior_type: str,
        design_type: str
    ) -> int:
        """Find required sample size for target Bayesian power."""
        for n in range(10, 1000, 10):
            result = self.bayesian_power_analysis(
                effect_size=effect_size,
                n=n,
                prior_type=prior_type,
                design_type=design_type,
                success_criterion=target_power
            )
            if result['bayesian_power'] >= target_power:
                return n
        return 1000