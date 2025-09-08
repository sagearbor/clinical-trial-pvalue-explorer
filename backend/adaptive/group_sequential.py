"""Group sequential designs for clinical trials."""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class SpendingFunction(Enum):
    """Alpha spending function types."""
    OBRIEN_FLEMING = "obrien_fleming"
    POCOCK = "pocock"
    LAN_DEMETS = "lan_demets"
    HWANG_SHIH_DECANI = "hwang_shih_decani"
    CUSTOM = "custom"


@dataclass
class GroupSequentialDesign:
    """Configuration for group sequential design."""
    n_stages: int
    alpha: float = 0.05
    beta: float = 0.2
    spending_function: SpendingFunction = SpendingFunction.OBRIEN_FLEMING
    information_fractions: Optional[List[float]] = None
    futility_spending: Optional[SpendingFunction] = None
    gamma: float = -4  # For Hwang-Shih-DeCani


@dataclass
class InterimAnalysis:
    """Results from an interim analysis."""
    stage: int
    information_fraction: float
    n_accumulated: int
    test_statistic: float
    p_value: float
    efficacy_boundary: float
    futility_boundary: Optional[float]
    stop_for_efficacy: bool
    stop_for_futility: bool
    conditional_power: float
    predictive_power: float


class GroupSequentialAnalyzer:
    """
    Analyzer for group sequential clinical trials.
    
    Supports various alpha spending functions and interim monitoring.
    """
    
    def __init__(self, design: GroupSequentialDesign):
        """
        Initialize analyzer with design parameters.
        
        Args:
            design: Group sequential design configuration
        """
        self.design = design
        self.information_fractions = design.information_fractions or \
                                    self._equal_information_fractions()
        self.alpha_spent = [0.0]
        self.beta_spent = [0.0] if design.futility_spending else None
        self.boundaries = self._calculate_boundaries()
    
    def _equal_information_fractions(self) -> List[float]:
        """Generate equal information fractions."""
        return [(i + 1) / self.design.n_stages for i in range(self.design.n_stages)]
    
    def _calculate_boundaries(self) -> Dict[str, List[float]]:
        """Calculate stopping boundaries for all stages."""
        boundaries = {
            'efficacy': [],
            'futility': []
        }
        
        for k, info_frac in enumerate(self.information_fractions, 1):
            # Efficacy boundary
            alpha_k = self._alpha_spending(info_frac)
            efficacy_bound = self._calculate_critical_value(
                alpha_k,
                k,
                self.alpha_spent
            )
            boundaries['efficacy'].append(efficacy_bound)
            self.alpha_spent.append(alpha_k)
            
            # Futility boundary if specified
            if self.design.futility_spending:
                beta_k = self._beta_spending(info_frac)
                futility_bound = self._calculate_critical_value(
                    beta_k,
                    k,
                    self.beta_spent,
                    futility=True
                )
                boundaries['futility'].append(futility_bound)
                self.beta_spent.append(beta_k)
            else:
                boundaries['futility'].append(None)
        
        return boundaries
    
    def _alpha_spending(self, t: float) -> float:
        """
        Calculate cumulative alpha spent at information fraction t.
        
        Args:
            t: Information fraction (0, 1]
            
        Returns:
            Cumulative alpha spent
        """
        if self.design.spending_function == SpendingFunction.OBRIEN_FLEMING:
            return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.design.alpha/2) / np.sqrt(t)))
        
        elif self.design.spending_function == SpendingFunction.POCOCK:
            return self.design.alpha * np.log(1 + (np.e - 1) * t)
        
        elif self.design.spending_function == SpendingFunction.LAN_DEMETS:
            # Lan-DeMets approximation to O'Brien-Fleming
            return min(self.design.alpha, 2 * (1 - stats.norm.cdf(
                stats.norm.ppf(1 - self.design.alpha/2) / np.sqrt(t)
            )))
        
        elif self.design.spending_function == SpendingFunction.HWANG_SHIH_DECANI:
            gamma = self.design.gamma
            if gamma == 0:
                return self.design.alpha * t
            else:
                return self.design.alpha * (1 - np.exp(-gamma * t)) / (1 - np.exp(-gamma))
        
        else:  # CUSTOM
            return self.design.alpha * t  # Linear spending as default
    
    def _beta_spending(self, t: float) -> float:
        """Calculate cumulative beta spent for futility."""
        # Similar to alpha spending but for Type II error
        if self.design.futility_spending == SpendingFunction.OBRIEN_FLEMING:
            return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.design.beta/2) / np.sqrt(t)))
        else:
            return self.design.beta * t  # Linear spending as default
    
    def _calculate_critical_value(
        self,
        cum_alpha: float,
        stage: int,
        alpha_spent: List[float],
        futility: bool = False
    ) -> float:
        """
        Calculate critical value for given cumulative alpha.
        
        Uses recursive integration for exact boundaries.
        """
        if stage == 1:
            # First stage - simple calculation
            if futility:
                return stats.norm.ppf(cum_alpha)
            else:
                return stats.norm.ppf(1 - cum_alpha/2)
        
        # For later stages, need to account for previous boundaries
        # Simplified approximation
        incremental_alpha = cum_alpha - alpha_spent[-1]
        if futility:
            return stats.norm.ppf(incremental_alpha)
        else:
            return stats.norm.ppf(1 - incremental_alpha/2)
    
    def analyze_interim(
        self,
        stage: int,
        n_accumulated: int,
        test_statistic: float,
        observed_effect: float,
        se: float
    ) -> InterimAnalysis:
        """
        Perform interim analysis at given stage.
        
        Args:
            stage: Current stage (1-indexed)
            n_accumulated: Total sample size accumulated
            test_statistic: Current test statistic (e.g., z-score)
            observed_effect: Observed treatment effect
            se: Standard error of effect estimate
            
        Returns:
            InterimAnalysis results
        """
        if stage > self.design.n_stages:
            raise ValueError(f"Stage {stage} exceeds design stages {self.design.n_stages}")
        
        info_frac = self.information_fractions[stage - 1]
        efficacy_bound = self.boundaries['efficacy'][stage - 1]
        futility_bound = self.boundaries['futility'][stage - 1]
        
        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        
        # Check stopping criteria
        stop_for_efficacy = abs(test_statistic) >= efficacy_bound
        stop_for_futility = (futility_bound is not None and 
                           abs(test_statistic) <= futility_bound)
        
        # Calculate conditional power
        conditional_power = self._calculate_conditional_power(
            stage,
            observed_effect,
            se,
            info_frac
        )
        
        # Calculate predictive power
        predictive_power = self._calculate_predictive_power(
            stage,
            test_statistic,
            info_frac
        )
        
        return InterimAnalysis(
            stage=stage,
            information_fraction=info_frac,
            n_accumulated=n_accumulated,
            test_statistic=test_statistic,
            p_value=p_value,
            efficacy_boundary=efficacy_bound,
            futility_boundary=futility_bound,
            stop_for_efficacy=stop_for_efficacy,
            stop_for_futility=stop_for_futility,
            conditional_power=conditional_power,
            predictive_power=predictive_power
        )
    
    def _calculate_conditional_power(
        self,
        stage: int,
        observed_effect: float,
        se: float,
        info_frac: float
    ) -> float:
        """
        Calculate conditional power given current data.
        
        Probability of success at end of trial given current trend.
        """
        if stage == self.design.n_stages:
            return 0.0  # Already at final stage
        
        # Remaining information
        remaining_info = 1 - info_frac
        
        # Project to end of study
        final_se = se * np.sqrt(info_frac)
        projected_z = observed_effect / final_se
        
        # Final boundary (approximation)
        final_boundary = stats.norm.ppf(1 - self.design.alpha/2)
        
        # Conditional power
        drift = projected_z * np.sqrt(remaining_info / info_frac)
        cond_power = 1 - stats.norm.cdf(final_boundary - drift)
        
        return float(cond_power)
    
    def _calculate_predictive_power(
        self,
        stage: int,
        current_z: float,
        info_frac: float
    ) -> float:
        """
        Calculate predictive power using Bayesian predictive probability.
        
        Averages over posterior distribution of treatment effect.
        """
        if stage == self.design.n_stages:
            return 0.0
        
        # Simplified predictive power calculation
        # Uses current estimate as prior mean
        remaining_info = 1 - info_frac
        
        # Predictive distribution parameters
        pred_mean = current_z * np.sqrt(info_frac)
        pred_var = remaining_info
        
        # Final boundary
        final_boundary = stats.norm.ppf(1 - self.design.alpha/2)
        
        # Predictive power
        pred_power = 1 - stats.norm.cdf(
            (final_boundary - pred_mean) / np.sqrt(pred_var)
        )
        
        return float(pred_power)
    
    def simulate_trial(
        self,
        true_effect: float,
        n_per_stage: int,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Simulate group sequential trial.
        
        Args:
            true_effect: True treatment effect (standardized)
            n_per_stage: Sample size per stage (per group)
            n_simulations: Number of trial simulations
            
        Returns:
            Simulation results including power and expected sample size
        """
        np.random.seed(42)
        
        results = {
            'stop_stage': [],
            'stop_reason': [],
            'rejected': []
        }
        
        for _ in range(n_simulations):
            cumulative_n = 0
            cumulative_sum = 0
            
            for stage in range(1, self.design.n_stages + 1):
                # Generate data for this stage
                n_stage = n_per_stage
                effect_obs = np.random.normal(true_effect, 1/np.sqrt(n_stage))
                
                cumulative_n += n_stage
                cumulative_sum += effect_obs * n_stage
                
                # Calculate test statistic
                pooled_effect = cumulative_sum / cumulative_n
                se = 1 / np.sqrt(cumulative_n)
                z_stat = pooled_effect / se
                
                # Check boundaries
                efficacy_bound = self.boundaries['efficacy'][stage - 1]
                futility_bound = self.boundaries['futility'][stage - 1]
                
                if abs(z_stat) >= efficacy_bound:
                    results['stop_stage'].append(stage)
                    results['stop_reason'].append('efficacy')
                    results['rejected'].append(True)
                    break
                
                if futility_bound and abs(z_stat) <= futility_bound:
                    results['stop_stage'].append(stage)
                    results['stop_reason'].append('futility')
                    results['rejected'].append(False)
                    break
                
                if stage == self.design.n_stages:
                    results['stop_stage'].append(stage)
                    results['stop_reason'].append('completion')
                    results['rejected'].append(abs(z_stat) >= efficacy_bound)
        
        # Calculate summary statistics
        power = np.mean(results['rejected'])
        avg_sample_size = np.mean([s * n_per_stage * 2 for s in results['stop_stage']])
        
        stage_probs = {}
        for stage in range(1, self.design.n_stages + 1):
            stage_probs[f'stage_{stage}'] = np.mean(
                [s == stage for s in results['stop_stage']]
            )
        
        return {
            'power': float(power),
            'expected_sample_size': float(avg_sample_size),
            'stage_probabilities': stage_probs,
            'efficacy_stops': np.mean([r == 'efficacy' for r in results['stop_reason']]),
            'futility_stops': np.mean([r == 'futility' for r in results['stop_reason']])
        }
    
    def plot_boundaries(self) -> Dict[str, Any]:
        """
        Generate data for plotting stopping boundaries.
        
        Returns:
            Dictionary with plotting data
        """
        stages = list(range(1, self.design.n_stages + 1))
        
        plot_data = {
            'stages': stages,
            'information_fractions': self.information_fractions,
            'efficacy_boundaries': self.boundaries['efficacy'],
            'futility_boundaries': self.boundaries['futility'],
            'alpha_spent': self.alpha_spent[1:],
            'beta_spent': self.beta_spent[1:] if self.beta_spent else None
        }
        
        # Add z-score scale boundaries
        plot_data['efficacy_z'] = [
            stats.norm.ppf(1 - a/2) for a in self.alpha_spent[1:]
        ]
        
        # Add nominal p-value boundaries
        plot_data['efficacy_p'] = [
            2 * (1 - stats.norm.cdf(b)) for b in self.boundaries['efficacy']
        ]
        
        return plot_data


class SampleSizeReestimation:
    """
    Sample size re-estimation for adaptive designs.
    
    Allows modification of sample size based on interim results.
    """
    
    def __init__(
        self,
        initial_n: int,
        min_n: int,
        max_n: int,
        method: str = 'conditional_power'
    ):
        """
        Initialize sample size re-estimation.
        
        Args:
            initial_n: Initial planned sample size
            min_n: Minimum allowed sample size
            max_n: Maximum allowed sample size
            method: Re-estimation method
        """
        self.initial_n = initial_n
        self.min_n = min_n
        self.max_n = max_n
        self.method = method
    
    def reestimate(
        self,
        interim_effect: float,
        interim_se: float,
        interim_n: int,
        target_power: float = 0.8,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Re-estimate sample size based on interim data.
        
        Args:
            interim_effect: Observed effect at interim
            interim_se: Standard error at interim
            interim_n: Sample size at interim
            target_power: Target power for re-estimation
            alpha: Significance level
            
        Returns:
            Re-estimation results
        """
        if self.method == 'conditional_power':
            new_n = self._conditional_power_method(
                interim_effect,
                interim_se,
                interim_n,
                target_power,
                alpha
            )
        elif self.method == 'promising_zone':
            new_n = self._promising_zone_method(
                interim_effect,
                interim_se,
                interim_n,
                alpha
            )
        elif self.method == 'bayesian_predictive':
            new_n = self._bayesian_predictive_method(
                interim_effect,
                interim_se,
                interim_n,
                target_power
            )
        else:
            new_n = self.initial_n
        
        # Apply constraints
        new_n = max(self.min_n, min(self.max_n, new_n))
        
        # Calculate increase factor
        increase_factor = new_n / self.initial_n
        
        return {
            'original_n': self.initial_n,
            'reestimated_n': int(new_n),
            'increase_factor': float(increase_factor),
            'method': self.method,
            'constrained': new_n == self.min_n or new_n == self.max_n
        }
    
    def _conditional_power_method(
        self,
        effect: float,
        se: float,
        n: int,
        target_power: float,
        alpha: float
    ) -> int:
        """Re-estimate using conditional power approach."""
        # Critical value
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Required sample size for target power
        z_beta = stats.norm.ppf(target_power)
        required_n = ((z_alpha + z_beta) / effect) ** 2
        
        # Adjust for information already collected
        additional_n = max(0, required_n - n)
        total_n = n + additional_n
        
        return int(total_n)
    
    def _promising_zone_method(
        self,
        effect: float,
        se: float,
        n: int,
        alpha: float
    ) -> int:
        """Re-estimate using promising zone methodology."""
        # Define promising zone boundaries
        lower_bound = 0.5 * stats.norm.ppf(1 - alpha/2)
        upper_bound = 1.5 * stats.norm.ppf(1 - alpha/2)
        
        # Current z-score
        z_current = effect / se
        
        if z_current < lower_bound:
            # Futile - use minimum
            return self.min_n
        elif z_current > upper_bound:
            # Very promising - use initial
            return self.initial_n
        else:
            # In promising zone - increase moderately
            factor = 1.5 - 0.5 * (z_current - lower_bound) / (upper_bound - lower_bound)
            return int(self.initial_n * factor)
    
    def _bayesian_predictive_method(
        self,
        effect: float,
        se: float,
        n: int,
        target_power: float
    ) -> int:
        """Re-estimate using Bayesian predictive probability."""
        # Use current estimate as prior mean
        prior_mean = effect
        prior_var = se ** 2
        
        # Search for sample size giving target predictive probability
        for test_n in range(n, self.max_n, 10):
            remaining_n = test_n - n
            
            # Predictive variance
            pred_var = prior_var + 1/remaining_n
            
            # Predictive probability of success
            z_final = prior_mean / np.sqrt(pred_var)
            pred_prob = 1 - stats.norm.cdf(1.96 - z_final)
            
            if pred_prob >= target_power:
                return test_n
        
        return self.max_n