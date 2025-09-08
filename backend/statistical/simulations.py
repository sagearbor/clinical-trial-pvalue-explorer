"""Monte Carlo simulation framework for clinical trials."""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SimulationParameters:
    """Parameters for trial simulation."""
    n_simulations: int = 10000
    n_per_group: int = 50
    effect_size: float = 0.5
    alpha: float = 0.05
    test_type: str = 'two_sample_t_test'
    distribution: str = 'normal'
    variance_ratio: float = 1.0
    dropout_rate: float = 0.0
    seed: Optional[int] = None


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation."""
    p_values: List[float]
    test_statistics: List[float]
    effect_sizes: List[float]
    power: float
    type_i_error: float
    mean_p_value: float
    median_p_value: float
    convergence_data: Dict[str, List[float]]
    metadata: Dict[str, Any]


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for clinical trials.
    
    Supports parallel processing and various trial designs.
    """
    
    def __init__(self, n_cores: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            n_cores: Number of CPU cores to use (None for auto-detect)
        """
        self.n_cores = n_cores or mp.cpu_count()
        self.rng = np.random.RandomState()
    
    def simulate_trial(
        self,
        params: SimulationParameters,
        show_progress: bool = False
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation for a clinical trial.
        
        Args:
            params: Simulation parameters
            show_progress: Whether to show progress updates
            
        Returns:
            SimulationResults object
        """
        if params.seed is not None:
            self.rng.seed(params.seed)
        
        # Determine simulation function based on test type
        sim_func = self._get_simulation_function(params.test_type)
        
        # Run simulations in parallel
        results = self._run_parallel_simulations(
            sim_func,
            params,
            show_progress
        )
        
        # Analyze results
        return self._analyze_results(results, params)
    
    def _get_simulation_function(
        self,
        test_type: str
    ) -> Callable:
        """Get appropriate simulation function for test type."""
        sim_functions = {
            'two_sample_t_test': self._simulate_two_sample_t,
            'paired_t_test': self._simulate_paired_t,
            'chi_square': self._simulate_chi_square,
            'one_way_anova': self._simulate_anova,
            'correlation': self._simulate_correlation,
            'mann_whitney': self._simulate_mann_whitney,
            'proportion': self._simulate_proportion
        }
        
        return sim_functions.get(test_type, self._simulate_two_sample_t)
    
    def _run_parallel_simulations(
        self,
        sim_func: Callable,
        params: SimulationParameters,
        show_progress: bool
    ) -> List[Dict[str, float]]:
        """Run simulations in parallel."""
        results = []
        
        # Split simulations across cores
        n_per_core = params.n_simulations // self.n_cores
        remainder = params.n_simulations % self.n_cores
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            futures = []
            
            for i in range(self.n_cores):
                n_sims = n_per_core + (1 if i < remainder else 0)
                if n_sims > 0:
                    future = executor.submit(
                        self._run_simulation_batch,
                        sim_func,
                        params,
                        n_sims,
                        seed=params.seed + i if params.seed else None
                    )
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                batch_results = future.result()
                results.extend(batch_results)
                
                if show_progress:
                    progress = len(results) / params.n_simulations
                    print(f"Progress: {progress:.1%}", end='\r')
        
        return results
    
    def _run_simulation_batch(
        self,
        sim_func: Callable,
        params: SimulationParameters,
        n_sims: int,
        seed: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """Run a batch of simulations."""
        if seed is not None:
            np.random.seed(seed)
        
        results = []
        for _ in range(n_sims):
            result = sim_func(params)
            results.append(result)
        
        return results
    
    def _simulate_two_sample_t(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate two-sample t-test."""
        # Generate data
        if params.distribution == 'normal':
            group1 = np.random.normal(0, 1, params.n_per_group)
            group2 = np.random.normal(
                params.effect_size,
                np.sqrt(params.variance_ratio),
                params.n_per_group
            )
        elif params.distribution == 'exponential':
            group1 = np.random.exponential(1, params.n_per_group)
            group2 = np.random.exponential(
                1 / (1 + params.effect_size),
                params.n_per_group
            )
        else:  # uniform
            group1 = np.random.uniform(-1, 1, params.n_per_group)
            group2 = np.random.uniform(
                -1 + params.effect_size,
                1 + params.effect_size,
                params.n_per_group
            )
        
        # Apply dropout
        if params.dropout_rate > 0:
            n1 = int(params.n_per_group * (1 - params.dropout_rate))
            n2 = int(params.n_per_group * (1 - params.dropout_rate))
            group1 = group1[:n1]
            group2 = group2[:n2]
        
        # Perform test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # Calculate observed effect size
        pooled_std = np.sqrt(
            ((len(group1) - 1) * np.var(group1, ddof=1) +
             (len(group2) - 1) * np.var(group2, ddof=1)) /
            (len(group1) + len(group2) - 2)
        )
        
        if pooled_std > 0:
            observed_d = (np.mean(group2) - np.mean(group1)) / pooled_std
        else:
            observed_d = 0
        
        return {
            'p_value': p_value,
            'test_statistic': t_stat,
            'effect_size': observed_d,
            'n1': len(group1),
            'n2': len(group2)
        }
    
    def _simulate_paired_t(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate paired t-test."""
        # Generate paired data
        baseline = np.random.normal(0, 1, params.n_per_group)
        treatment_effect = np.random.normal(
            params.effect_size,
            0.5,  # Within-subject variance
            params.n_per_group
        )
        follow_up = baseline + treatment_effect
        
        # Perform test
        t_stat, p_value = stats.ttest_rel(baseline, follow_up)
        
        # Calculate effect size
        differences = follow_up - baseline
        observed_d = np.mean(differences) / np.std(differences, ddof=1)
        
        return {
            'p_value': p_value,
            'test_statistic': t_stat,
            'effect_size': observed_d,
            'n_pairs': params.n_per_group
        }
    
    def _simulate_chi_square(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate chi-square test."""
        # Create contingency table based on effect size
        # Effect size here is odds ratio
        n_total = params.n_per_group * 2
        
        # Base probabilities
        p1 = 0.5
        odds1 = p1 / (1 - p1)
        odds2 = odds1 * np.exp(params.effect_size)  # Log odds ratio
        p2 = odds2 / (1 + odds2)
        
        # Generate data
        group1_success = np.random.binomial(params.n_per_group, p1)
        group2_success = np.random.binomial(params.n_per_group, p2)
        
        # Create contingency table
        table = np.array([
            [group1_success, params.n_per_group - group1_success],
            [group2_success, params.n_per_group - group2_success]
        ])
        
        # Perform test
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        
        # Calculate Cramér's V
        n = np.sum(table)
        min_dim = min(table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        return {
            'p_value': p_value,
            'test_statistic': chi2,
            'effect_size': cramers_v,
            'table': table.tolist()
        }
    
    def _simulate_anova(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate one-way ANOVA."""
        n_groups = 3  # Default to 3 groups
        groups = []
        
        # Generate data for each group
        for i in range(n_groups):
            mean = i * params.effect_size
            group = np.random.normal(mean, 1, params.n_per_group)
            groups.append(group)
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate eta-squared
        grand_mean = np.mean(np.concatenate(groups))
        ssb = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        sst = sum(np.sum((g - grand_mean)**2) for g in groups)
        
        if sst > 0:
            eta_squared = ssb / sst
        else:
            eta_squared = 0
        
        return {
            'p_value': p_value,
            'test_statistic': f_stat,
            'effect_size': eta_squared,
            'n_groups': n_groups
        }
    
    def _simulate_correlation(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate correlation test."""
        # Generate correlated data
        mean = [0, 0]
        cov = [[1, params.effect_size], [params.effect_size, 1]]
        
        try:
            data = np.random.multivariate_normal(
                mean,
                cov,
                params.n_per_group
            )
            x, y = data[:, 0], data[:, 1]
        except:
            # If covariance matrix is not positive definite
            x = np.random.normal(0, 1, params.n_per_group)
            y = params.effect_size * x + \
                np.sqrt(1 - params.effect_size**2) * \
                np.random.normal(0, 1, params.n_per_group)
        
        # Calculate correlation
        r, p_value = stats.pearsonr(x, y)
        
        # Fisher z-transformation for test statistic
        z = np.arctanh(r)
        
        return {
            'p_value': p_value,
            'test_statistic': z,
            'effect_size': r,
            'n': params.n_per_group
        }
    
    def _simulate_mann_whitney(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate Mann-Whitney U test."""
        # Generate data (can be non-normal)
        if params.distribution == 'normal':
            group1 = np.random.normal(0, 1, params.n_per_group)
            group2 = np.random.normal(params.effect_size, 1, params.n_per_group)
        else:
            # Use exponential for non-normal
            group1 = np.random.exponential(1, params.n_per_group)
            group2 = np.random.exponential(
                np.exp(-params.effect_size),
                params.n_per_group
            )
        
        # Perform test
        u_stat, p_value = stats.mannwhitneyu(
            group1,
            group2,
            alternative='two-sided'
        )
        
        # Calculate rank-biserial correlation
        n1, n2 = len(group1), len(group2)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'p_value': p_value,
            'test_statistic': u_stat,
            'effect_size': rank_biserial,
            'n1': n1,
            'n2': n2
        }
    
    def _simulate_proportion(
        self,
        params: SimulationParameters
    ) -> Dict[str, float]:
        """Simulate proportion test."""
        # Base proportion
        p1 = 0.3
        
        # Calculate p2 based on effect size (risk difference)
        p2 = p1 + params.effect_size
        p2 = max(0, min(1, p2))  # Ensure valid probability
        
        # Generate data
        x1 = np.random.binomial(params.n_per_group, p1)
        x2 = np.random.binomial(params.n_per_group, p2)
        
        # Perform test
        count = np.array([x1, x2])
        nobs = np.array([params.n_per_group, params.n_per_group])
        
        # Two-proportion z-test
        p_pooled = np.sum(count) / np.sum(nobs)
        se = np.sqrt(p_pooled * (1 - p_pooled) * np.sum(1 / nobs))
        
        if se > 0:
            z_stat = (x2/params.n_per_group - x1/params.n_per_group) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat = 0
            p_value = 1
        
        # Risk difference as effect size
        risk_diff = x2/params.n_per_group - x1/params.n_per_group
        
        return {
            'p_value': p_value,
            'test_statistic': z_stat,
            'effect_size': risk_diff,
            'p1': x1/params.n_per_group,
            'p2': x2/params.n_per_group
        }
    
    def _analyze_results(
        self,
        results: List[Dict[str, float]],
        params: SimulationParameters
    ) -> SimulationResults:
        """Analyze simulation results."""
        p_values = [r['p_value'] for r in results]
        test_stats = [r['test_statistic'] for r in results]
        effect_sizes = [r['effect_size'] for r in results]
        
        # Calculate power and Type I error
        if params.effect_size == 0:
            # Under null hypothesis
            type_i_error = np.mean([p < params.alpha for p in p_values])
            power = np.nan
        else:
            # Under alternative hypothesis
            power = np.mean([p < params.alpha for p in p_values])
            type_i_error = np.nan
        
        # Calculate convergence data
        convergence_data = self._calculate_convergence(p_values, params.alpha)
        
        # Create metadata
        metadata = {
            'test_type': params.test_type,
            'n_simulations': params.n_simulations,
            'n_per_group': params.n_per_group,
            'true_effect_size': params.effect_size,
            'alpha': params.alpha,
            'distribution': params.distribution,
            'dropout_rate': params.dropout_rate
        }
        
        return SimulationResults(
            p_values=p_values,
            test_statistics=test_stats,
            effect_sizes=effect_sizes,
            power=power,
            type_i_error=type_i_error,
            mean_p_value=np.mean(p_values),
            median_p_value=np.median(p_values),
            convergence_data=convergence_data,
            metadata=metadata
        )
    
    def _calculate_convergence(
        self,
        p_values: List[float],
        alpha: float
    ) -> Dict[str, List[float]]:
        """Calculate convergence of power estimate."""
        n_sims = len(p_values)
        check_points = [100, 500, 1000, 2500, 5000, 7500, 10000]
        
        convergence = {
            'n': [],
            'power': [],
            'se': []
        }
        
        for n in check_points:
            if n <= n_sims:
                power_est = np.mean([p < alpha for p in p_values[:n]])
                se = np.sqrt(power_est * (1 - power_est) / n)
                
                convergence['n'].append(n)
                convergence['power'].append(power_est)
                convergence['se'].append(se)
        
        return convergence
    
    def calculate_operating_characteristics(
        self,
        effect_sizes: List[float],
        sample_sizes: List[int],
        params: SimulationParameters
    ) -> Dict[str, Any]:
        """
        Calculate operating characteristics across parameter grid.
        
        Args:
            effect_sizes: List of effect sizes to evaluate
            sample_sizes: List of sample sizes to evaluate
            params: Base simulation parameters
            
        Returns:
            Dictionary with operating characteristics
        """
        results = {
            'effect_sizes': effect_sizes,
            'sample_sizes': sample_sizes,
            'power_matrix': [],
            'type_i_error': None
        }
        
        for n in sample_sizes:
            power_row = []
            for effect in effect_sizes:
                # Update parameters
                sim_params = SimulationParameters(
                    n_simulations=params.n_simulations,
                    n_per_group=n,
                    effect_size=effect,
                    alpha=params.alpha,
                    test_type=params.test_type,
                    seed=params.seed
                )
                
                # Run simulation
                sim_results = self.simulate_trial(sim_params)
                
                if effect == 0 and results['type_i_error'] is None:
                    results['type_i_error'] = sim_results.type_i_error
                
                power_row.append(sim_results.power)
            
            results['power_matrix'].append(power_row)
        
        return results