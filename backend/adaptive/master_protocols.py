"""Master protocol implementations for complex trial designs."""

import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class TrialStatus(Enum):
    """Status of trial arms."""
    ENROLLING = "enrolling"
    SUSPENDED = "suspended"
    GRADUATED = "graduated"
    TERMINATED = "terminated"


@dataclass
class BasketDesign:
    """
    Basket trial design configuration.
    
    Tests single treatment across multiple disease subtypes/baskets.
    """
    n_baskets: int
    basket_names: List[str]
    target_response_rate: float = 0.3
    null_response_rate: float = 0.1
    borrowing_strength: str = "moderate"  # none, weak, moderate, strong
    alpha: float = 0.05
    power: float = 0.8
    max_n_per_basket: int = 30


@dataclass
class UmbrellaDesign:
    """
    Umbrella trial design configuration.
    
    Tests multiple treatments in single disease with biomarker subgroups.
    """
    n_arms: int
    n_biomarkers: int
    biomarker_prevalence: List[float]
    control_response_rate: float = 0.2
    target_response_rates: List[float]
    alpha: float = 0.05
    power: float = 0.8
    shared_control: bool = True


@dataclass
class PlatformDesign:
    """
    Platform trial design configuration.
    
    Perpetual trial with arms entering and leaving over time.
    """
    initial_arms: int
    max_concurrent_arms: int
    control_response_rate: float = 0.2
    enrollment_rate: int = 10  # patients per month
    interim_frequency: int = 3  # months
    graduation_threshold: float = 0.975  # posterior probability
    futility_threshold: float = 0.05


class BasketTrial:
    """
    Basket trial implementation with Bayesian hierarchical modeling.
    
    Allows information borrowing across baskets for efficiency.
    """
    
    def __init__(self, design: BasketDesign):
        """
        Initialize basket trial.
        
        Args:
            design: Basket trial design configuration
        """
        self.design = design
        self.basket_data = {
            name: {'n': 0, 'responses': 0, 'status': TrialStatus.ENROLLING}
            for name in design.basket_names
        }
        self.borrowing_param = self._get_borrowing_parameter()
    
    def _get_borrowing_parameter(self) -> float:
        """Get borrowing parameter based on strength setting."""
        borrowing_map = {
            "none": 0.0,
            "weak": 0.1,
            "moderate": 0.5,
            "strong": 1.0
        }
        return borrowing_map.get(self.design.borrowing_strength, 0.5)
    
    def update_basket(
        self,
        basket_name: str,
        n_new: int,
        n_responses: int
    ) -> None:
        """
        Update basket with new patient data.
        
        Args:
            basket_name: Name of basket to update
            n_new: Number of new patients
            n_responses: Number of responses in new patients
        """
        if basket_name not in self.basket_data:
            raise ValueError(f"Unknown basket: {basket_name}")
        
        self.basket_data[basket_name]['n'] += n_new
        self.basket_data[basket_name]['responses'] += n_responses
    
    def analyze_baskets(self) -> Dict[str, Any]:
        """
        Perform Bayesian analysis with information borrowing.
        
        Returns:
            Analysis results for all baskets
        """
        results = {}
        
        # Extract data
        basket_names = []
        ns = []
        responses = []
        
        for name, data in self.basket_data.items():
            if data['n'] > 0:
                basket_names.append(name)
                ns.append(data['n'])
                responses.append(data['responses'])
        
        if not basket_names:
            return {"error": "No data available"}
        
        # Calculate posterior probabilities with borrowing
        posteriors = self._calculate_posteriors_with_borrowing(ns, responses)
        
        # Analyze each basket
        for i, name in enumerate(basket_names):
            n = ns[i]
            r = responses[i]
            
            # Posterior statistics
            post_mean = posteriors['means'][i]
            post_lower = posteriors['ci_lower'][i]
            post_upper = posteriors['ci_upper'][i]
            
            # Probability of exceeding null
            prob_exceeds_null = posteriors['prob_exceeds_null'][i]
            
            # Probability of exceeding target
            prob_exceeds_target = posteriors['prob_exceeds_target'][i]
            
            # Decision
            if prob_exceeds_target > 0.9:
                decision = "Graduate (Success)"
            elif prob_exceeds_null < 0.1:
                decision = "Terminate (Futility)"
            else:
                decision = "Continue Enrollment"
            
            results[name] = {
                'n': n,
                'responses': r,
                'observed_rate': r/n if n > 0 else 0,
                'posterior_mean': float(post_mean),
                'credible_interval': [float(post_lower), float(post_upper)],
                'prob_exceeds_null': float(prob_exceeds_null),
                'prob_exceeds_target': float(prob_exceeds_target),
                'decision': decision,
                'borrowing_weight': posteriors['borrowing_weights'][i]
            }
        
        # Overall summary
        results['summary'] = {
            'n_baskets': len(basket_names),
            'total_n': sum(ns),
            'total_responses': sum(responses),
            'overall_response_rate': sum(responses) / sum(ns) if sum(ns) > 0 else 0,
            'borrowing_strength': self.design.borrowing_strength,
            'n_graduating': sum(1 for r in results.values() 
                              if isinstance(r, dict) and r.get('decision') == "Graduate (Success)"),
            'n_terminating': sum(1 for r in results.values() 
                               if isinstance(r, dict) and r.get('decision') == "Terminate (Futility)")
        }
        
        return results
    
    def _calculate_posteriors_with_borrowing(
        self,
        ns: List[int],
        responses: List[int]
    ) -> Dict[str, List[float]]:
        """
        Calculate posterior distributions with information borrowing.
        
        Uses empirical Bayes approach for computational efficiency.
        """
        n_baskets = len(ns)
        
        # Prior parameters (uniform prior)
        alpha_prior = 1
        beta_prior = 1
        
        # Calculate individual posterior parameters
        alphas = [alpha_prior + r for r in responses]
        betas = [beta_prior + n - r for n, r in zip(ns, responses)]
        
        # Calculate borrowing weights based on similarity
        response_rates = [r/n if n > 0 else 0.5 for r, n in zip(responses, ns)]
        overall_rate = sum(responses) / sum(ns) if sum(ns) > 0 else 0.5
        
        # Weight based on distance from overall rate
        distances = [abs(rate - overall_rate) for rate in response_rates]
        max_dist = max(distances) if distances else 1
        
        if max_dist > 0:
            similarities = [1 - d/max_dist for d in distances]
        else:
            similarities = [1.0] * n_baskets
        
        borrowing_weights = [s * self.borrowing_param for s in similarities]
        
        # Adjust posteriors with borrowing
        adjusted_alphas = []
        adjusted_betas = []
        
        for i in range(n_baskets):
            # Weighted average with other baskets
            weight_self = 1 - borrowing_weights[i]
            weight_others = borrowing_weights[i] / (n_baskets - 1) if n_baskets > 1 else 0
            
            adj_alpha = weight_self * alphas[i]
            adj_beta = weight_self * betas[i]
            
            for j in range(n_baskets):
                if i != j:
                    adj_alpha += weight_others * alphas[j]
                    adj_beta += weight_others * betas[j]
            
            adjusted_alphas.append(adj_alpha)
            adjusted_betas.append(adj_beta)
        
        # Calculate posterior statistics
        means = [a / (a + b) for a, b in zip(adjusted_alphas, adjusted_betas)]
        
        # Credible intervals (using beta distribution)
        ci_lower = [stats.beta.ppf(0.025, a, b) 
                   for a, b in zip(adjusted_alphas, adjusted_betas)]
        ci_upper = [stats.beta.ppf(0.975, a, b) 
                   for a, b in zip(adjusted_alphas, adjusted_betas)]
        
        # Probabilities
        prob_exceeds_null = [1 - stats.beta.cdf(self.design.null_response_rate, a, b)
                            for a, b in zip(adjusted_alphas, adjusted_betas)]
        prob_exceeds_target = [1 - stats.beta.cdf(self.design.target_response_rate, a, b)
                             for a, b in zip(adjusted_alphas, adjusted_betas)]
        
        return {
            'means': means,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'prob_exceeds_null': prob_exceeds_null,
            'prob_exceeds_target': prob_exceeds_target,
            'borrowing_weights': borrowing_weights
        }
    
    def simulate_trial(
        self,
        true_response_rates: List[float],
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Simulate basket trial to evaluate operating characteristics.
        
        Args:
            true_response_rates: True response rate for each basket
            n_simulations: Number of trial simulations
            
        Returns:
            Operating characteristics
        """
        results = {
            'power_by_basket': {},
            'type1_error_by_basket': {},
            'expected_n_by_basket': {},
            'prob_graduate': {},
            'prob_terminate': {}
        }
        
        for basket_idx, basket_name in enumerate(self.design.basket_names):
            graduates = 0
            terminates = 0
            sample_sizes = []
            
            for _ in range(n_simulations):
                # Reset trial
                self.basket_data = {
                    name: {'n': 0, 'responses': 0, 'status': TrialStatus.ENROLLING}
                    for name in self.design.basket_names
                }
                
                # Simulate enrollment
                for stage in range(3):  # 3 interim analyses
                    n_stage = self.design.max_n_per_basket // 3
                    
                    # Generate responses
                    for idx, name in enumerate(self.design.basket_names):
                        if self.basket_data[name]['status'] == TrialStatus.ENROLLING:
                            responses = np.random.binomial(
                                n_stage,
                                true_response_rates[idx]
                            )
                            self.update_basket(name, n_stage, responses)
                    
                    # Analyze
                    analysis = self.analyze_baskets()
                    
                    # Check decisions
                    if basket_name in analysis:
                        decision = analysis[basket_name]['decision']
                        if "Graduate" in decision:
                            graduates += 1
                            self.basket_data[basket_name]['status'] = TrialStatus.GRADUATED
                            break
                        elif "Terminate" in decision:
                            terminates += 1
                            self.basket_data[basket_name]['status'] = TrialStatus.TERMINATED
                            break
                
                sample_sizes.append(self.basket_data[basket_name]['n'])
            
            # Calculate metrics
            true_rate = true_response_rates[basket_idx]
            
            if true_rate > self.design.target_response_rate:
                # Should graduate (power)
                results['power_by_basket'][basket_name] = graduates / n_simulations
            else:
                # Should not graduate (type I error)
                results['type1_error_by_basket'][basket_name] = graduates / n_simulations
            
            results['expected_n_by_basket'][basket_name] = np.mean(sample_sizes)
            results['prob_graduate'][basket_name] = graduates / n_simulations
            results['prob_terminate'][basket_name] = terminates / n_simulations
        
        return results


class UmbrellaTrial:
    """
    Umbrella trial implementation with biomarker-driven allocation.
    
    Tests multiple treatments in a single disease with biomarker stratification.
    """
    
    def __init__(self, design: UmbrellaDesign):
        """
        Initialize umbrella trial.
        
        Args:
            design: Umbrella trial design configuration
        """
        self.design = design
        self.arm_data = {}
        
        # Initialize arms
        for i in range(design.n_arms):
            self.arm_data[f"Arm_{i}"] = {
                'n': 0,
                'responses': 0,
                'biomarker_counts': {j: 0 for j in range(design.n_biomarkers)},
                'status': TrialStatus.ENROLLING
            }
        
        # Control arm if shared
        if design.shared_control:
            self.arm_data["Control"] = {
                'n': 0,
                'responses': 0,
                'biomarker_counts': {j: 0 for j in range(design.n_biomarkers)},
                'status': TrialStatus.ENROLLING
            }
    
    def allocate_patient(self, biomarker_profile: List[int]) -> str:
        """
        Allocate patient to treatment arm based on biomarker profile.
        
        Args:
            biomarker_profile: Binary vector of biomarker presence
            
        Returns:
            Assigned arm name
        """
        # Simple allocation: assign to first matching biomarker arm
        for i, has_biomarker in enumerate(biomarker_profile):
            if has_biomarker and i < self.design.n_arms:
                if self.arm_data[f"Arm_{i}"]['status'] == TrialStatus.ENROLLING:
                    return f"Arm_{i}"
        
        # If no match or arms closed, assign to control
        return "Control" if self.design.shared_control else None
    
    def update_arm(
        self,
        arm_name: str,
        n_new: int,
        n_responses: int,
        biomarker_profile: Optional[List[int]] = None
    ) -> None:
        """Update arm with new patient data."""
        if arm_name not in self.arm_data:
            raise ValueError(f"Unknown arm: {arm_name}")
        
        self.arm_data[arm_name]['n'] += n_new
        self.arm_data[arm_name]['responses'] += n_responses
        
        if biomarker_profile:
            for i, has_biomarker in enumerate(biomarker_profile):
                if has_biomarker:
                    self.arm_data[arm_name]['biomarker_counts'][i] += n_new
    
    def analyze_arms(self) -> Dict[str, Any]:
        """
        Analyze all treatment arms.
        
        Returns:
            Analysis results for all arms
        """
        results = {}
        
        # Get control rate if available
        control_rate = self.design.control_response_rate
        if "Control" in self.arm_data and self.arm_data["Control"]['n'] > 0:
            control_data = self.arm_data["Control"]
            control_rate = control_data['responses'] / control_data['n']
        
        # Analyze each treatment arm
        for arm_name, arm_data in self.arm_data.items():
            if arm_data['n'] == 0:
                continue
            
            n = arm_data['n']
            r = arm_data['responses']
            observed_rate = r / n
            
            # Bayesian posterior (Beta(1,1) prior)
            post_alpha = 1 + r
            post_beta = 1 + n - r
            
            # Posterior statistics
            post_mean = post_alpha / (post_alpha + post_beta)
            ci_lower = stats.beta.ppf(0.025, post_alpha, post_beta)
            ci_upper = stats.beta.ppf(0.975, post_alpha, post_beta)
            
            # Probability of superiority over control
            if arm_name != "Control":
                # Sample from posteriors
                samples_treatment = np.random.beta(post_alpha, post_beta, 10000)
                
                if "Control" in self.arm_data and self.arm_data["Control"]['n'] > 0:
                    control_r = self.arm_data["Control"]['responses']
                    control_n = self.arm_data["Control"]['n']
                    samples_control = np.random.beta(1 + control_r, 1 + control_n - control_r, 10000)
                else:
                    samples_control = np.full(10000, control_rate)
                
                prob_superior = np.mean(samples_treatment > samples_control)
            else:
                prob_superior = None
            
            # Biomarker analysis
            biomarker_rates = {}
            for bio_idx, count in arm_data['biomarker_counts'].items():
                if count > 0:
                    # Approximate response rate in biomarker subgroup
                    biomarker_rates[f"Biomarker_{bio_idx}"] = {
                        'n': count,
                        'proportion': count / n
                    }
            
            # Decision
            if arm_name != "Control":
                if prob_superior > 0.95:
                    decision = "Graduate (Superior to control)"
                elif prob_superior < 0.2:
                    decision = "Terminate (Futility)"
                else:
                    decision = "Continue Enrollment"
            else:
                decision = "Control Arm"
            
            results[arm_name] = {
                'n': n,
                'responses': r,
                'observed_rate': float(observed_rate),
                'posterior_mean': float(post_mean),
                'credible_interval': [float(ci_lower), float(ci_upper)],
                'prob_superior_to_control': float(prob_superior) if prob_superior else None,
                'biomarker_distribution': biomarker_rates,
                'decision': decision
            }
        
        # Overall summary
        total_n = sum(d['n'] for d in self.arm_data.values())
        total_responses = sum(d['responses'] for d in self.arm_data.values())
        
        results['summary'] = {
            'n_arms': len([a for a in self.arm_data if a != "Control"]),
            'total_n': total_n,
            'total_responses': total_responses,
            'overall_response_rate': total_responses / total_n if total_n > 0 else 0,
            'shared_control': self.design.shared_control,
            'control_n': self.arm_data["Control"]['n'] if "Control" in self.arm_data else 0
        }
        
        return results


class PlatformTrial:
    """
    Platform trial implementation with perpetual enrollment.
    
    Allows arms to enter and leave the trial over time.
    """
    
    def __init__(self, design: PlatformDesign):
        """
        Initialize platform trial.
        
        Args:
            design: Platform trial design configuration
        """
        self.design = design
        self.current_time = 0  # months
        self.arms = {}
        self.control_arm = {
            'name': 'Control',
            'n': 0,
            'responses': 0,
            'start_time': 0,
            'status': TrialStatus.ENROLLING
        }
        
        # Initialize starting arms
        for i in range(design.initial_arms):
            self.add_arm(f"Arm_{i}", 0)
    
    def add_arm(self, arm_name: str, entry_time: int) -> None:
        """
        Add new treatment arm to platform.
        
        Args:
            arm_name: Name of new arm
            entry_time: Time (months) when arm enters
        """
        if len([a for a in self.arms.values() if a['status'] == TrialStatus.ENROLLING]) >= self.design.max_concurrent_arms:
            raise ValueError(f"Maximum concurrent arms ({self.design.max_concurrent_arms}) reached")
        
        self.arms[arm_name] = {
            'name': arm_name,
            'n': 0,
            'responses': 0,
            'start_time': entry_time,
            'status': TrialStatus.ENROLLING
        }
    
    def update_enrollment(self, months: int = 1) -> Dict[str, int]:
        """
        Update enrollment for specified time period.
        
        Args:
            months: Number of months to simulate
            
        Returns:
            Number of patients enrolled per arm
        """
        enrolled = {}
        
        for _ in range(months):
            self.current_time += 1
            
            # Enroll patients
            n_patients = np.random.poisson(self.design.enrollment_rate)
            
            # Allocate to arms (equal randomization among active arms)
            active_arms = [a for a in self.arms.values() 
                          if a['status'] == TrialStatus.ENROLLING]
            
            if active_arms:
                # Include control in allocation
                all_arms = active_arms + [self.control_arm]
                n_per_arm = n_patients // len(all_arms)
                remainder = n_patients % len(all_arms)
                
                for i, arm in enumerate(all_arms):
                    n_enrolled = n_per_arm + (1 if i < remainder else 0)
                    arm['n'] += n_enrolled
                    
                    # Simulate responses (simplified)
                    if arm['name'] == 'Control':
                        response_rate = self.design.control_response_rate
                    else:
                        # Random response rate for treatment arms (for simulation)
                        response_rate = np.random.uniform(0.15, 0.35)
                    
                    arm['responses'] += np.random.binomial(n_enrolled, response_rate)
                    
                    if arm['name'] not in enrolled:
                        enrolled[arm['name']] = 0
                    enrolled[arm['name']] += n_enrolled
        
        return enrolled
    
    def perform_interim_analysis(self) -> Dict[str, Any]:
        """
        Perform interim analysis and make arm decisions.
        
        Returns:
            Analysis results and decisions
        """
        results = {}
        
        # Analyze each arm
        for arm_name, arm_data in self.arms.items():
            if arm_data['status'] != TrialStatus.ENROLLING:
                continue
            
            if arm_data['n'] < 20:  # Minimum sample size
                results[arm_name] = {
                    'decision': 'Continue (insufficient data)',
                    'n': arm_data['n']
                }
                continue
            
            # Bayesian analysis
            n = arm_data['n']
            r = arm_data['responses']
            
            # Posterior (Beta(1,1) prior)
            post_alpha = 1 + r
            post_beta = 1 + n - r
            
            # Compare to control
            control_n = self.control_arm['n']
            control_r = self.control_arm['responses']
            
            if control_n > 0:
                # Sample from posteriors
                samples_treatment = np.random.beta(post_alpha, post_beta, 10000)
                samples_control = np.random.beta(1 + control_r, 1 + control_n - control_r, 10000)
                
                prob_superior = np.mean(samples_treatment > samples_control)
            else:
                # Use historical control rate
                samples_treatment = np.random.beta(post_alpha, post_beta, 10000)
                prob_superior = np.mean(samples_treatment > self.design.control_response_rate)
            
            # Make decision
            if prob_superior > self.design.graduation_threshold:
                decision = "Graduate (Success)"
                arm_data['status'] = TrialStatus.GRADUATED
            elif prob_superior < self.design.futility_threshold:
                decision = "Terminate (Futility)"
                arm_data['status'] = TrialStatus.TERMINATED
            else:
                decision = "Continue Enrollment"
            
            results[arm_name] = {
                'n': n,
                'responses': r,
                'observed_rate': r/n if n > 0 else 0,
                'prob_superior': float(prob_superior),
                'decision': decision,
                'months_in_trial': self.current_time - arm_data['start_time']
            }
        
        # Control arm summary
        results['Control'] = {
            'n': self.control_arm['n'],
            'responses': self.control_arm['responses'],
            'observed_rate': self.control_arm['responses'] / self.control_arm['n'] 
                           if self.control_arm['n'] > 0 else 0
        }
        
        # Overall summary
        results['summary'] = {
            'current_time': self.current_time,
            'n_active_arms': len([a for a in self.arms.values() 
                                 if a['status'] == TrialStatus.ENROLLING]),
            'n_graduated': len([a for a in self.arms.values() 
                              if a['status'] == TrialStatus.GRADUATED]),
            'n_terminated': len([a for a in self.arms.values() 
                               if a['status'] == TrialStatus.TERMINATED]),
            'total_n': sum(a['n'] for a in self.arms.values()) + self.control_arm['n']
        }
        
        return results
    
    def simulate_platform(
        self,
        duration_months: int,
        new_arms_schedule: Optional[Dict[int, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simulate platform trial over time.
        
        Args:
            duration_months: Total trial duration in months
            new_arms_schedule: Dictionary of {month: arm_name} for new arms
            
        Returns:
            List of interim analysis results
        """
        results = []
        new_arms_schedule = new_arms_schedule or {}
        
        for month in range(duration_months):
            # Add new arms if scheduled
            if month in new_arms_schedule:
                try:
                    self.add_arm(new_arms_schedule[month], month)
                except ValueError:
                    pass  # Max arms reached
            
            # Enroll patients
            self.update_enrollment(1)
            
            # Perform interim analysis every X months
            if month > 0 and month % self.design.interim_frequency == 0:
                analysis = self.perform_interim_analysis()
                analysis['month'] = month
                results.append(analysis)
        
        return results