"""Domain-specific specializations for different research fields."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats


class ResearchDomain(Enum):
    """Supported research domains."""
    CLINICAL = "clinical"
    PSYCHOLOGY = "psychology"
    EDUCATION = "education"
    MARKETING = "marketing"
    ENGINEERING = "engineering"
    SOCIAL_SCIENCE = "social_science"


@dataclass
class DomainConfig:
    """Configuration for domain-specific analysis."""
    domain: ResearchDomain
    default_alpha: float
    default_power: float
    effect_size_thresholds: Dict[str, float]
    common_designs: List[str]
    terminology: Dict[str, str]
    regulatory_requirements: Optional[Dict[str, Any]] = None


class DomainSpecialization:
    """Base class for domain-specific specializations."""
    
    def __init__(self, config: DomainConfig):
        """Initialize with domain configuration."""
        self.config = config
    
    def get_domain_prompt(self, study_description: str) -> str:
        """Get domain-specific LLM prompt."""
        raise NotImplementedError
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters for domain requirements."""
        raise NotImplementedError
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide domain-specific interpretation."""
        raise NotImplementedError
    
    def get_reporting_template(self) -> Dict[str, Any]:
        """Get domain-specific reporting template."""
        raise NotImplementedError


class ClinicalTrialSpecialization(DomainSpecialization):
    """Specialization for clinical trials."""
    
    def __init__(self):
        """Initialize clinical trial specialization."""
        config = DomainConfig(
            domain=ResearchDomain.CLINICAL,
            default_alpha=0.05,
            default_power=0.80,
            effect_size_thresholds={
                'small': 0.2,
                'medium': 0.5,
                'large': 0.8,
                'clinically_meaningful': 0.3
            },
            common_designs=[
                'parallel_group_rct',
                'crossover',
                'factorial',
                'adaptive',
                'cluster_randomized',
                'stepped_wedge'
            ],
            terminology={
                'subjects': 'patients',
                'treatment': 'intervention',
                'outcome': 'endpoint',
                'group': 'arm',
                'effect': 'treatment_effect'
            },
            regulatory_requirements={
                'fda': {
                    'alpha': 0.05,
                    'power_minimum': 0.80,
                    'multiplicity_adjustment': True,
                    'itt_analysis': True
                },
                'ema': {
                    'alpha': 0.05,
                    'power_minimum': 0.80,
                    'estimands_framework': True
                }
            }
        )
        super().__init__(config)
    
    def get_domain_prompt(self, study_description: str) -> str:
        """Get clinical trial specific prompt."""
        return f"""
        Analyze this clinical trial design as a medical statistician:
        
        {study_description}
        
        Consider:
        1. Primary and secondary endpoints
        2. Safety considerations
        3. Regulatory requirements (FDA/EMA)
        4. Clinical meaningfulness of effect size
        5. Patient population characteristics
        6. Multiplicity adjustments if multiple endpoints
        7. Interim analyses and stopping rules
        8. Missing data handling (ITT vs PP)
        
        Recommend appropriate statistical test considering:
        - Endpoint type (continuous, binary, time-to-event, ordinal)
        - Study phase (I, II, III, IV)
        - Design (superiority, non-inferiority, equivalence)
        - Multiplicity concerns
        """
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate clinical trial parameters."""
        # Check regulatory requirements
        if parameters.get('alpha', 0.05) > 0.05:
            return False, "Alpha level exceeds regulatory standard (0.05)"
        
        if parameters.get('power', 0.8) < 0.8:
            return False, "Power below regulatory minimum (0.80)"
        
        # Check for safety monitoring
        if parameters.get('n_total', 0) > 100 and 'safety_monitoring' not in parameters:
            return False, "Safety monitoring plan required for trials >100 patients"
        
        # Non-inferiority specific
        if parameters.get('design_type') == 'non_inferiority':
            if 'margin' not in parameters:
                return False, "Non-inferiority margin must be specified"
            if parameters.get('margin', 0) <= 0:
                return False, "Non-inferiority margin must be positive"
        
        return True, None
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide clinical interpretation."""
        p_value = results.get('p_value', 1)
        effect_size = results.get('effect_size', 0)
        ci = results.get('confidence_interval', {})
        
        # Statistical significance
        if p_value < 0.001:
            stat_sig = "highly statistically significant"
        elif p_value < 0.05:
            stat_sig = "statistically significant"
        else:
            stat_sig = "not statistically significant"
        
        # Clinical significance
        abs_effect = abs(effect_size)
        if abs_effect < self.config.effect_size_thresholds['clinically_meaningful']:
            clin_sig = "below clinically meaningful threshold"
        elif abs_effect < self.config.effect_size_thresholds['medium']:
            clin_sig = "clinically meaningful"
        else:
            clin_sig = "clinically important"
        
        # NNT calculation if binary outcome
        if results.get('outcome_type') == 'binary':
            control_rate = results.get('control_rate', 0.2)
            treatment_rate = results.get('treatment_rate', 0.15)
            risk_diff = abs(treatment_rate - control_rate)
            if risk_diff > 0:
                nnt = int(1 / risk_diff)
                nnt_text = f" Number needed to treat (NNT): {nnt}."
            else:
                nnt_text = ""
        else:
            nnt_text = ""
        
        interpretation = f"""
        Clinical Trial Results Interpretation:
        
        The primary endpoint analysis shows the treatment effect is {stat_sig} 
        (p={p_value:.4f}) and {clin_sig} (effect size={effect_size:.3f}).
        
        95% Confidence Interval: [{ci.get('lower', 'NA'):.3f}, {ci.get('upper', 'NA'):.3f}]
        
        This confidence interval {'excludes' if ci.get('lower', -1) > 0 or ci.get('upper', 1) < 0 else 'includes'} 
        the null effect, providing {'evidence of' if ci.get('lower', -1) > 0 else 'no clear evidence of'} 
        treatment benefit.{nnt_text}
        
        Regulatory Perspective:
        {'✓ Meets FDA statistical significance criteria (p<0.05)' if p_value < 0.05 else '✗ Does not meet FDA criteria'}
        {'✓ Clinically meaningful effect size' if abs_effect >= self.config.effect_size_thresholds['clinically_meaningful'] else '✗ Effect size below clinical threshold'}
        
        Recommendation: {'Consider regulatory submission' if p_value < 0.05 and abs_effect >= self.config.effect_size_thresholds['clinically_meaningful'] else 'Additional studies may be needed'}
        """
        
        return interpretation.strip()
    
    def get_reporting_template(self) -> Dict[str, Any]:
        """Get CONSORT-compliant reporting template."""
        return {
            'title': 'Clinical Trial Statistical Report',
            'sections': [
                {
                    'name': 'Trial Design',
                    'fields': ['design_type', 'randomization', 'blinding', 'allocation_concealment']
                },
                {
                    'name': 'Participants',
                    'fields': ['eligibility_criteria', 'recruitment_period', 'baseline_characteristics']
                },
                {
                    'name': 'Interventions',
                    'fields': ['treatment_description', 'control_description', 'concomitant_medications']
                },
                {
                    'name': 'Outcomes',
                    'fields': ['primary_endpoint', 'secondary_endpoints', 'safety_endpoints']
                },
                {
                    'name': 'Sample Size',
                    'fields': ['target_sample_size', 'power_calculation', 'assumptions']
                },
                {
                    'name': 'Statistical Methods',
                    'fields': ['analysis_populations', 'statistical_tests', 'missing_data_handling']
                },
                {
                    'name': 'Results',
                    'fields': ['participant_flow', 'baseline_data', 'outcomes_and_estimation', 'ancillary_analyses']
                },
                {
                    'name': 'Safety',
                    'fields': ['adverse_events', 'serious_adverse_events', 'discontinuations']
                }
            ],
            'regulatory_sections': ['protocol_deviations', 'data_monitoring_committee', 'interim_analyses']
        }


class PsychologySpecialization(DomainSpecialization):
    """Specialization for psychology research."""
    
    def __init__(self):
        """Initialize psychology specialization."""
        config = DomainConfig(
            domain=ResearchDomain.PSYCHOLOGY,
            default_alpha=0.05,
            default_power=0.80,
            effect_size_thresholds={
                'small': 0.2,
                'medium': 0.5,
                'large': 0.8,
                'minimal_important_difference': 0.41  # Based on meta-analyses
            },
            common_designs=[
                'between_subjects',
                'within_subjects',
                'mixed_design',
                'factorial',
                'repeated_measures',
                'longitudinal'
            ],
            terminology={
                'subjects': 'participants',
                'treatment': 'condition',
                'outcome': 'dependent_variable',
                'group': 'condition',
                'effect': 'effect_size'
            }
        )
        super().__init__(config)
    
    def get_domain_prompt(self, study_description: str) -> str:
        """Get psychology specific prompt."""
        return f"""
        Analyze this psychology research study:
        
        {study_description}
        
        Consider:
        1. Experimental design (between/within/mixed)
        2. Measurement reliability and validity
        3. Statistical assumptions (normality, homogeneity)
        4. Effect size importance (Cohen's conventions)
        5. Power analysis for replication
        6. Control for confounding variables
        7. Multiple comparisons corrections
        8. Practical significance
        
        Recommend appropriate test considering:
        - Scale of measurement (nominal, ordinal, interval, ratio)
        - Independence of observations
        - Sample size and power
        - Replication crisis considerations
        """
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide psychology-specific interpretation."""
        p_value = results.get('p_value', 1)
        effect_size = results.get('effect_size', 0)
        power = results.get('power', 0)
        
        # Replication probability (simplified)
        if p_value < 0.005:
            rep_prob = "high"
        elif p_value < 0.05:
            rep_prob = "moderate"
        else:
            rep_prob = "low"
        
        interpretation = f"""
        Psychology Research Results:
        
        Statistical Significance: p = {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})
        Effect Size: {effect_size:.3f} ({self._classify_effect_size(effect_size)})
        Statistical Power: {power:.2f} ({'adequate' if power >= 0.8 else 'underpowered'})
        
        Interpretation:
        {self._get_effect_interpretation(effect_size)}
        
        Replication Considerations:
        - Replication probability: {rep_prob}
        - {'Consider pre-registration for follow-up studies' if p_value < 0.05 else 'Insufficient evidence for effect'}
        - {'Well-powered study' if power >= 0.8 else 'Consider larger sample for replication'}
        
        Practical Significance:
        {'Effect exceeds minimal important difference' if abs(effect_size) >= self.config.effect_size_thresholds['minimal_important_difference'] else 'Effect below practical significance threshold'}
        """
        
        return interpretation.strip()
    
    def _classify_effect_size(self, d: float) -> str:
        """Classify effect size using Cohen's conventions."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _get_effect_interpretation(self, d: float) -> str:
        """Get detailed effect interpretation."""
        abs_d = abs(d)
        percentile = stats.norm.cdf(d) * 100
        
        if abs_d < 0.2:
            return f"The effect is negligible (d={d:.3f}), suggesting minimal practical impact."
        elif abs_d < 0.5:
            return f"A small effect (d={d:.3f}) was observed. The average person in the treatment group scores higher than {percentile:.0f}% of the control group."
        elif abs_d < 0.8:
            return f"A medium effect (d={d:.3f}) was observed. This represents a noticeable difference, with the treatment group average exceeding {percentile:.0f}% of controls."
        else:
            return f"A large effect (d={d:.3f}) was observed. This is a substantial difference, with the treatment group average exceeding {percentile:.0f}% of controls."
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate psychology study parameters."""
        # Check for adequate power
        if parameters.get('power', 0.8) < 0.8:
            return False, "Studies should be powered at ≥80% (replication crisis guidelines)"
        
        # Check for multiple comparisons
        if parameters.get('n_comparisons', 1) > 1 and 'correction_method' not in parameters:
            return False, "Multiple comparison correction required"
        
        return True, None


class EducationSpecialization(DomainSpecialization):
    """Specialization for education research."""
    
    def __init__(self):
        """Initialize education specialization."""
        config = DomainConfig(
            domain=ResearchDomain.EDUCATION,
            default_alpha=0.05,
            default_power=0.80,
            effect_size_thresholds={
                'small': 0.2,
                'medium': 0.5,
                'large': 0.8,
                'educationally_significant': 0.25  # Quarter SD improvement
            },
            common_designs=[
                'randomized_controlled_trial',
                'quasi_experimental',
                'cluster_randomized',
                'regression_discontinuity',
                'difference_in_differences',
                'pre_post'
            ],
            terminology={
                'subjects': 'students',
                'treatment': 'intervention',
                'outcome': 'achievement',
                'group': 'classroom',
                'effect': 'learning_gain'
            }
        )
        super().__init__(config)
    
    def get_domain_prompt(self, study_description: str) -> str:
        """Get education specific prompt."""
        return f"""
        Analyze this education research study:
        
        {study_description}
        
        Consider:
        1. Study design (RCT, quasi-experimental, observational)
        2. Clustering effects (students within classrooms/schools)
        3. Baseline achievement covariate
        4. Implementation fidelity
        5. Effect size in terms of months of learning
        6. Subgroup analyses (e.g., by prior achievement)
        7. Cost-effectiveness considerations
        
        Recommend appropriate test considering:
        - Nested data structure
        - Standardized test scores vs. other outcomes
        - Duration of intervention
        - What Works Clearinghouse standards
        """
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide education-specific interpretation."""
        effect_size = results.get('effect_size', 0)
        
        # Convert to months of learning (approximation)
        months_gain = effect_size * 9  # Assume 1 SD = 1 academic year
        
        # WWC evidence standards
        if abs(effect_size) >= 0.25:
            wwc_rating = "Substantively important"
        else:
            wwc_rating = "Not substantively important"
        
        interpretation = f"""
        Education Research Results:
        
        Effect Size: {effect_size:.3f} ({self._classify_effect_size(effect_size)})
        Learning Gain: {months_gain:.1f} months of additional learning
        WWC Rating: {wwc_rating}
        
        Practical Implications:
        {'This intervention shows promise for improving student achievement.' if effect_size >= 0.25 else 'Limited evidence of educational benefit.'}
        
        Cost-Benefit Consideration:
        An effect of {effect_size:.3f} SD represents {months_gain:.1f} months of learning,
        which should be weighed against implementation costs and resources.
        """
        
        return interpretation.strip()
    
    def _classify_effect_size(self, d: float) -> str:
        """Classify effect size for education."""
        abs_d = abs(d)
        if abs_d < 0.10:
            return "minimal"
        elif abs_d < 0.25:
            return "small"
        elif abs_d < 0.40:
            return "moderate"
        else:
            return "large"


class MarketingSpecialization(DomainSpecialization):
    """Specialization for marketing/business research."""
    
    def __init__(self):
        """Initialize marketing specialization."""
        config = DomainConfig(
            domain=ResearchDomain.MARKETING,
            default_alpha=0.05,
            default_power=0.80,
            effect_size_thresholds={
                'small': 0.01,  # 1% lift
                'medium': 0.05,  # 5% lift
                'large': 0.10,  # 10% lift
                'practically_significant': 0.02  # 2% lift often considered worthwhile
            },
            common_designs=[
                'ab_test',
                'multivariate_test',
                'factorial_design',
                'time_series',
                'cohort_analysis',
                'rct'
            ],
            terminology={
                'subjects': 'customers',
                'treatment': 'variant',
                'outcome': 'conversion',
                'group': 'segment',
                'effect': 'lift'
            }
        )
        super().__init__(config)
    
    def get_domain_prompt(self, study_description: str) -> str:
        """Get marketing specific prompt."""
        return f"""
        Analyze this marketing/business experiment:
        
        {study_description}
        
        Consider:
        1. Business metrics (conversion, revenue, retention)
        2. Sample size and statistical power
        3. Seasonality and external factors
        4. Customer segmentation effects
        5. Long-term vs. short-term impacts
        6. Practical significance (ROI)
        7. Multiple testing (if testing multiple metrics)
        
        Recommend appropriate test considering:
        - Metric type (binary, continuous, count)
        - Business importance
        - Cost of implementation
        - Minimum detectable effect
        """
    
    def interpret_results(self, results: Dict[str, Any]) -> str:
        """Provide marketing-specific interpretation."""
        p_value = results.get('p_value', 1)
        effect_size = results.get('effect_size', 0)
        
        # Convert to percentage lift
        lift = effect_size * 100
        
        # Business impact
        if abs(lift) >= 2:
            business_impact = "likely worthwhile to implement"
        elif abs(lift) >= 1:
            business_impact = "marginal benefit, consider costs"
        else:
            business_impact = "unlikely to justify implementation"
        
        interpretation = f"""
        Marketing Experiment Results:
        
        Statistical Significance: p = {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})
        Lift: {lift:.2f}% {'increase' if lift > 0 else 'decrease'}
        
        Business Recommendation:
        {business_impact.capitalize()}
        
        Confidence: {'High' if p_value < 0.01 else 'Moderate' if p_value < 0.05 else 'Low'}
        Risk: {'Low' if p_value < 0.05 and abs(lift) >= 2 else 'Consider running longer test'}
        
        Next Steps:
        {'Proceed with rollout' if p_value < 0.05 and lift >= 2 else 'Continue testing' if p_value < 0.10 else 'Abandon variant'}
        """
        
        return interpretation.strip()


class DomainFactory:
    """Factory for creating domain specializations."""
    
    SPECIALIZATIONS = {
        ResearchDomain.CLINICAL: ClinicalTrialSpecialization,
        ResearchDomain.PSYCHOLOGY: PsychologySpecialization,
        ResearchDomain.EDUCATION: EducationSpecialization,
        ResearchDomain.MARKETING: MarketingSpecialization
    }
    
    @classmethod
    def get_specialization(cls, domain: ResearchDomain) -> DomainSpecialization:
        """
        Get domain specialization instance.
        
        Args:
            domain: Research domain
            
        Returns:
            Domain specialization instance
        """
        specialization_class = cls.SPECIALIZATIONS.get(domain)
        if not specialization_class:
            raise ValueError(f"Unsupported domain: {domain}")
        
        return specialization_class()
    
    @classmethod
    def detect_domain(cls, study_description: str) -> ResearchDomain:
        """
        Detect research domain from study description.
        
        Args:
            study_description: Study description text
            
        Returns:
            Detected research domain
        """
        description_lower = study_description.lower()
        
        # Clinical keywords
        clinical_keywords = ['patient', 'treatment', 'drug', 'therapy', 'clinical', 
                           'disease', 'medical', 'surgery', 'diagnosis', 'placebo']
        if any(keyword in description_lower for keyword in clinical_keywords):
            return ResearchDomain.CLINICAL
        
        # Psychology keywords
        psych_keywords = ['behavior', 'cognitive', 'emotion', 'personality', 'mental',
                         'psychological', 'perception', 'memory', 'anxiety', 'depression']
        if any(keyword in description_lower for keyword in psych_keywords):
            return ResearchDomain.PSYCHOLOGY
        
        # Education keywords
        edu_keywords = ['student', 'learning', 'teaching', 'education', 'school',
                       'curriculum', 'achievement', 'classroom', 'instruction']
        if any(keyword in description_lower for keyword in edu_keywords):
            return ResearchDomain.EDUCATION
        
        # Marketing keywords
        marketing_keywords = ['customer', 'conversion', 'marketing', 'sales', 'revenue',
                            'campaign', 'advertisement', 'brand', 'purchase', 'retention']
        if any(keyword in description_lower for keyword in marketing_keywords):
            return ResearchDomain.MARKETING
        
        # Default to clinical
        return ResearchDomain.CLINICAL