"""Tests for Bayesian statistical engine."""

import pytest
import numpy as np
from backend.statistical.bayesian import BayesianEngine


class TestBayesianEngine:
    """Test suite for Bayesian statistical engine."""
    
    @pytest.fixture
    def engine(self):
        """Create BayesianEngine instance."""
        return BayesianEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert 'uninformative' in engine.prior_library
        assert 'skeptical' in engine.prior_library
        assert 'optimistic' in engine.prior_library
    
    def test_bayesian_t_test_basic(self, engine):
        """Test basic Bayesian t-test functionality."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 50)
        group2 = np.random.normal(0.5, 1, 50)
        
        result = engine.bayesian_t_test(
            group1=group1,
            group2=group2,
            prior_type='uninformative'
        )
        
        assert 'posterior_mean' in result
        assert 'posterior_std' in result
        assert 'credible_interval' in result
        assert 'prob_positive_effect' in result
        assert 'bayes_factor' in result
        assert 'effect_size' in result
        
        # Should detect positive effect
        assert result['prob_positive_effect'] > 0.8
        assert result['credible_interval']['lower'] > 0
    
    def test_bayesian_t_test_with_rope(self, engine):
        """Test Bayesian t-test with ROPE analysis."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 100)
        group2 = np.random.normal(0.1, 1, 100)  # Small effect
        
        result = engine.bayesian_t_test(
            group1=group1,
            group2=group2,
            rope=(-0.2, 0.2)
        )
        
        assert 'rope_analysis' in result
        assert result['rope_analysis'] is not None
        assert 'prob_in_rope' in result['rope_analysis']
        assert 'prob_below_rope' in result['rope_analysis']
        assert 'prob_above_rope' in result['rope_analysis']
        
        # Small effect should be mostly in ROPE
        assert result['rope_analysis']['prob_in_rope'] > 0.5
    
    def test_bayesian_t_test_different_priors(self, engine):
        """Test Bayesian t-test with different priors."""
        np.random.seed(42)
        group1 = np.random.normal(0, 1, 30)
        group2 = np.random.normal(0.3, 1, 30)
        
        # Test with different priors
        priors = ['uninformative', 'skeptical', 'optimistic']
        results = {}
        
        for prior in priors:
            results[prior] = engine.bayesian_t_test(
                group1=group1,
                group2=group2,
                prior_type=prior
            )
        
        # Skeptical prior should give lower probability
        assert results['skeptical']['prob_positive_effect'] < \
               results['optimistic']['prob_positive_effect']
        
        # All should detect some effect
        for prior in priors:
            assert 0 < results[prior]['prob_positive_effect'] < 1
    
    def test_bayesian_t_test_summary_stats(self, engine):
        """Test Bayesian t-test with summary statistics."""
        result = engine.bayesian_t_test(
            n1=50,
            n2=50,
            mean1=0,
            mean2=0.5,
            std1=1,
            std2=1,
            prior_type='uninformative'
        )
        
        assert 'posterior_mean' in result
        assert result['observed_difference'] == 0.5
        assert result['sample_sizes']['n1'] == 50
        assert result['sample_sizes']['n2'] == 50
    
    def test_bayesian_proportion_test(self, engine):
        """Test Bayesian proportion test."""
        result = engine.bayesian_proportion_test(
            successes1=20,
            n1=100,
            successes2=30,
            n2=100,
            prior_alpha=1,
            prior_beta=1
        )
        
        assert 'posterior_mean_diff' in result
        assert 'credible_interval' in result
        assert 'prob_positive_effect' in result
        assert 'risk_ratio' in result
        assert 'odds_ratio' in result
        
        # Should detect higher proportion in group 2
        assert result['prob_positive_effect'] > 0.8
        assert result['risk_ratio']['mean'] > 1
    
    def test_bayesian_proportion_no_difference(self, engine):
        """Test Bayesian proportion with no difference."""
        result = engine.bayesian_proportion_test(
            successes1=25,
            n1=100,
            successes2=26,
            n2=100
        )
        
        # Should not detect strong difference
        assert 0.3 < result['prob_positive_effect'] < 0.7
        assert result['credible_interval']['lower'] < 0
        assert result['credible_interval']['upper'] > 0
    
    def test_bayesian_anova(self, engine):
        """Test Bayesian one-way ANOVA."""
        np.random.seed(42)
        groups = [
            np.random.normal(0, 1, 30),
            np.random.normal(0.5, 1, 30),
            np.random.normal(1, 1, 30)
        ]
        
        result = engine.bayesian_anova(groups=groups)
        
        assert 'bayes_factor' in result
        assert 'eta_squared' in result
        assert 'group_means' in result
        assert 'pairwise_comparisons' in result
        assert result['n_groups'] == 3
        assert result['n_total'] == 90
        
        # Should detect group differences
        assert result['bayes_factor'] > 1
        assert result['eta_squared'] > 0.05
    
    def test_bayesian_anova_no_difference(self, engine):
        """Test Bayesian ANOVA with no group differences."""
        np.random.seed(42)
        groups = [
            np.random.normal(0, 1, 25),
            np.random.normal(0, 1, 25),
            np.random.normal(0, 1, 25),
            np.random.normal(0, 1, 25)
        ]
        
        result = engine.bayesian_anova(groups=groups)
        
        # Should not detect strong differences
        assert result['bayes_factor'] < 3
        assert result['eta_squared'] < 0.06  # Small effect
    
    def test_bayesian_correlation(self, engine):
        """Test Bayesian correlation analysis."""
        np.random.seed(42)
        n = 100
        # Create correlated data
        mean = [0, 0]
        cov = [[1, 0.6], [0.6, 1]]
        data = np.random.multivariate_normal(mean, cov, n)
        
        result = engine.bayesian_correlation(
            x=data[:, 0],
            y=data[:, 1]
        )
        
        assert 'correlation' in result
        assert 'posterior_mean' in result
        assert 'credible_interval' in result
        assert 'prob_positive' in result
        assert 'bayes_factor' in result
        assert 'r_squared' in result
        
        # Should detect positive correlation
        assert result['prob_positive'] > 0.95
        assert result['credible_interval']['lower'] > 0
        assert 0.4 < result['correlation'] < 0.8
    
    def test_bayesian_correlation_no_relationship(self, engine):
        """Test Bayesian correlation with no relationship."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        
        result = engine.bayesian_correlation(x=x, y=y)
        
        # Should not detect strong correlation
        assert abs(result['correlation']) < 0.3
        assert result['credible_interval']['lower'] < 0
        assert result['credible_interval']['upper'] > 0
        assert result['bayes_factor'] < 3
    
    def test_bayesian_power_analysis_two_sample(self, engine):
        """Test Bayesian power analysis for two-sample design."""
        result = engine.bayesian_power_analysis(
            effect_size=0.5,
            n=50,
            prior_type='uninformative',
            design_type='two_sample',
            success_criterion=0.95
        )
        
        assert 'bayesian_power' in result
        assert 'required_n_for_criterion' in result
        assert 0 <= result['bayesian_power'] <= 1
        assert result['required_n_for_criterion'] > 0
    
    def test_bayesian_power_analysis_paired(self, engine):
        """Test Bayesian power analysis for paired design."""
        result = engine.bayesian_power_analysis(
            effect_size=0.5,
            n=30,
            design_type='paired',
            success_criterion=0.95
        )
        
        assert 'bayesian_power' in result
        assert result['design_type'] == 'paired'
        assert 0 <= result['bayesian_power'] <= 1
    
    def test_bayesian_power_analysis_correlation(self, engine):
        """Test Bayesian power analysis for correlation."""
        result = engine.bayesian_power_analysis(
            effect_size=0.4,
            n=50,
            design_type='correlation',
            success_criterion=0.95
        )
        
        assert 'bayesian_power' in result
        assert result['design_type'] == 'correlation'
        assert 0 <= result['bayesian_power'] <= 1
    
    def test_bayes_factor_interpretation(self, engine):
        """Test Bayes Factor interpretation."""
        interpretations = {
            0.5: "Evidence for null",
            2: "Anecdotal evidence",
            5: "Moderate evidence",
            15: "Strong evidence",
            50: "Very strong evidence",
            150: "Decisive evidence"
        }
        
        for bf, expected_phrase in interpretations.items():
            interp = engine._interpret_bayes_factor(bf)
            assert any(phrase in interp for phrase in expected_phrase.split())
    
    def test_prior_library_structure(self, engine):
        """Test prior library structure."""
        for prior_name, prior_params in engine.prior_library.items():
            assert 'description' in prior_params
            
            if prior_name != 'historical':
                # Non-historical priors should have values
                assert prior_params['mu_mean'] is not None or prior_name == 'historical'
    
    def test_edge_cases(self, engine):
        """Test edge cases and error handling."""
        # Very small sample size
        result = engine.bayesian_t_test(
            n1=5,
            n2=5,
            mean1=0,
            mean2=1,
            std1=1,
            std2=1
        )
        assert 'posterior_mean' in result
        
        # Equal groups (no effect)
        group = np.random.normal(0, 1, 20)
        result = engine.bayesian_t_test(
            group1=group,
            group2=group
        )
        assert abs(result['posterior_mean']) < 0.1
        
        # Perfect correlation
        x = np.arange(20)
        result = engine.bayesian_correlation(x=x, y=x)
        assert result['correlation'] > 0.99