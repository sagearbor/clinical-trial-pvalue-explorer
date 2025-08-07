"""
Comprehensive pytest tests for all statistical test implementations.

Tests all 8 statistical tests:
- TwoSampleTTest, ChiSquareTest, OneWayANOVA, CorrelationTest (original)
- ANCOVATest, FishersExactTest, LogisticRegressionTest, RepeatedMeasuresANOVA (new)
"""

import pytest
import numpy as np
from src.statistical_tests import (
    get_factory, TwoSampleTTest, ChiSquareTest, OneWayANOVA, CorrelationTest,
    ANCOVATest, FishersExactTest, LogisticRegressionTest, RepeatedMeasuresANOVA
)


class TestStatisticalTestFactory:
    """Test the factory pattern for statistical tests."""
    
    def test_factory_initialization(self):
        """Test factory creates successfully."""
        factory = get_factory()
        assert factory is not None
    
    def test_all_tests_registered(self):
        """Test all 8 statistical tests are registered."""
        factory = get_factory()
        available_tests = factory.get_available_tests()
        
        # Should have at least these core test types
        core_tests = ['two_sample_t_test', 'chi_square', 'one_way_anova', 'correlation',
                     'ancova', 'fishers_exact', 'logistic_regression', 'repeated_measures_anova']
        
        # available_tests is a list of strings, not dictionaries
        for test_type in core_tests:
            assert test_type in available_tests, f"{test_type} not registered in factory"
    
    def test_factory_returns_correct_classes(self):
        """Test factory returns correct test class instances."""
        factory = get_factory()
        
        assert isinstance(factory.get_test('two_sample_t_test'), TwoSampleTTest)
        assert isinstance(factory.get_test('chi_square'), ChiSquareTest)
        assert isinstance(factory.get_test('one_way_anova'), OneWayANOVA)
        assert isinstance(factory.get_test('correlation'), CorrelationTest)
        assert isinstance(factory.get_test('ancova'), ANCOVATest)
        assert isinstance(factory.get_test('fishers_exact'), FishersExactTest)
        assert isinstance(factory.get_test('logistic_regression'), LogisticRegressionTest)
        assert isinstance(factory.get_test('repeated_measures_anova'), RepeatedMeasuresANOVA)


class TestTwoSampleTTest:
    """Test two-sample t-test implementation."""
    
    def test_valid_parameters(self):
        """Test with valid parameters."""
        test = TwoSampleTTest()
        p_val, error = test.calculate_p_value(N_total=100, cohens_d=0.5)
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1
    
    def test_power_calculation(self):
        """Test power calculation."""
        test = TwoSampleTTest()
        power, error = test.calculate_power(N_total=100, cohens_d=0.5, alpha=0.05)
        
        assert error is None
        assert power is not None
        assert 0 <= power <= 1
    
    def test_invalid_parameters(self):
        """Test parameter validation."""
        test = TwoSampleTTest()
        
        # Invalid N
        p_val, error = test.calculate_p_value(N_total=1, cohens_d=0.5)
        assert error is not None
        assert p_val is None
        
        # Missing parameters
        p_val, error = test.calculate_p_value(N_total=100)
        assert error is not None


class TestANCOVATest:
    """Test ANCOVA implementation."""
    
    def test_valid_ancova_calculation(self):
        """Test ANCOVA with valid parameters."""
        test = ANCOVATest()
        p_val, error = test.calculate_p_value(
            N_total=100, cohens_d=0.5, covariate_correlation=0.3
        )
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1
    
    def test_ancova_power_benefit(self):
        """Test ANCOVA provides power benefit over t-test."""
        ancova = ANCOVATest()
        ttest = TwoSampleTTest()
        
        # Compare power with same effect size
        ancova_power, _ = ancova.calculate_power(
            N_total=100, cohens_d=0.5, covariate_correlation=0.5, alpha=0.05
        )
        ttest_power, _ = ttest.calculate_power(
            N_total=100, cohens_d=0.5, alpha=0.05
        )
        
        # ANCOVA should have higher power due to covariate adjustment
        assert ancova_power > ttest_power
    
    def test_invalid_covariate_correlation(self):
        """Test invalid covariate correlation."""
        test = ANCOVATest()
        
        # Correlation outside [-1, 1]
        p_val, error = test.calculate_p_value(
            N_total=100, cohens_d=0.5, covariate_correlation=1.5
        )
        assert error is not None
        assert "correlation must be between -1 and 1" in error


class TestFishersExactTest:
    """Test Fisher's exact test implementation."""
    
    def test_valid_fishers_exact(self):
        """Test Fisher's exact with valid contingency table."""
        test = FishersExactTest()
        table = [[10, 5], [8, 12]]
        
        p_val, error = test.calculate_p_value(contingency_table=table)
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1
    
    def test_power_approximation(self):
        """Test power calculation approximation."""
        test = FishersExactTest()
        table = [[15, 5], [8, 12]]
        
        power, error = test.calculate_power(contingency_table=table, alpha=0.05)
        
        assert error is None
        assert power is not None
        assert 0 <= power <= 1
    
    def test_invalid_table_format(self):
        """Test invalid contingency table formats."""
        test = FishersExactTest()
        
        # Wrong shape
        p_val, error = test.calculate_p_value(contingency_table=[[10, 5, 3], [8, 12, 4]])
        assert error is not None
        assert "2x2 contingency table" in error
        
        # Negative values
        p_val, error = test.calculate_p_value(contingency_table=[[10, -5], [8, 12]])
        assert error is not None
        assert "non-negative" in error


class TestLogisticRegressionTest:
    """Test logistic regression implementation."""
    
    def test_valid_logistic_regression(self):
        """Test logistic regression with valid parameters."""
        test = LogisticRegressionTest()
        
        p_val, error = test.calculate_p_value(
            N_total=100, baseline_rate=0.3, odds_ratio=2.0
        )
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1
    
    def test_odds_ratio_effects(self):
        """Test different odds ratios produce expected effects."""
        test = LogisticRegressionTest()
        
        # Large odds ratio should give smaller p-value
        p_val_large, _ = test.calculate_p_value(
            N_total=100, baseline_rate=0.3, odds_ratio=5.0
        )
        p_val_small, _ = test.calculate_p_value(
            N_total=100, baseline_rate=0.3, odds_ratio=1.2
        )
        
        assert p_val_large < p_val_small
    
    def test_invalid_baseline_rate(self):
        """Test invalid baseline rate."""
        test = LogisticRegressionTest()
        
        # Rate outside (0, 1)
        p_val, error = test.calculate_p_value(
            N_total=100, baseline_rate=1.5, odds_ratio=2.0
        )
        assert error is not None
        assert "between 0 and 1" in error


class TestRepeatedMeasuresANOVA:
    """Test repeated measures ANOVA implementation."""
    
    def test_valid_rm_anova(self):
        """Test repeated measures ANOVA with valid parameters."""
        test = RepeatedMeasuresANOVA()
        
        p_val, error = test.calculate_p_value(
            N_subjects=30, n_timepoints=4, cohens_f=0.25, 
            correlation_between_measures=0.6
        )
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1
    
    def test_repeated_measures_power_benefit(self):
        """Test RM-ANOVA has power benefit from correlation."""
        test = RepeatedMeasuresANOVA()
        
        # Higher correlation should increase power
        power_high_corr, _ = test.calculate_power(
            N_subjects=30, n_timepoints=4, cohens_f=0.25,
            correlation_between_measures=0.8, alpha=0.05
        )
        power_low_corr, _ = test.calculate_power(
            N_subjects=30, n_timepoints=4, cohens_f=0.25,
            correlation_between_measures=0.2, alpha=0.05
        )
        
        assert power_high_corr > power_low_corr
    
    def test_invalid_timepoints(self):
        """Test invalid number of timepoints."""
        test = RepeatedMeasuresANOVA()
        
        # Less than 2 timepoints
        p_val, error = test.calculate_p_value(
            N_subjects=30, n_timepoints=1, cohens_f=0.25,
            correlation_between_measures=0.6
        )
        assert error is not None
        assert "at least 2" in error


class TestChiSquareTest:
    """Test chi-square test (existing functionality)."""
    
    def test_valid_chi_square(self):
        """Test chi-square with valid contingency table."""
        test = ChiSquareTest()
        table = [[20, 30], [25, 25]]
        
        p_val, error = test.calculate_p_value(contingency_table=table)
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1


class TestOneWayANOVA:
    """Test one-way ANOVA (existing functionality)."""
    
    def test_valid_anova(self):
        """Test ANOVA with valid group data."""
        test = OneWayANOVA()
        
        # Generate sample data for 3 groups
        np.random.seed(42)
        group1 = np.random.normal(10, 2, 30).tolist()
        group2 = np.random.normal(12, 2, 30).tolist()
        group3 = np.random.normal(11, 2, 30).tolist()
        
        groups = [group1, group2, group3]
        
        p_val, error = test.calculate_p_value(groups=groups)
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1


class TestCorrelationTest:
    """Test correlation analysis (existing functionality)."""
    
    def test_valid_correlation(self):
        """Test correlation with valid data."""
        test = CorrelationTest()
        
        # Generate correlated data
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = 0.5 * x + np.random.normal(0, 1, 50)
        
        p_val, error = test.calculate_p_value(
            x_values=x.tolist(), y_values=y.tolist(), correlation_type='pearson'
        )
        
        assert error is None
        assert p_val is not None
        assert 0 <= p_val <= 1


# Integration tests
class TestStatisticalTestIntegration:
    """Test integration between different components."""
    
    def test_all_tests_have_required_methods(self):
        """Test all tests implement required methods."""
        factory = get_factory()
        test_types = ['two_sample_t_test', 'chi_square', 'one_way_anova', 'correlation',
                     'ancova', 'fishers_exact', 'logistic_regression', 'repeated_measures_anova']
        
        for test_type in test_types:
            test_instance = factory.get_test(test_type)
            
            # All tests should have these methods
            assert hasattr(test_instance, 'calculate_p_value')
            assert hasattr(test_instance, 'calculate_power')
            assert hasattr(test_instance, 'validate_params')
            assert hasattr(test_instance, 'get_required_params')
    
    def test_parameter_validation_consistency(self):
        """Test parameter validation is consistent."""
        factory = get_factory()
        
        # All tests should reject invalid parameters gracefully
        for test_type in ['two_sample_t_test', 'ancova', 'logistic_regression']:
            test_instance = factory.get_test(test_type)
            
            # Test with missing parameters
            p_val, error = test_instance.calculate_p_value()
            assert error is not None
            assert p_val is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])