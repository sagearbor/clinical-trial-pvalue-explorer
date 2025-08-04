"""
Comprehensive tests for CorrelationTest implementation.

Tests cover:
- Basic functionality (p-value and power calculations)
- Parameter validation
- Pearson vs Spearman correlation tests
- Edge cases and error handling
- Integration with factory pattern
- Real-world correlation examples
- Backwards compatibility
"""

import pytest
import numpy as np
from statistical_tests import CorrelationTest, get_factory
from scipy import stats


class TestCorrelationTest:
    """Test CorrelationTest implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.corr_test = CorrelationTest()
        
        # Sample data for testing - real-world inspired examples
        np.random.seed(42)  # For reproducible results
        
        # Strong positive correlation (height vs weight example)
        self.height = [65, 67, 70, 72, 69, 71, 68, 74, 66, 73]
        self.weight = [140, 155, 180, 200, 165, 190, 160, 210, 145, 195]
        
        # Moderate negative correlation (age vs reaction time)
        self.age = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        self.reaction_time = [250, 245, 260, 270, 275, 285, 295, 310, 320, 335]
        
        # Weak correlation (temperature vs ice cream sales - non-linear relationship)
        self.temperature = [65, 70, 75, 80, 85, 90, 95, 100, 60, 55]
        self.ice_cream_sales = [150, 180, 220, 280, 350, 420, 500, 600, 120, 100]
        
        # No correlation (random data)
        self.random_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.random_y = [15, 8, 23, 12, 7, 19, 3, 11, 25, 14]
        
        # Perfect correlation (for edge case testing)
        self.perfect_x = [1, 2, 3, 4, 5]
        self.perfect_y = [2, 4, 6, 8, 10]  # y = 2x
        
        # Constant values (for error case testing)
        self.constant_x = [5, 5, 5, 5, 5]
        self.constant_y = [10, 15, 20, 25, 30]
        
    def test_required_params(self):
        """Test required parameters are correctly defined."""
        required = self.corr_test.get_required_params()
        assert "x_values" in required
        assert "y_values" in required
        assert len(required) == 2
    
    def test_validate_params_valid_data(self):
        """Test parameter validation with valid data."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height, 
            y_values=self.weight
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_pearson_type(self):
        """Test parameter validation with Pearson correlation type."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height, 
            y_values=self.weight,
            correlation_type="pearson"
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_spearman_type(self):
        """Test parameter validation with Spearman correlation type."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.age, 
            y_values=self.reaction_time,
            correlation_type="spearman"
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_missing_x_values(self):
        """Test parameter validation fails when x_values is missing."""
        is_valid, error = self.corr_test.validate_params(y_values=self.weight)
        assert not is_valid
        assert "Missing required parameter: x_values" in error
    
    def test_validate_params_missing_y_values(self):
        """Test parameter validation fails when y_values is missing."""
        is_valid, error = self.corr_test.validate_params(x_values=self.height)
        assert not is_valid
        assert "Missing required parameter: y_values" in error
    
    def test_validate_params_invalid_x_type(self):
        """Test parameter validation fails with invalid x_values type."""
        is_valid, error = self.corr_test.validate_params(
            x_values="invalid", 
            y_values=self.weight
        )
        assert not is_valid
        assert "x_values must be a list, tuple, or numpy array" in error
    
    def test_validate_params_invalid_y_type(self):
        """Test parameter validation fails with invalid y_values type."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height, 
            y_values="invalid"
        )
        assert not is_valid
        assert "y_values must be a list, tuple, or numpy array" in error
    
    def test_validate_params_non_numeric_x(self):
        """Test parameter validation fails with non-numeric x_values."""
        is_valid, error = self.corr_test.validate_params(
            x_values=["a", "b", "c"], 
            y_values=self.weight[:3]
        )
        assert not is_valid
        assert "x_values must contain numeric values" in error
    
    def test_validate_params_non_numeric_y(self):
        """Test parameter validation fails with non-numeric y_values."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height[:3], 
            y_values=["a", "b", "c"]
        )
        assert not is_valid
        assert "y_values must contain numeric values" in error
    
    def test_validate_params_empty_arrays(self):
        """Test parameter validation fails with empty arrays."""
        is_valid, error = self.corr_test.validate_params(x_values=[], y_values=[])
        assert not is_valid
        assert "x_values cannot be empty" in error
    
    def test_validate_params_mismatched_lengths(self):
        """Test parameter validation fails with mismatched array lengths."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height[:5], 
            y_values=self.weight[:3]
        )
        assert not is_valid
        assert "x_values and y_values must have the same length" in error
    
    def test_validate_params_insufficient_data(self):
        """Test parameter validation fails with insufficient data points."""
        is_valid, error = self.corr_test.validate_params(
            x_values=[1, 2], 
            y_values=[3, 4]
        )
        assert not is_valid
        assert "Correlation requires at least 3 paired observations" in error
    
    def test_validate_params_infinite_values(self):
        """Test parameter validation fails with infinite values."""
        is_valid, error = self.corr_test.validate_params(
            x_values=[1, 2, float('inf')], 
            y_values=[3, 4, 5]
        )
        assert not is_valid
        assert "x_values must contain only finite numbers" in error
    
    def test_validate_params_nan_values(self):
        """Test parameter validation fails with NaN values."""
        is_valid, error = self.corr_test.validate_params(
            x_values=[1, 2, 3], 
            y_values=[3, 4, float('nan')]
        )
        assert not is_valid
        assert "y_values must contain only finite numbers" in error
    
    def test_validate_params_zero_variance_pearson(self):
        """Test parameter validation fails with zero variance for Pearson."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.constant_x, 
            y_values=self.constant_y,
            correlation_type="pearson"
        )
        assert not is_valid
        assert "zero variance" in error
    
    def test_validate_params_no_variation_spearman(self):
        """Test parameter validation fails with no variation for Spearman."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.constant_x, 
            y_values=self.constant_y,
            correlation_type="spearman"
        )
        assert not is_valid
        assert "no variation" in error
    
    def test_validate_params_invalid_correlation_type(self):
        """Test parameter validation fails with invalid correlation type."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height, 
            y_values=self.weight,
            correlation_type="invalid"
        )
        assert not is_valid
        assert "must be either 'pearson' or 'spearman'" in error
    
    def test_validate_params_invalid_alpha(self):
        """Test parameter validation fails with invalid alpha."""
        is_valid, error = self.corr_test.validate_params(
            x_values=self.height, 
            y_values=self.weight,
            alpha=1.5  # Invalid alpha > 1
        )
        assert not is_valid
        assert "between 0 and 1" in error
    
    def test_calculate_p_value_pearson_strong_positive(self):
        """Test p-value calculation for strong positive Pearson correlation."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.height,
            y_values=self.weight,
            correlation_type="pearson"
        )
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Strong correlation should be significant
        
        # Verify against scipy calculation
        expected_r, expected_p = stats.pearsonr(self.height, self.weight)
        assert abs(p_value - expected_p) < 1e-10
    
    def test_calculate_p_value_spearman_moderate_negative(self):
        """Test p-value calculation for moderate negative Spearman correlation."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.age,
            y_values=self.reaction_time,
            correlation_type="spearman"
        )
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        
        # Verify against scipy calculation
        expected_rho, expected_p = stats.spearmanr(self.age, self.reaction_time)
        assert abs(p_value - expected_p) < 1e-10
    
    def test_calculate_p_value_no_correlation(self):
        """Test p-value calculation with no correlation (should be high p-value)."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.random_x,
            y_values=self.random_y,
            correlation_type="pearson"
        )
        assert p_value is not None
        assert error is None
        assert p_value > 0.1  # No correlation should have high p-value
    
    def test_calculate_p_value_perfect_correlation_warning(self):
        """Test p-value calculation with perfect correlation returns warning."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.perfect_x,
            y_values=self.perfect_y,
            correlation_type="pearson"
        )
        assert p_value is not None
        assert error is not None
        assert "Perfect correlation detected" in error
        assert p_value < 0.001  # Perfect correlation should have very low p-value
    
    def test_calculate_p_value_default_pearson(self):
        """Test p-value calculation defaults to Pearson when correlation_type not specified."""
        p_value_default, error_default = self.corr_test.calculate_p_value(
            x_values=self.height,
            y_values=self.weight
        )
        
        p_value_pearson, error_pearson = self.corr_test.calculate_p_value(
            x_values=self.height,
            y_values=self.weight,
            correlation_type="pearson"
        )
        
        assert p_value_default == p_value_pearson
        assert error_default == error_pearson
    
    def test_calculate_p_value_invalid_params(self):
        """Test p-value calculation with invalid parameters."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.constant_x,
            y_values=self.constant_y
        )
        assert p_value is None
        assert error is not None
    
    def test_calculate_power_with_effect_size(self):
        """Test power calculation with provided effect size."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            effect_size=0.7,  # Strong correlation
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
        assert power > 0.8  # Strong effect should have high power
    
    def test_calculate_power_calculated_effect_size(self):
        """Test power calculation with calculated effect size."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_spearman(self):
        """Test power calculation for Spearman correlation."""
        power, error = self.corr_test.calculate_power(
            x_values=self.age,
            y_values=self.reaction_time,
            correlation_type="spearman",
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_default_alpha(self):
        """Test power calculation uses default alpha=0.05 when not provided."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_with_sample_size(self):
        """Test power calculation with specified sample size."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            n=50,
            effect_size=0.5,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_zero_correlation(self):
        """Test power calculation with zero correlation."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            effect_size=0.0,
            alpha=0.05
        )
        assert power is not None
        assert error is not None
        assert "no true relationship to detect" in error
        assert abs(power - 0.05) < 0.01  # Power should equal alpha
    
    def test_calculate_power_perfect_correlation(self):
        """Test power calculation with perfect correlation."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            effect_size=1.0,
            alpha=0.05
        )
        assert power is not None
        assert error is not None
        assert "essentially 1.0" in error
        assert power > 0.99  # Power should be essentially 1
    
    def test_calculate_power_invalid_effect_size(self):
        """Test power calculation fails with invalid effect size."""
        power, error = self.corr_test.calculate_power(
            x_values=self.height,
            y_values=self.weight,
            effect_size=1.5  # Invalid effect size > 1
        )
        assert power is None
        assert error is not None
        assert "between -1 and 1" in error
    
    def test_calculate_power_insufficient_sample_size(self):
        """Test power calculation fails with insufficient sample size."""
        power, error = self.corr_test.calculate_power(
            x_values=[1, 2],
            y_values=[3, 4],
            n=2
        )
        assert power is None
        assert error is not None
        assert "at least 3" in error
    
    def test_calculate_power_invalid_params(self):
        """Test power calculation with invalid parameters."""
        power, error = self.corr_test.calculate_power(
            x_values=self.constant_x,
            y_values=self.constant_y
        )
        assert power is None
        assert error is not None
    
    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        x_array = np.array(self.height)
        y_array = np.array(self.weight)
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=x_array,
            y_values=y_array
        )
        assert p_value is not None
        assert error is None
    
    def test_tuple_input(self):
        """Test that tuples work as input."""
        x_tuple = tuple(self.height)
        y_tuple = tuple(self.weight)
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=x_tuple,
            y_values=y_tuple
        )
        assert p_value is not None
        assert error is None


class TestCorrelationFactoryIntegration:
    """Test CorrelationTest integration with factory pattern."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
        
    def test_factory_registration(self):
        """Test that correlation test is properly registered in factory."""
        available_tests = self.factory.get_available_tests()
        
        # Check main registration and aliases
        assert "correlation" in available_tests
        assert "pearson" in available_tests
        assert "spearman" in available_tests
        assert "relationship" in available_tests
        assert "correlation_test" in available_tests
        assert "pearson_correlation" in available_tests
        assert "spearman_correlation" in available_tests
    
    def test_factory_get_test_correlation(self):
        """Test getting correlation test from factory."""
        test = self.factory.get_test("correlation")
        assert isinstance(test, CorrelationTest)
    
    def test_factory_get_test_aliases(self):
        """Test getting correlation test using aliases."""
        test_pearson = self.factory.get_test("pearson")
        test_spearman = self.factory.get_test("spearman")
        test_relationship = self.factory.get_test("relationship")
        
        assert isinstance(test_pearson, CorrelationTest)
        assert isinstance(test_spearman, CorrelationTest)
        assert isinstance(test_relationship, CorrelationTest)
    
    def test_factory_test_availability(self):
        """Test that factory correctly reports correlation test availability."""
        assert self.factory.is_test_available("correlation")
        assert self.factory.is_test_available("pearson")
        assert self.factory.is_test_available("spearman")
        assert self.factory.is_test_available("relationship")
        assert not self.factory.is_test_available("nonexistent_correlation")


class TestCorrelationRealWorldExamples:
    """Test correlation with real-world examples and use cases."""
    
    def setup_method(self):
        """Set up test fixtures with realistic data."""
        self.corr_test = CorrelationTest()
        
        # Clinical trial example: Drug dose vs efficacy score
        self.drug_dose = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.efficacy_score = [2.1, 3.8, 5.2, 6.9, 8.1, 9.3, 10.5, 11.8, 12.9, 14.2]
        
        # Healthcare example: Age vs blood pressure (non-linear, better for Spearman)
        self.age_bp = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        self.blood_pressure = [115, 118, 122, 128, 135, 142, 150, 158, 167, 175, 185, 195]
        
        # Psychology example: Stress level vs performance (negative correlation)
        self.stress_level = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.performance = [95, 92, 88, 85, 80, 75, 68, 60, 52, 45]
        
    def test_drug_dose_efficacy_pearson(self):
        """Test Pearson correlation for drug dose vs efficacy (linear relationship)."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.drug_dose,
            y_values=self.efficacy_score,
            correlation_type="pearson"
        )
        
        assert p_value is not None
        assert error is None
        assert p_value < 0.001  # Should be highly significant
        
        # Calculate power
        power, power_error = self.corr_test.calculate_power(
            x_values=self.drug_dose,
            y_values=self.efficacy_score,
            correlation_type="pearson"
        )
        
        assert power is not None
        assert power_error is None
        assert power > 0.9  # Should have very high power
    
    def test_age_blood_pressure_spearman(self):
        """Test Spearman correlation for age vs blood pressure (monotonic relationship)."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.age_bp,
            y_values=self.blood_pressure,
            correlation_type="spearman"
        )
        
        assert p_value is not None
        # Note: error might contain perfect correlation warning, which is acceptable
        assert p_value < 0.01  # Should be significant
        
        # Compare with Pearson to see if Spearman is more appropriate
        p_value_pearson, _ = self.corr_test.calculate_p_value(
            x_values=self.age_bp,
            y_values=self.blood_pressure,
            correlation_type="pearson"
        )
        
        # Both should be significant, but this shows the method works for both
        assert p_value_pearson < 0.01
    
    def test_stress_performance_negative_correlation(self):
        """Test negative correlation between stress and performance."""
        p_value, error = self.corr_test.calculate_p_value(
            x_values=self.stress_level,
            y_values=self.performance,
            correlation_type="pearson"
        )
        
        assert p_value is not None
        assert error is None
        assert p_value < 0.001  # Strong negative correlation should be significant
        
        # Verify the correlation is indeed negative by checking with scipy
        r, _ = stats.pearsonr(self.stress_level, self.performance)
        assert r < -0.8  # Should be strong negative correlation
    
    def test_small_sample_size_warning(self):
        """Test behavior with small but valid sample sizes."""
        small_x = [1, 2, 3, 4, 5]
        small_y = [2, 4, 6, 8, 10]
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=small_x,
            y_values=small_y
        )
        
        assert p_value is not None
        # With perfect correlation, should get warning
        assert error is not None
        assert "Perfect correlation detected" in error
        
        # Power should be very high despite small sample
        power, power_error = self.corr_test.calculate_power(
            x_values=small_x,
            y_values=small_y
        )
        assert power > 0.9


class TestCorrelationEdgeCases:
    """Test edge cases and error handling for CorrelationTest."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.corr_test = CorrelationTest()
    
    def test_identical_arrays(self):
        """Test with identical x and y arrays (perfect correlation)."""
        identical_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=identical_data,
            y_values=identical_data
        )
        
        assert p_value is not None
        # May or may not get perfect correlation warning depending on floating point precision
        if error is not None:
            assert "Perfect correlation detected" in error
        assert abs(p_value) < 1e-5  # Should be very close to 0
    
    def test_large_dataset(self):
        """Test with large dataset (performance and numerical stability)."""
        np.random.seed(123)
        n = 1000
        x_large = np.random.normal(50, 10, n)
        y_large = 2 * x_large + np.random.normal(0, 5, n)  # Strong correlation with noise
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=x_large,
            y_values=y_large
        )
        
        assert p_value is not None
        assert error is None
        assert p_value < 0.001  # Should be highly significant with large sample
        
        # Power should be very high
        power, power_error = self.corr_test.calculate_power(
            x_values=x_large,
            y_values=y_large
        )
        assert power > 0.99
    
    def test_very_small_correlation(self):
        """Test with very small correlation (near independence)."""
        np.random.seed(456)
        x_small = np.random.normal(0, 1, 100)
        y_small = 0.01 * x_small + np.random.normal(0, 1, 100)  # Very weak correlation
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=x_small,
            y_values=y_small
        )
        
        assert p_value is not None
        assert error is None
        assert p_value > 0.1  # Should not be significant
        
        # Power should be low
        power, power_error = self.corr_test.calculate_power(
            x_values=x_small,
            y_values=y_small
        )
        assert power < 0.3
    
    def test_outlier_resistance_spearman_vs_pearson(self):
        """Test Spearman's resistance to outliers compared to Pearson."""
        # Data with one extreme outlier
        x_outlier = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        y_outlier = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # Linear trend disrupted
        
        p_value_pearson, _ = self.corr_test.calculate_p_value(
            x_values=x_outlier,
            y_values=y_outlier,
            correlation_type="pearson"
        )
        
        p_value_spearman, _ = self.corr_test.calculate_p_value(
            x_values=x_outlier,
            y_values=y_outlier,
            correlation_type="spearman"
        )
        
        # Both should work, but Spearman should be more robust
        assert p_value_pearson is not None
        assert p_value_spearman is not None
        # Spearman should detect the monotonic relationship better
        assert p_value_spearman < p_value_pearson
    
    def test_tied_ranks_spearman(self):
        """Test Spearman correlation with tied ranks."""
        x_tied = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7]  # Multiple ties
        y_tied = [2, 3, 3, 5, 6, 6, 6, 8, 9, 10]  # Corresponding ties
        
        p_value, error = self.corr_test.calculate_p_value(
            x_values=x_tied,
            y_values=y_tied,
            correlation_type="spearman"
        )
        
        assert p_value is not None
        # May get perfect correlation warning if correlation is exactly 1.0
        if error is not None:
            assert "Perfect correlation detected" in error
        assert 0 <= p_value <= 1
    
    def test_numerical_precision_edge_cases(self):
        """Test numerical precision with very small or large numbers."""
        # Very small numbers
        x_small = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        y_small = [2e-10, 4e-10, 6e-10, 8e-10, 10e-10]
        
        p_value_small, error_small = self.corr_test.calculate_p_value(
            x_values=x_small,
            y_values=y_small
        )
        
        assert p_value_small is not None
        # May detect perfect correlation depending on numerical precision
        if error_small is not None:
            assert "Perfect correlation detected" in error_small
        
        # Very large numbers
        x_large = [1e10, 2e10, 3e10, 4e10, 5e10]
        y_large = [2e10, 4e10, 6e10, 8e10, 10e10]
        
        p_value_large, error_large = self.corr_test.calculate_p_value(
            x_values=x_large,
            y_values=y_large
        )
        
        assert p_value_large is not None
        # May detect perfect correlation depending on numerical precision
        if error_large is not None:
            assert "Perfect correlation detected" in error_large


class TestCorrelationBackwardsCompatibility:
    """Test that adding CorrelationTest doesn't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
    
    def test_existing_tests_still_available(self):
        """Test that existing tests are still available after adding correlation."""
        available_tests = self.factory.get_available_tests()
        
        # Check that original test registrations are still there
        assert "two_sample_t_test" in available_tests
        assert "t_test" in available_tests
        assert "chi_square" in available_tests
        assert "one_way_anova" in available_tests
        assert "anova" in available_tests
    
    def test_other_tests_still_work(self):
        """Test that other test functionality is unchanged."""
        from statistical_tests import TwoSampleTTest, ChiSquareTest, OneWayANOVA
        
        # Test t-test
        t_test = self.factory.get_test("two_sample_t_test")
        assert isinstance(t_test, TwoSampleTTest)
        
        p_value, error = t_test.calculate_p_value(N_total=100, cohens_d=0.5)
        assert p_value is not None
        assert error is None
        
        # Test chi-square
        chi_test = self.factory.get_test("chi_square")
        assert isinstance(chi_test, ChiSquareTest)
        
        p_value, error = chi_test.calculate_p_value(contingency_table=[[10, 20], [30, 40]])
        assert p_value is not None
        assert error is None
        
        # Test ANOVA with larger groups to avoid small sample warning
        anova_test = self.factory.get_test("one_way_anova")
        assert isinstance(anova_test, OneWayANOVA)
        
        groups = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]
        p_value, error = anova_test.calculate_p_value(groups=groups)
        assert p_value is not None
        # May have warning about group sizes, but should work
    
    def test_factory_robust_to_new_additions(self):
        """Test that factory pattern is robust to new test additions."""
        # Test that we can still get unknown test error
        with pytest.raises(ValueError, match="Unknown test type"):
            self.factory.get_test("nonexistent_test")
        
        # Test that factory reports correct number of tests
        available_tests = self.factory.get_available_tests()
        # Should have all test registrations including new correlation tests
        assert len(available_tests) >= 18  # t-test(4) + chi-square(6) + anova(6) + correlation(7) = 23


if __name__ == "__main__":
    pytest.main([__file__])