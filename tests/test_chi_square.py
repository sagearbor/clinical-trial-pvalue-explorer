"""
Comprehensive tests for ChiSquareTest implementation.

Tests cover:
- Basic functionality (p-value and power calculations)
- Parameter validation
- Independence tests vs goodness-of-fit tests
- Edge cases and error handling
- Integration with factory pattern
- Backwards compatibility
"""

import pytest
import numpy as np
from statistical_tests import ChiSquareTest, get_factory
from scipy import stats


class TestChiSquareTest:
    """Test ChiSquareTest implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chi_test = ChiSquareTest()
        
        # Sample contingency tables for testing
        self.simple_2x2 = [[10, 20], [30, 40]]
        self.medium_3x3 = [[15, 25, 10], [20, 30, 15], [10, 15, 20]]
        self.large_effect = [[50, 10], [10, 50]]  # Strong association
        self.small_effect = [[25, 25], [25, 25]]  # No association
        
        # Expected frequencies for goodness-of-fit tests
        self.expected_2x2 = [[15, 25], [25, 35]]
        
    def test_required_params(self):
        """Test required parameters are correctly defined."""
        required = self.chi_test.get_required_params()
        assert "contingency_table" in required
        assert len(required) == 1
    
    def test_validate_params_valid_2x2(self):
        """Test parameter validation with valid 2x2 table."""
        is_valid, error = self.chi_test.validate_params(contingency_table=self.simple_2x2)
        assert is_valid
        assert error is None
    
    def test_validate_params_valid_3x3(self):
        """Test parameter validation with valid 3x3 table."""
        is_valid, error = self.chi_test.validate_params(contingency_table=self.medium_3x3)
        assert is_valid
        assert error is None
    
    def test_validate_params_with_expected_frequencies(self):
        """Test parameter validation with expected frequencies."""
        is_valid, error = self.chi_test.validate_params(
            contingency_table=self.simple_2x2,
            expected_frequencies=self.expected_2x2
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_missing_table(self):
        """Test parameter validation fails when contingency_table is missing."""
        is_valid, error = self.chi_test.validate_params()
        assert not is_valid
        assert "Missing required parameter: contingency_table" in error
    
    def test_validate_params_invalid_table_type(self):
        """Test parameter validation fails with invalid table type."""
        is_valid, error = self.chi_test.validate_params(contingency_table="invalid")
        assert not is_valid
        assert "must be a 2D array" in error
    
    def test_validate_params_1d_table(self):
        """Test parameter validation fails with 1D table."""
        is_valid, error = self.chi_test.validate_params(contingency_table=[1, 2, 3])
        assert not is_valid
        assert "must be 2-dimensional" in error
    
    def test_validate_params_negative_frequencies(self):
        """Test parameter validation fails with negative frequencies."""
        is_valid, error = self.chi_test.validate_params(contingency_table=[[10, -5], [20, 30]])
        assert not is_valid
        assert "must be non-negative" in error
    
    def test_validate_params_zero_total(self):
        """Test parameter validation fails with all zero frequencies."""
        is_valid, error = self.chi_test.validate_params(contingency_table=[[0, 0], [0, 0]])
        assert not is_valid
        assert "Total frequency count must be greater than 0" in error
    
    def test_validate_params_1x1_table(self):
        """Test parameter validation fails with 1x1 table."""
        is_valid, error = self.chi_test.validate_params(contingency_table=[[10]])
        assert not is_valid
        assert "must have at least 2 rows or 2 columns" in error
    
    def test_validate_params_mismatched_expected_shape(self):
        """Test parameter validation fails when expected frequencies shape doesn't match."""
        is_valid, error = self.chi_test.validate_params(
            contingency_table=self.simple_2x2,
            expected_frequencies=[[10, 20, 30], [40, 50, 60]]  # Wrong shape
        )
        assert not is_valid
        assert "must have same shape" in error
    
    def test_validate_params_zero_expected_frequencies(self):
        """Test parameter validation fails with zero expected frequencies."""
        is_valid, error = self.chi_test.validate_params(
            contingency_table=self.simple_2x2,
            expected_frequencies=[[0, 20], [30, 40]]
        )
        assert not is_valid
        assert "must be positive" in error
    
    def test_validate_params_invalid_alpha(self):
        """Test parameter validation fails with invalid alpha."""
        is_valid, error = self.chi_test.validate_params(
            contingency_table=self.simple_2x2,
            alpha=1.5  # Invalid alpha > 1
        )
        assert not is_valid
        assert "between 0 and 1" in error
    
    def test_calculate_p_value_independence_test(self):
        """Test p-value calculation for independence test (no expected frequencies)."""
        p_value, error = self.chi_test.calculate_p_value(contingency_table=self.simple_2x2)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        
        # Verify against scipy calculation
        chi2_stat, expected_p, dof, expected = stats.chi2_contingency(self.simple_2x2)
        assert abs(p_value - expected_p) < 1e-10
    
    def test_calculate_p_value_goodness_of_fit(self):
        """Test p-value calculation for goodness-of-fit test (with expected frequencies)."""
        p_value, error = self.chi_test.calculate_p_value(
            contingency_table=self.simple_2x2,
            expected_frequencies=self.expected_2x2
        )
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
    
    def test_calculate_p_value_strong_association(self):
        """Test p-value calculation with strong association (should be low p-value)."""
        p_value, error = self.chi_test.calculate_p_value(contingency_table=self.large_effect)
        assert p_value is not None
        assert error is None
        assert p_value < 0.01  # Strong effect should have low p-value
    
    def test_calculate_p_value_no_association(self):
        """Test p-value calculation with no association (should be high p-value)."""
        p_value, error = self.chi_test.calculate_p_value(contingency_table=self.small_effect)
        assert p_value is not None
        assert error is None
        assert p_value > 0.5  # No effect should have high p-value
    
    def test_calculate_p_value_low_expected_frequencies_warning(self):
        """Test that warning is returned when expected frequencies are low."""
        # Create a table with low expected frequencies
        low_freq_table = [[2, 1], [1, 2]]
        p_value, error = self.chi_test.calculate_p_value(contingency_table=low_freq_table)
        assert p_value is not None
        assert error is not None
        assert "Warning" in error
        assert "less than 5" in error
    
    def test_calculate_p_value_invalid_params(self):
        """Test p-value calculation with invalid parameters."""
        p_value, error = self.chi_test.calculate_p_value(contingency_table="invalid")
        assert p_value is None
        assert error is not None
    
    def test_calculate_power_with_effect_size(self):
        """Test power calculation with provided effect size."""
        power, error = self.chi_test.calculate_power(
            contingency_table=self.simple_2x2,
            effect_size=0.3,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_calculated_effect_size(self):
        """Test power calculation with calculated effect size (Cramér's V)."""
        power, error = self.chi_test.calculate_power(
            contingency_table=self.simple_2x2,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_default_alpha(self):
        """Test power calculation uses default alpha=0.05 when not provided."""
        power, error = self.chi_test.calculate_power(contingency_table=self.simple_2x2)
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_with_total_n(self):
        """Test power calculation with specified total sample size."""
        power, error = self.chi_test.calculate_power(
            contingency_table=self.simple_2x2,
            total_n=200,
            effect_size=0.3,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_invalid_effect_size(self):
        """Test power calculation fails with invalid effect size."""
        power, error = self.chi_test.calculate_power(
            contingency_table=self.simple_2x2,
            effect_size=-0.1  # Negative effect size
        )
        assert power is None
        assert error is not None
        assert "non-negative" in error
    
    def test_calculate_power_invalid_params(self):
        """Test power calculation with invalid parameters."""
        power, error = self.chi_test.calculate_power(contingency_table="invalid")
        assert power is None
        assert error is not None
    
    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        table = np.array(self.simple_2x2)
        p_value, error = self.chi_test.calculate_p_value(contingency_table=table)
        assert p_value is not None
        assert error is None
    
    def test_list_of_lists_input(self):
        """Test that list of lists work as input."""
        p_value, error = self.chi_test.calculate_p_value(contingency_table=self.simple_2x2)
        assert p_value is not None
        assert error is None
    
    def test_tuple_input(self):
        """Test that tuples work as input."""
        table_tuple = ((10, 20), (30, 40))
        p_value, error = self.chi_test.calculate_p_value(contingency_table=table_tuple)
        assert p_value is not None
        assert error is None


class TestChiSquareFactoryIntegration:
    """Test ChiSquareTest integration with factory pattern."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
        
    def test_factory_registration(self):
        """Test that chi-square test is properly registered in factory."""
        available_tests = self.factory.get_available_tests()
        
        # Check main registration
        assert "chi_square" in available_tests
        
        # Check aliases
        assert "chi2" in available_tests
        assert "categorical_test" in available_tests
        assert "chi_square_test" in available_tests
        assert "independence_test" in available_tests
        assert "goodness_of_fit" in available_tests
    
    def test_factory_get_test_chi_square(self):
        """Test getting chi-square test from factory."""
        test = self.factory.get_test("chi_square")
        assert isinstance(test, ChiSquareTest)
    
    def test_factory_get_test_aliases(self):
        """Test getting chi-square test using aliases."""
        test_chi2 = self.factory.get_test("chi2")
        test_categorical = self.factory.get_test("categorical_test")
        test_independence = self.factory.get_test("independence_test")
        
        assert isinstance(test_chi2, ChiSquareTest)
        assert isinstance(test_categorical, ChiSquareTest)
        assert isinstance(test_independence, ChiSquareTest)
    
    def test_factory_test_availability(self):
        """Test that factory correctly reports chi-square test availability."""
        assert self.factory.is_test_available("chi_square")
        assert self.factory.is_test_available("chi2")
        assert self.factory.is_test_available("categorical_test")
        assert not self.factory.is_test_available("nonexistent_test")


class TestChiSquareEdgeCases:
    """Test edge cases and error handling for ChiSquareTest."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chi_test = ChiSquareTest()
    
    def test_single_row_table_valid(self):
        """Test that single row table with multiple columns is valid (goodness-of-fit)."""
        single_row = [[10, 20, 30]]
        is_valid, error = self.chi_test.validate_params(contingency_table=single_row)
        assert is_valid
        assert error is None
    
    def test_single_column_table_valid(self):
        """Test that single column table with multiple rows is valid (goodness-of-fit)."""
        single_col = [[10], [20], [30]]
        is_valid, error = self.chi_test.validate_params(contingency_table=single_col)
        assert is_valid
        assert error is None
    
    def test_single_row_single_column_invalid(self):
        """Test that 1x1 table is invalid."""
        single_cell = [[10]]
        is_valid, error = self.chi_test.validate_params(contingency_table=single_cell)
        assert not is_valid
        assert "at least 2 rows or 2 columns" in error
    
    def test_empty_table(self):
        """Test that empty table is rejected."""
        empty_table = []
        is_valid, error = self.chi_test.validate_params(contingency_table=empty_table)
        assert not is_valid
        # Empty list creates 1D array, so expect 2D error
        assert "2-dimensional" in error
    
    def test_non_numeric_values(self):
        """Test that non-numeric values are rejected."""
        non_numeric = [["a", "b"], ["c", "d"]]
        is_valid, error = self.chi_test.validate_params(contingency_table=non_numeric)
        assert not is_valid
        assert "numeric" in error
    
    def test_infinite_values(self):
        """Test that infinite values are rejected."""
        infinite_table = [[10, float('inf')], [20, 30]]
        is_valid, error = self.chi_test.validate_params(contingency_table=infinite_table)
        assert not is_valid
        assert "finite" in error
    
    def test_nan_values(self):
        """Test that NaN values are rejected."""
        nan_table = [[10, float('nan')], [20, 30]]
        is_valid, error = self.chi_test.validate_params(contingency_table=nan_table)
        assert not is_valid
        assert "finite" in error
    
    def test_very_large_table(self):
        """Test with a larger contingency table."""
        # Create a 5x4 table
        large_table = [
            [10, 15, 20, 25],
            [12, 18, 22, 28],
            [8, 12, 16, 20],
            [14, 21, 28, 35],
            [11, 16, 21, 26]
        ]
        
        p_value, error = self.chi_test.calculate_p_value(contingency_table=large_table)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
    
    def test_power_calculation_edge_cases(self):
        """Test power calculation edge cases."""
        # Test with very small effect size
        power_small, error = self.chi_test.calculate_power(
            contingency_table=[[50, 50], [50, 50]],
            effect_size=0.01
        )
        assert power_small is not None
        assert error is None
        assert power_small < 0.2  # Should be low power
        
        # Test with very large effect size
        power_large, error = self.chi_test.calculate_power(
            contingency_table=[[100, 10], [10, 100]],
            effect_size=0.9
        )
        assert power_large is not None
        assert error is None
        assert power_large > 0.8  # Should be high power


class TestChiSquareBackwardsCompatibility:
    """Test that adding ChiSquareTest doesn't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
    
    def test_existing_tests_still_available(self):
        """Test that existing tests are still available after adding chi-square."""
        available_tests = self.factory.get_available_tests()
        
        # Check that original t-test registrations are still there
        assert "two_sample_t_test" in available_tests
        assert "t_test" in available_tests
        assert "two_sample_ttest" in available_tests
        assert "independent_samples_t_test" in available_tests
    
    def test_t_test_still_works(self):
        """Test that t-test functionality is unchanged."""
        from statistical_tests import TwoSampleTTest
        
        t_test = self.factory.get_test("two_sample_t_test")
        assert isinstance(t_test, TwoSampleTTest)
        
        # Test basic t-test calculation
        p_value, error = t_test.calculate_p_value(N_total=100, cohens_d=0.5)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
    
    def test_factory_robust_to_new_additions(self):
        """Test that factory pattern is robust to new test additions."""
        # Test that we can still get unknown test error
        with pytest.raises(ValueError, match="Unknown test type"):
            self.factory.get_test("nonexistent_test")
        
        # Test that factory reports correct number of tests
        available_tests = self.factory.get_available_tests()
        # Should have original 4 t-test registrations + 6 chi-square registrations = 10
        assert len(available_tests) >= 10


if __name__ == "__main__":
    pytest.main([__file__])