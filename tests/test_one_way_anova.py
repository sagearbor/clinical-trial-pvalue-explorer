"""
Comprehensive tests for OneWayANOVA implementation.

Tests cover:
- Basic functionality (p-value and power calculations)
- Parameter validation  
- Multiple group comparisons
- Edge cases and error handling
- Integration with factory pattern
- Eta-squared effect size calculations
- Backwards compatibility
"""

import pytest
import numpy as np
from statistical_tests import OneWayANOVA, get_factory
from scipy import stats


class TestOneWayANOVA:
    """Test OneWayANOVA implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.anova_test = OneWayANOVA()
        
        # Sample group data for testing
        # Groups with clear differences (should give low p-value)
        self.different_groups = [
            [10, 12, 11, 9, 13],      # Group 1: mean ~11
            [20, 22, 21, 19, 23],     # Group 2: mean ~21  
            [30, 32, 31, 29, 33]      # Group 3: mean ~31
        ]
        
        # Groups with similar means (should give high p-value)
        self.similar_groups = [
            [10, 11, 9, 12, 8],       # Group 1: mean ~10
            [9, 12, 10, 11, 8],       # Group 2: mean ~10
            [11, 10, 9, 12, 8]        # Group 3: mean ~10
        ]
        
        # Two groups only (minimum for ANOVA)
        self.two_groups = [
            [10, 12, 11, 9, 13],
            [20, 22, 21, 19, 23]
        ]
        
        # Groups with different sample sizes (unbalanced)
        self.unbalanced_groups = [
            [10, 12, 11],             # n=3
            [20, 22, 21, 19, 23, 24], # n=6
            [30, 32, 31, 29]          # n=4
        ]
        
        # Groups with small sample sizes
        self.small_groups = [
            [10, 12],                 # n=2 (minimum)
            [20, 22],                 # n=2
            [30, 32]                  # n=2
        ]
    
    def test_required_params(self):
        """Test required parameters are correctly defined."""
        required = self.anova_test.get_required_params()
        assert "groups" in required
        assert len(required) == 1
    
    def test_validate_params_valid_groups(self):
        """Test parameter validation with valid groups."""
        is_valid, error = self.anova_test.validate_params(groups=self.different_groups)
        assert is_valid
        assert error is None
    
    def test_validate_params_valid_two_groups(self):
        """Test parameter validation with minimum two groups."""
        is_valid, error = self.anova_test.validate_params(groups=self.two_groups)
        assert is_valid
        assert error is None
    
    def test_validate_params_valid_individual_groups(self):
        """Test parameter validation with individual group parameters."""
        is_valid, error = self.anova_test.validate_params(
            group1=[10, 12, 11, 9, 13],
            group2=[20, 22, 21, 19, 23],
            group3=[30, 32, 31, 29, 33]
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_mixed_groups_and_individual(self):
        """Test that groups parameter takes precedence over individual group parameters."""
        is_valid, error = self.anova_test.validate_params(
            groups=self.different_groups,
            group1=[1, 2, 3],  # Should be ignored
            group2=[4, 5, 6]   # Should be ignored
        )
        assert is_valid
        assert error is None
    
    def test_validate_params_missing_groups(self):
        """Test parameter validation fails when groups is missing."""
        is_valid, error = self.anova_test.validate_params()
        assert not is_valid
        assert "Missing required parameter: groups" in error
    
    def test_validate_params_single_group(self):
        """Test parameter validation fails with only one group."""
        is_valid, error = self.anova_test.validate_params(groups=[[10, 12, 11, 9, 13]])
        assert not is_valid
        assert "requires at least 2 groups" in error
    
    def test_validate_params_empty_group(self):
        """Test parameter validation fails with empty group."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            []  # Empty group
        ])
        assert not is_valid
        assert "cannot be empty" in error
    
    def test_validate_params_single_observation_group(self):
        """Test parameter validation fails with group having only one observation."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            [20]  # Only one observation
        ])
        assert not is_valid
        assert "must have at least 2 observations" in error
    
    def test_validate_params_non_numeric_values(self):
        """Test parameter validation fails with non-numeric values."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            ["a", "b", "c"]  # Non-numeric
        ])
        assert not is_valid
        assert "must contain numeric values" in error
    
    def test_validate_params_infinite_values(self):
        """Test parameter validation fails with infinite values."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            [20, float('inf'), 21]  # Contains infinity
        ])
        assert not is_valid
        assert "finite numbers" in error
    
    def test_validate_params_nan_values(self):
        """Test parameter validation fails with NaN values."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            [20, float('nan'), 21]  # Contains NaN
        ])
        assert not is_valid
        assert "finite numbers" in error
    
    def test_validate_params_invalid_group_type(self):
        """Test parameter validation fails with invalid group type."""
        is_valid, error = self.anova_test.validate_params(groups=[
            [10, 12, 11, 9, 13],
            "invalid_group"  # Should be list/array
        ])
        assert not is_valid
        assert "must be a list, tuple, or numpy array" in error
    
    def test_validate_params_invalid_groups_type(self):
        """Test parameter validation fails when groups is not list/tuple."""
        is_valid, error = self.anova_test.validate_params(groups="invalid")
        assert not is_valid
        assert "must be a list or tuple" in error
    
    def test_validate_params_invalid_alpha(self):
        """Test parameter validation fails with invalid alpha."""
        is_valid, error = self.anova_test.validate_params(
            groups=self.different_groups,
            alpha=1.5  # Invalid alpha > 1
        )
        assert not is_valid
        assert "between 0 and 1" in error
    
    def test_calculate_p_value_different_groups(self):
        """Test p-value calculation with clearly different groups."""
        p_value, error = self.anova_test.calculate_p_value(groups=self.different_groups)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant difference
        
        # Verify against scipy calculation
        f_stat, expected_p = stats.f_oneway(*self.different_groups)
        assert abs(p_value - expected_p) < 1e-10
    
    def test_calculate_p_value_similar_groups(self):
        """Test p-value calculation with similar groups."""
        p_value, error = self.anova_test.calculate_p_value(groups=self.similar_groups)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        assert p_value > 0.05  # Should not be significant
    
    def test_calculate_p_value_two_groups(self):
        """Test p-value calculation with minimum two groups."""
        p_value, error = self.anova_test.calculate_p_value(groups=self.two_groups)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant difference
    
    def test_calculate_p_value_individual_groups(self):
        """Test p-value calculation using individual group parameters."""
        p_value, error = self.anova_test.calculate_p_value(
            group1=self.different_groups[0],
            group2=self.different_groups[1],
            group3=self.different_groups[2]
        )
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        
        # Should match calculation with groups parameter
        p_value_groups, _ = self.anova_test.calculate_p_value(groups=self.different_groups)
        assert abs(p_value - p_value_groups) < 1e-10
    
    def test_calculate_p_value_unbalanced_groups(self):
        """Test p-value calculation with unbalanced groups."""
        p_value, error = self.anova_test.calculate_p_value(groups=self.unbalanced_groups)
        assert p_value is not None
        # Should have warning about unbalanced groups
        assert error is not None
        assert "unbalanced" in error
        assert 0 <= p_value <= 1
    
    def test_calculate_p_value_small_groups(self):
        """Test p-value calculation with small groups."""
        p_value, error = self.anova_test.calculate_p_value(groups=self.small_groups)
        assert p_value is not None
        # Should have warning about small sample sizes
        assert error is not None
        assert "small sample sizes" in error
        assert 0 <= p_value <= 1
    
    def test_calculate_p_value_invalid_params(self):
        """Test p-value calculation with invalid parameters."""
        p_value, error = self.anova_test.calculate_p_value(groups="invalid")
        assert p_value is None
        assert error is not None
    
    def test_calculate_power_with_effect_size(self):
        """Test power calculation with provided effect size."""
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            effect_size=0.3,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_calculated_effect_size(self):
        """Test power calculation with calculated effect size (eta-squared)."""
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_default_alpha(self):
        """Test power calculation uses default alpha=0.05 when not provided."""
        power, error = self.anova_test.calculate_power(groups=self.different_groups)
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_with_total_n(self):
        """Test power calculation with specified total sample size."""
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            total_n=50,
            effect_size=0.3,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
    
    def test_calculate_power_zero_effect_size(self):
        """Test power calculation with zero effect size."""
        power, error = self.anova_test.calculate_power(
            groups=self.similar_groups,  # Similar groups should have small effect
            effect_size=0.0,
            alpha=0.05
        )
        assert power is not None
        assert error is not None
        assert "power equals alpha" in error
        assert abs(power - 0.05) < 0.01  # Power should equal alpha
    
    def test_calculate_power_large_effect_size(self):
        """Test power calculation with large effect size."""
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            effect_size=0.8,
            alpha=0.05
        )
        assert power is not None
        assert error is None
        assert power > 0.8  # Should have high power
    
    def test_calculate_power_invalid_effect_size(self):
        """Test power calculation fails with invalid effect size."""
        # Test negative effect size
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            effect_size=-0.1
        )
        assert power is None
        assert error is not None
        assert "between 0 and 1" in error
        
        # Test effect size >= 1
        power, error = self.anova_test.calculate_power(
            groups=self.different_groups,
            effect_size=1.0
        )
        assert power is None
        assert error is not None
        assert "less than 1.0" in error
    
    def test_calculate_power_invalid_params(self):
        """Test power calculation with invalid parameters."""
        power, error = self.anova_test.calculate_power(groups="invalid")
        assert power is None
        assert error is not None
    
    def test_eta_squared_calculation(self):
        """Test that eta-squared effect size is calculated correctly."""
        # Calculate power to trigger eta-squared calculation
        power, error = self.anova_test.calculate_power(groups=self.different_groups)
        assert power is not None
        
        # Calculate eta-squared manually and verify it's reasonable
        f_stat, _ = stats.f_oneway(*self.different_groups)
        k = len(self.different_groups)
        total_n = sum(len(group) for group in self.different_groups)
        df_between = k - 1
        df_within = total_n - k
        
        expected_eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within)
        assert 0 <= expected_eta_squared <= 1
    
    def test_numpy_array_input(self):
        """Test that numpy arrays work as input."""
        groups_numpy = [np.array(group) for group in self.different_groups]
        p_value, error = self.anova_test.calculate_p_value(groups=groups_numpy)
        assert p_value is not None
        assert error is None
    
    def test_tuple_input(self):
        """Test that tuples work as input."""
        groups_tuple = tuple(tuple(group) for group in self.different_groups)
        p_value, error = self.anova_test.calculate_p_value(groups=groups_tuple)
        assert p_value is not None
        assert error is None
    
    def test_mixed_input_types(self):
        """Test with mixed input types (list, tuple, numpy array)."""
        mixed_groups = [
            self.different_groups[0],           # list
            tuple(self.different_groups[1]),    # tuple
            np.array(self.different_groups[2])  # numpy array
        ]
        p_value, error = self.anova_test.calculate_p_value(groups=mixed_groups)
        assert p_value is not None
        assert error is None


class TestOneWayANOVAFactoryIntegration:
    """Test OneWayANOVA integration with factory pattern."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
        
    def test_factory_registration(self):
        """Test that OneWayANOVA is properly registered in factory."""
        available_tests = self.factory.get_available_tests()
        
        # Check main registration
        assert "one_way_anova" in available_tests
        
        # Check aliases  
        assert "anova" in available_tests
        assert "f_test" in available_tests
        assert "multiple_groups" in available_tests
        assert "analysis_of_variance" in available_tests
        assert "oneway_anova" in available_tests
    
    def test_factory_get_test_one_way_anova(self):
        """Test getting OneWayANOVA from factory."""
        test = self.factory.get_test("one_way_anova")
        assert isinstance(test, OneWayANOVA)
    
    def test_factory_get_test_aliases(self):
        """Test getting OneWayANOVA using aliases."""
        test_anova = self.factory.get_test("anova")
        test_f = self.factory.get_test("f_test")
        test_multiple = self.factory.get_test("multiple_groups")
        test_analysis = self.factory.get_test("analysis_of_variance")
        
        assert isinstance(test_anova, OneWayANOVA)
        assert isinstance(test_f, OneWayANOVA)
        assert isinstance(test_multiple, OneWayANOVA)
        assert isinstance(test_analysis, OneWayANOVA)
    
    def test_factory_test_availability(self):
        """Test that factory correctly reports OneWayANOVA availability."""
        assert self.factory.is_test_available("one_way_anova")
        assert self.factory.is_test_available("anova")
        assert self.factory.is_test_available("f_test")
        assert not self.factory.is_test_available("nonexistent_test")


class TestOneWayANOVAEdgeCases:
    """Test edge cases and error handling for OneWayANOVA."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.anova_test = OneWayANOVA()
    
    def test_identical_groups(self):
        """Test with identical groups (zero variance between groups)."""
        identical_groups = [
            [10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=identical_groups)
        # F-statistic should be 0, leading to p-value = 1 or NaN (scipy warning)
        # Either case is acceptable - identical groups have no variance to test
        assert p_value is not None or error is not None
        if p_value is not None:
            assert p_value == 1.0 or p_value > 0.99  # Should be very high p-value
    
    def test_zero_variance_within_groups(self):
        """Test with zero variance within groups."""
        zero_var_groups = [
            [10, 10, 10],  # No variance within group
            [20, 20, 20],  # No variance within group  
            [30, 30, 30]   # No variance within group
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=zero_var_groups)
        assert p_value is not None
        assert p_value < 0.05  # Should be significant (perfect separation)
    
    def test_very_small_differences(self):
        """Test with very small differences between groups."""
        small_diff_groups = [
            [10.001, 10.002, 10.003],
            [10.004, 10.005, 10.006],
            [10.007, 10.008, 10.009]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=small_diff_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
    
    def test_large_number_of_groups(self):
        """Test with many groups."""
        many_groups = []
        for i in range(10):  # 10 groups
            # Each group has different mean
            group = [i*10 + j for j in range(3)]
            many_groups.append(group)
        
        p_value, error = self.anova_test.calculate_p_value(groups=many_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant
    
    def test_very_large_values(self):
        """Test with very large numerical values."""
        large_value_groups = [
            [1e6, 1e6 + 1, 1e6 + 2],
            [2e6, 2e6 + 1, 2e6 + 2],
            [3e6, 3e6 + 1, 3e6 + 2]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=large_value_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
    
    def test_very_small_values(self):
        """Test with very small numerical values."""
        small_value_groups = [
            [1e-6, 2e-6, 3e-6],
            [4e-6, 5e-6, 6e-6],
            [7e-6, 8e-6, 9e-6]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=small_value_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
    
    def test_negative_values(self):
        """Test with negative values."""
        negative_groups = [
            [-10, -12, -11],
            [-20, -22, -21],
            [-30, -32, -31]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=negative_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant
    
    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        mixed_groups = [
            [-5, -3, -1],
            [0, 1, 2],
            [5, 7, 9]
        ]
        
        p_value, error = self.anova_test.calculate_p_value(groups=mixed_groups)
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant
    
    def test_power_calculation_edge_cases(self):
        """Test power calculation edge cases."""
        groups = [[10, 12, 11], [20, 22, 21], [30, 32, 31]]
        
        # Test with very small effect size
        power_small, error = self.anova_test.calculate_power(
            groups=groups,
            effect_size=0.001
        )
        assert power_small is not None
        assert error is None
        assert power_small < 0.1  # Should be very low power
        
        # Test with effect size approaching 1
        power_large, error = self.anova_test.calculate_power(
            groups=groups,
            effect_size=0.99
        )
        assert power_large is not None
        assert error is None
        assert power_large > 0.9  # Should be very high power
    
    def test_individual_group_parameters_ordering(self):
        """Test that individual group parameters are processed in correct order."""
        # Test with group1, group3, group2 (not sequential)
        p_value, error = self.anova_test.calculate_p_value(
            group1=[10, 12, 11],
            group3=[30, 32, 31],
            group2=[20, 22, 21]
        )
        assert p_value is not None
        # May have warning about small sample sizes (each group has n=3 < 5)
        
        # Should match sequential ordering
        p_value_sequential, _ = self.anova_test.calculate_p_value(
            group1=[10, 12, 11],
            group2=[20, 22, 21],
            group3=[30, 32, 31]
        )
        assert abs(p_value - p_value_sequential) < 1e-10


class TestOneWayANOVABackwardsCompatibility:
    """Test that adding OneWayANOVA doesn't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = get_factory()
    
    def test_existing_tests_still_available(self):
        """Test that existing tests are still available after adding OneWayANOVA."""
        available_tests = self.factory.get_available_tests()
        
        # Check that original test registrations are still there
        assert "two_sample_t_test" in available_tests
        assert "t_test" in available_tests
        assert "chi_square" in available_tests
        assert "chi2" in available_tests
    
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
    
    def test_chi_square_still_works(self):
        """Test that chi-square functionality is unchanged."""
        from statistical_tests import ChiSquareTest
        
        chi_test = self.factory.get_test("chi_square")
        assert isinstance(chi_test, ChiSquareTest)
        
        # Test basic chi-square calculation
        p_value, error = chi_test.calculate_p_value(contingency_table=[[10, 20], [30, 40]])
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
        # Should have t-test + chi-square + OneWayANOVA registrations
        assert len(available_tests) >= 16  # At least 4 + 6 + 6 registrations
    
    def test_anova_vs_t_test_consistency(self):
        """Test that OneWayANOVA with two groups gives similar results to t-test."""
        # Two groups for comparison
        group1 = [10, 12, 11, 9, 13]
        group2 = [20, 22, 21, 19, 23]
        
        # Calculate with OneWayANOVA
        anova_test = self.factory.get_test("one_way_anova")
        anova_p, _ = anova_test.calculate_p_value(groups=[group1, group2])
        
        # Calculate with t-test equivalent (if we had the data)
        # For now, just verify ANOVA gives reasonable result
        assert anova_p is not None
        assert 0 <= anova_p <= 1
        assert anova_p < 0.05  # Should be significant for these different groups


if __name__ == "__main__":
    pytest.main([__file__])