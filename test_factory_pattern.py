"""
Comprehensive tests for Phase 1.2 Statistical Test Factory Pattern.

This test suite validates:
1. Factory pattern functionality
2. Statistical test implementations
3. Backwards compatibility with existing functions
4. API integration with factory pattern
5. Error handling and edge cases

Run with: python -m pytest test_factory_pattern.py -v
"""

import pytest
import numpy as np
from typing import Tuple, Optional

# Import factory pattern components
from statistical_tests import (
    StatisticalTest, 
    TwoSampleTTest, 
    StatisticalTestFactory,
    get_factory,
    # Backwards compatibility functions
    validate_statistical_inputs,
    calculate_p_value_from_N_d,
    calculate_power_from_N_d
)

# Import original functions for comparison
from statistical_utils import (
    validate_statistical_inputs as original_validate,
    calculate_p_value_from_N_d as original_p_value,
    calculate_power_from_N_d as original_power_calc
)


class TestStatisticalTestFactory:
    """Test the factory pattern implementation."""
    
    def test_factory_initialization(self):
        """Test factory initializes with default tests."""
        factory = StatisticalTestFactory()
        available_tests = factory.get_available_tests()
        
        # Should have at least the t-test and its aliases
        assert "two_sample_t_test" in available_tests
        assert "t_test" in available_tests
        assert "independent_samples_t_test" in available_tests
        assert len(available_tests) >= 3
    
    def test_factory_get_test(self):
        """Test factory can create test instances."""
        factory = StatisticalTestFactory()
        
        # Test getting a valid test
        test = factory.get_test("two_sample_t_test")
        assert isinstance(test, TwoSampleTTest)
        assert isinstance(test, StatisticalTest)
        
        # Test case insensitive
        test_upper = factory.get_test("TWO_SAMPLE_T_TEST")
        assert isinstance(test_upper, TwoSampleTTest)
    
    def test_factory_invalid_test(self):
        """Test factory raises error for invalid test types."""
        factory = StatisticalTestFactory()
        
        with pytest.raises(ValueError, match="Unknown test type"):
            factory.get_test("nonexistent_test")
    
    def test_factory_is_test_available(self):
        """Test factory test availability checking."""
        factory = StatisticalTestFactory()
        
        assert factory.is_test_available("two_sample_t_test")
        assert factory.is_test_available("T_TEST")  # Case insensitive
        assert not factory.is_test_available("nonexistent_test")
    
    def test_factory_register_new_test(self):
        """Test factory can register new test types."""
        factory = StatisticalTestFactory()
        
        # Create a mock test class
        class MockTest(StatisticalTest):
            def calculate_p_value(self, **params):
                return 0.05, None
            def calculate_power(self, **params):
                return 0.8, None
            def get_required_params(self):
                return ["param1"]
            def validate_params(self, **params):
                return True, None
        
        # Register the test
        factory.register_test("mock_test", MockTest)
        
        # Verify it's available
        assert factory.is_test_available("mock_test")
        test = factory.get_test("mock_test")
        assert isinstance(test, MockTest)
    
    def test_factory_register_invalid_test(self):
        """Test factory rejects invalid test classes."""
        factory = StatisticalTestFactory()
        
        class InvalidTest:
            pass
        
        with pytest.raises(ValueError, match="must inherit from StatisticalTest"):
            factory.register_test("invalid_test", InvalidTest)
    
    def test_global_factory(self):
        """Test global factory instance."""
        factory1 = get_factory()
        factory2 = get_factory()
        
        # Should be the same instance
        assert factory1 is factory2
        assert isinstance(factory1, StatisticalTestFactory)


class TestTwoSampleTTest:
    """Test the TwoSampleTTest implementation."""
    
    def test_required_params(self):
        """Test required parameters are correctly defined."""
        test = TwoSampleTTest()
        required = test.get_required_params()
        
        assert "N_total" in required
        assert "cohens_d" in required
        assert len(required) == 2
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid inputs."""
        test = TwoSampleTTest()
        
        # Valid parameters
        is_valid, error = test.validate_params(N_total=100, cohens_d=0.5)
        assert is_valid
        assert error is None
        
        # Valid with alpha for power calculation
        is_valid, error = test.validate_params(N_total=100, cohens_d=0.5, alpha=0.05)
        assert is_valid
        assert error is None
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid inputs."""
        test = TwoSampleTTest()
        
        # Missing required parameter
        is_valid, error = test.validate_params(N_total=100)
        assert not is_valid
        assert "Missing required parameter: cohens_d" in error
        
        # Invalid N_total
        is_valid, error = test.validate_params(N_total=2, cohens_d=0.5)
        assert not is_valid
        assert "Total N must be" in error and "greater than 2" in error
        
        # Invalid cohens_d type
        is_valid, error = test.validate_params(N_total=100, cohens_d="invalid")
        assert not is_valid
        assert "Cohen's d must be a number" in error
        
        # Invalid alpha
        is_valid, error = test.validate_params(N_total=100, cohens_d=0.5, alpha=1.5)
        assert not is_valid
        assert "Alpha must be a number between 0 and 1" in error
    
    def test_calculate_p_value_valid(self):
        """Test p-value calculation with valid inputs."""
        test = TwoSampleTTest()
        
        # Test with medium effect size
        p_value, error = test.calculate_p_value(N_total=100, cohens_d=0.5)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
        assert p_value < 0.05  # Should be significant with this effect size
        
        # Test with zero effect size
        p_value, error = test.calculate_p_value(N_total=100, cohens_d=0.0)
        assert p_value == 1.0
        assert "Cohen's d = 0" in error
    
    def test_calculate_p_value_invalid(self):
        """Test p-value calculation with invalid inputs."""
        test = TwoSampleTTest()
        
        # Invalid parameters
        p_value, error = test.calculate_p_value(N_total=2, cohens_d=0.5)
        assert p_value is None
        assert error is not None
        assert "Total N must be" in error and "greater than 2" in error
    
    def test_calculate_power_valid(self):
        """Test power calculation with valid inputs."""
        test = TwoSampleTTest()
        
        # Test with medium effect size
        power, error = test.calculate_power(N_total=100, cohens_d=0.5, alpha=0.05)
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
        assert power > 0.6  # Should have reasonable power with this effect size
        
        # Test with default alpha
        power, error = test.calculate_power(N_total=100, cohens_d=0.5)
        assert power is not None
        assert error is None
    
    def test_calculate_power_invalid(self):
        """Test power calculation with invalid inputs."""
        test = TwoSampleTTest()
        
        # Invalid parameters
        power, error = test.calculate_power(N_total=2, cohens_d=0.5)
        assert power is None
        assert error is not None
        assert "Total N must be" in error and "greater than 2" in error


class TestBackwardsCompatibility:
    """Test backwards compatibility with original statistical_utils functions."""
    
    def test_validate_statistical_inputs_compatibility(self):
        """Test that new validation matches original."""
        test_cases = [
            (100, 0.5),  # Valid case
            (50, 0.3),   # Valid case
            (2, 0.5),    # Invalid N_total
            (10, "invalid"),  # Invalid cohens_d
        ]
        
        for N_total, cohens_d in test_cases:
            try:
                original_result = original_validate(N_total, cohens_d)
                new_result = validate_statistical_inputs(N_total, cohens_d)
                assert original_result == new_result
            except Exception as e:
                # Both should raise similar exceptions for invalid inputs
                with pytest.raises(type(e)):
                    validate_statistical_inputs(N_total, cohens_d)
    
    def test_calculate_p_value_compatibility(self):
        """Test that new p-value calculation matches original."""
        test_cases = [
            (100, 0.5),
            (50, 0.3),
            (30, 0.8),
            (200, 0.2),
        ]
        
        for N_total, cohens_d in test_cases:
            original_p, original_error = original_p_value(N_total, cohens_d)
            new_p, new_error = calculate_p_value_from_N_d(N_total, cohens_d)
            
            if original_p is not None and new_p is not None:
                # Should be very close (within numerical precision)
                assert abs(original_p - new_p) < 1e-10
            else:
                # Both should be None for invalid cases
                assert original_p == new_p
    
    def test_calculate_power_compatibility(self):
        """Test that new power calculation matches original."""
        test_cases = [
            (100, 0.5, 0.05),
            (50, 0.3, 0.01),
            (30, 0.8, 0.05),
            (200, 0.2, 0.05),
        ]
        
        for N_total, cohens_d, alpha in test_cases:
            original_power, original_error = original_power_calc(N_total, cohens_d, alpha)
            new_power, new_error = calculate_power_from_N_d(N_total, cohens_d, alpha)
            
            if original_power is not None and new_power is not None:
                # Should be very close (within numerical precision)
                assert abs(original_power - new_power) < 1e-10
            else:
                # Both should be None for invalid cases
                assert original_power == new_power
    
    def test_factory_vs_direct_calculation(self):
        """Test that factory pattern gives same results as direct calculation."""
        factory = get_factory()
        test = factory.get_test("two_sample_t_test")
        
        test_cases = [
            (100, 0.5),
            (50, 0.3),
            (30, 0.8),
        ]
        
        for N_total, cohens_d in test_cases:
            # Compare p-values
            direct_p, _ = calculate_p_value_from_N_d(N_total, cohens_d)
            factory_p, _ = test.calculate_p_value(N_total=N_total, cohens_d=cohens_d)
            
            if direct_p is not None and factory_p is not None:
                assert abs(direct_p - factory_p) < 1e-10
            
            # Compare power
            direct_power, _ = calculate_power_from_N_d(N_total, cohens_d, 0.05)
            factory_power, _ = test.calculate_power(N_total=N_total, cohens_d=cohens_d, alpha=0.05)
            
            if direct_power is not None and factory_power is not None:
                assert abs(direct_power - factory_power) < 1e-10


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_factory_error_handling(self):
        """Test factory handles errors gracefully."""
        factory = StatisticalTestFactory()
        
        # Test with None input
        with pytest.raises(ValueError):
            factory.get_test(None)
        
        # Test with empty string
        with pytest.raises(ValueError):
            factory.get_test("")
    
    def test_statistical_test_edge_cases(self):
        """Test statistical test handles edge cases."""
        test = TwoSampleTTest()
        
        # Very small sample size
        p_value, error = test.calculate_p_value(N_total=4, cohens_d=0.5)
        assert p_value is not None or error is not None
        
        # Very large effect size
        p_value, error = test.calculate_p_value(N_total=100, cohens_d=5.0)
        assert p_value is not None
        assert p_value < 0.001  # Should be highly significant
        
        # Negative effect size
        p_value, error = test.calculate_p_value(N_total=100, cohens_d=-0.5)
        assert p_value is not None
        
        # Very small effect size
        p_value, error = test.calculate_p_value(N_total=100, cohens_d=0.01)
        assert p_value is not None
        assert p_value > 0.05  # Should not be significant


class TestAPIIntegration:
    """Test API helper functions for factory integration."""
    
    def test_map_study_type_to_test(self):
        """Test study type mapping function."""
        from api import map_study_type_to_test
        
        # Test known mappings
        assert map_study_type_to_test("two_sample_t_test") == "two_sample_t_test"
        assert map_study_type_to_test("t_test") == "two_sample_t_test"
        assert map_study_type_to_test("independent_samples_t_test") == "two_sample_t_test"
        
        # Test case insensitive
        assert map_study_type_to_test("TWO_SAMPLE_T_TEST") == "two_sample_t_test"
        
        # Test fallback
        assert map_study_type_to_test("unknown_test") == "two_sample_t_test"
        assert map_study_type_to_test(None) == "two_sample_t_test"
    
    def test_perform_statistical_calculations(self):
        """Test statistical calculations function."""
        from api import perform_statistical_calculations
        
        # Test valid parameters
        parameters = {
            'total_n': 100,
            'cohens_d': 0.5,
            'alpha': 0.05
        }
        
        result = perform_statistical_calculations("two_sample_t_test", parameters)
        
        assert result['calculated_p_value'] is not None
        assert result['calculated_power'] is not None
        assert result['statistical_test_used'] == "two_sample_t_test"
        assert result['calculation_error'] is None
        
        # Test missing parameters
        incomplete_params = {'total_n': 100}
        result = perform_statistical_calculations("two_sample_t_test", incomplete_params)
        
        assert result['calculated_p_value'] is None
        assert result['calculated_power'] is None
        assert "Missing required parameters" in result['calculation_error']


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])