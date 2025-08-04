# Adding New Statistical Tests to the Factory Pattern

This guide explains how to add new statistical tests to the Phase 1.2 factory pattern architecture.

## Overview

The statistical test factory pattern enables easy addition of new statistical tests without modifying existing code. All tests implement a common interface defined by the `StatisticalTest` abstract base class.

## Step 1: Create Your Test Class

Create a new class that inherits from `StatisticalTest` in `statistical_tests.py`:

```python
class YourNewTest(StatisticalTest):
    """
    Your statistical test implementation.
    
    Brief description of what this test does and its assumptions.
    """
    
    def get_required_params(self) -> List[str]:
        """Return list of required parameter names."""
        return ["param1", "param2", "param3"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        """Validate input parameters."""
        # Check required parameters are present
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        # Add your specific validation logic here
        param1 = params.get("param1")
        if not isinstance(param1, (int, float)) or param1 <= 0:
            return False, "param1 must be a positive number"
        
        # Return True if all validations pass
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate p-value for your test."""
        # Validate parameters first
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            # Extract parameters
            param1 = params["param1"]
            param2 = params["param2"]
            
            # Implement your p-value calculation logic here
            # This is where you'd use scipy.stats or other libraries
            p_value = your_calculation_logic(param1, param2)
            
            return p_value, None
            
        except Exception as e:
            return None, f"Error during p-value calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        """Calculate statistical power for your test."""
        # Add default alpha if not provided
        if "alpha" not in params:
            params = dict(params)
            params["alpha"] = 0.05
        
        # Validate parameters
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            # Extract parameters
            param1 = params["param1"]
            param2 = params["param2"]
            alpha = params["alpha"]
            
            # Implement your power calculation logic here
            power = your_power_calculation_logic(param1, param2, alpha)
            
            return power, None
            
        except Exception as e:
            return None, f"Error during power calculation: {str(e)}"
```

## Step 2: Register Your Test in the Factory

Add registration to the `_register_default_tests()` method in `StatisticalTestFactory`:

```python
def _register_default_tests(self):
    """Register default statistical tests."""
    # Existing registrations...
    self.register_test("two_sample_t_test", TwoSampleTTest)
    
    # Add your new test
    self.register_test("your_new_test", YourNewTest)
    # Add aliases if needed
    self.register_test("new_test_alias", YourNewTest)
```

## Step 3: Update API Mapping

Add mapping from LLM study types to your test in `api.py`:

```python
def map_study_type_to_test(suggested_study_type: str) -> str:
    """Map LLM-suggested study types to factory test identifiers."""
    type_mapping = {
        # Existing mappings...
        "two_sample_t_test": "two_sample_t_test",
        
        # Add your new test mappings
        "your_study_type": "your_new_test",
        "alternative_name": "your_new_test",
    }
    # ... rest of function
```

## Step 4: Create Tests

Create comprehensive tests for your new statistical test:

```python
class TestYourNewTest:
    """Test your new statistical test implementation."""
    
    def test_required_params(self):
        """Test required parameters are correctly defined."""
        test = YourNewTest()
        required = test.get_required_params()
        assert "param1" in required
        assert "param2" in required
    
    def test_validate_params_valid(self):
        """Test parameter validation with valid inputs."""
        test = YourNewTest()
        is_valid, error = test.validate_params(param1=100, param2=0.5)
        assert is_valid
        assert error is None
    
    def test_validate_params_invalid(self):
        """Test parameter validation with invalid inputs."""
        test = YourNewTest()
        is_valid, error = test.validate_params(param1=-1, param2=0.5)
        assert not is_valid
        assert "positive number" in error
    
    def test_calculate_p_value_valid(self):
        """Test p-value calculation with valid inputs."""
        test = YourNewTest()
        p_value, error = test.calculate_p_value(param1=100, param2=0.5)
        assert p_value is not None
        assert error is None
        assert 0 <= p_value <= 1
    
    def test_calculate_power_valid(self):
        """Test power calculation with valid inputs."""
        test = YourNewTest()
        power, error = test.calculate_power(param1=100, param2=0.5, alpha=0.05)
        assert power is not None
        assert error is None
        assert 0 <= power <= 1
```

## Step 5: Update LLM Prompts (Optional)

If your test should be suggested by the LLM, update the enhanced system prompt in `api.py` to include your new test type in the examples and available options.

## Step 6: Test Integration

Test that your new test works with the factory pattern:

```python
# Test factory integration
factory = get_factory()
test = factory.get_test("your_new_test")
p_value, error = test.calculate_p_value(param1=100, param2=0.5)
print(f"P-value: {p_value}, Error: {error}")
```

## Best Practices

1. **Validation**: Always validate parameters thoroughly
2. **Error Handling**: Use try-catch blocks and return meaningful error messages
3. **Documentation**: Include clear docstrings explaining the test and its assumptions
4. **Testing**: Write comprehensive tests covering edge cases
5. **Backwards Compatibility**: Ensure your test doesn't break existing functionality
6. **Consistent Interface**: Follow the same return patterns as existing tests

## Example: Adding a Chi-Square Test

Here's a complete example of adding a chi-square test:

```python
class ChiSquareTest(StatisticalTest):
    """
    Chi-square test for independence in contingency tables.
    
    Assumes:
    - Categorical data in contingency table format
    - Expected frequencies ≥ 5 in each cell
    - Independent observations
    """
    
    def get_required_params(self) -> List[str]:
        return ["contingency_table"]
    
    def validate_params(self, **params) -> Tuple[bool, Optional[str]]:
        required_params = self.get_required_params()
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        
        table = params.get("contingency_table")
        if not isinstance(table, (list, np.ndarray)):
            return False, "contingency_table must be a 2D array or list"
        
        # Additional validation for chi-square assumptions
        table = np.array(table)
        if table.ndim != 2:
            return False, "contingency_table must be 2-dimensional"
        
        if np.any(table < 0):
            return False, "All frequencies must be non-negative"
        
        return True, None
    
    def calculate_p_value(self, **params) -> Tuple[Optional[float], Optional[str]]:
        is_valid, error_msg = self.validate_params(**params)
        if not is_valid:
            return None, error_msg
        
        try:
            from scipy.stats import chi2_contingency
            table = np.array(params["contingency_table"])
            chi2_stat, p_value, dof, expected = chi2_contingency(table)
            return p_value, None
        except Exception as e:
            return None, f"Error during chi-square calculation: {str(e)}"
    
    def calculate_power(self, **params) -> Tuple[Optional[float], Optional[str]]:
        # Chi-square power calculation is more complex and would require
        # effect size (Cramer's V) and sample size
        return None, "Power calculation not yet implemented for chi-square test"
```

This framework makes it easy to add any new statistical test while maintaining consistency and reliability across the system.