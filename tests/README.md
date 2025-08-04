# Test Suite Directory

This directory contains the comprehensive test suite for the Universal Study P-Value Explorer.

## Active Test Files

### Core Statistical Tests
- `test_app.py` - Main application and backwards compatibility tests (2 tests)
- `test_chi_square.py` - Chi-square test implementation tests (44 tests)
- `test_correlation.py` - Correlation analysis tests (54 tests)  
- `test_one_way_anova.py` - One-way ANOVA tests (53 tests)

### System Integration Tests
- `test_factory_pattern.py` - Statistical test factory tests (22 tests)
- `test_enhanced_api_endpoints.py` - Enhanced API endpoint tests (6 tests)
- `test_frontend_integration.py` - Frontend integration tests (6 tests)

### Legacy Tests (`/legacy/`)
Historical test files from development phases:
- `test_enhanced_api.py` - Early API enhancement tests
- `test_comprehensive_fixes.py` - Bug fix validation tests
- `test_backwards_compatibility.py` - Legacy compatibility tests

## Test Coverage Summary

**Total Test Functions**: 195+ across all components
**Pass Rate**: 99.5% (194/195 tests passing)
**Coverage Areas**:
- Statistical calculations and edge cases
- API endpoint functionality
- Frontend component integration  
- Factory pattern and extensibility
- Error handling and validation
- Backwards compatibility

## Running Tests

### Individual Test Categories
```bash
# Core statistical tests
python -m pytest tests/test_chi_square.py -v
python -m pytest tests/test_correlation.py -v
python -m pytest tests/test_one_way_anova.py -v

# Integration tests
python -m pytest test_factory_pattern.py -v
python -m pytest test_enhanced_api_endpoints.py -v
python -m pytest test_frontend_integration.py -v

# Backwards compatibility
python -m pytest tests/test_app.py -v
```

### Full Test Suite
```bash
# Run all active tests
python -m pytest tests/ test_factory_pattern.py test_enhanced_api_endpoints.py test_frontend_integration.py -v

# Quick validation
python -m pytest tests/test_app.py tests/test_correlation.py -v
```

### Legacy Tests
```bash
# Run archived tests (may have dependencies on moved files)
python -m pytest tests/legacy/ -v
```

## Test Organization

### Naming Convention
- `test_[component].py` - Component-specific tests
- `Test[ClassName]` - Test class for specific functionality
- `test_[specific_feature]` - Individual test methods

### Test Categories
- **Unit Tests**: Individual function validation
- **Integration Tests**: Cross-component functionality  
- **Edge Case Tests**: Boundary conditions and error handling
- **Regression Tests**: Ensure changes don't break existing features
- **Compatibility Tests**: Backwards compatibility validation

### Quality Standards
- Comprehensive parameter validation testing
- Edge case coverage for statistical calculations
- Error handling verification
- Performance baseline testing
- Documentation and example validation

## Adding New Tests

When implementing new features:
1. Create test file following naming convention
2. Include comprehensive edge case coverage
3. Test both success and failure scenarios
4. Validate integration with existing components
5. Update this README with test counts and coverage

The test suite serves as both quality assurance and living documentation of system capabilities.