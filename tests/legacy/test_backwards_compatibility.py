#!/usr/bin/env python3
"""
Test backwards compatibility between original and new implementations.
"""

from statistical_utils import calculate_p_value_from_N_d as original_p, calculate_power_from_N_d as original_power
from statistical_tests import calculate_p_value_from_N_d as new_p, calculate_power_from_N_d as new_power, get_factory

def test_backwards_compatibility():
    """Test backwards compatibility with multiple scenarios."""
    test_cases = [
        (100, 0.5),
        (50, 0.3), 
        (30, 0.8),
        (200, 0.2),
        (128, 0.4)
    ]

    print('Testing backwards compatibility...')
    all_compatible = True

    for n, d in test_cases:
        # Test p-values
        orig_p_val, orig_err = original_p(n, d)
        new_p_val, new_err = new_p(n, d)
        
        if orig_p_val is not None and new_p_val is not None:
            diff = abs(orig_p_val - new_p_val)
            if diff > 1e-10:
                print(f'P-value mismatch for N={n}, d={d}: {diff}')
                all_compatible = False
        elif orig_p_val != new_p_val:  # Both should be None
            print(f'P-value None mismatch for N={n}, d={d}')
            all_compatible = False
        
        # Test power
        orig_pow, orig_err = original_power(n, d, 0.05)
        new_pow, new_err = new_power(n, d, 0.05)
        
        if orig_pow is not None and new_pow is not None:
            diff = abs(orig_pow - new_pow)
            if diff > 1e-10:
                print(f'Power mismatch for N={n}, d={d}: {diff}')
                all_compatible = False
        elif orig_pow != new_pow:  # Both should be None
            print(f'Power None mismatch for N={n}, d={d}')
            all_compatible = False

    if all_compatible:
        print('✓ All backwards compatibility tests passed!')
    else:
        print('✗ Some backwards compatibility issues found')

    # Test factory pattern direct access
    print('\nTesting factory pattern...')
    factory = get_factory()
    test = factory.get_test('two_sample_t_test')

    for n, d in test_cases[:2]:  # Test a few cases
        factory_p, _ = test.calculate_p_value(N_total=n, cohens_d=d)
        direct_p, _ = new_p(n, d)
        
        if factory_p is not None and direct_p is not None:
            diff = abs(factory_p - direct_p)
            if diff > 1e-10:
                print(f'Factory vs direct mismatch for N={n}, d={d}: {diff}')
            else:
                print(f'✓ Factory matches direct for N={n}, d={d}')

    print('Backwards compatibility validation complete!')
    return all_compatible

if __name__ == "__main__":
    test_backwards_compatibility()