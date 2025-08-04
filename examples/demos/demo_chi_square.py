#!/usr/bin/env python3
"""
Chi-Square Test Implementation Demo

This script demonstrates the newly implemented Chi-Square test functionality
in the Clinical Trial P-Value Explorer. It shows both independence tests
and goodness-of-fit tests using the factory pattern.

Usage: python examples/demos/demo_chi_square.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from statistical_tests import get_factory


def demo_independence_test():
    """Demonstrate chi-square independence test."""
    print("=== Chi-Square Independence Test Demo ===")
    print("Testing association between treatment group and outcome")
    
    # Example: Treatment effectiveness study
    # Rows: Treatment groups (Treatment A, Treatment B)
    # Columns: Outcomes (Success, Failure)
    contingency_table = [
        [45, 15],  # Treatment A: 45 successes, 15 failures
        [30, 30]   # Treatment B: 30 successes, 30 failures
    ]
    
    print("Contingency Table:")
    print("                Success  Failure")
    print(f"Treatment A        {contingency_table[0][0]}       {contingency_table[0][1]}")
    print(f"Treatment B        {contingency_table[1][0]}       {contingency_table[1][1]}")
    print()
    
    # Get chi-square test from factory
    factory = get_factory()
    chi_test = factory.get_test('chi_square')
    
    # Calculate p-value
    p_value, error = chi_test.calculate_p_value(contingency_table=contingency_table)
    if error:
        print(f"Error: {error}")
        return
    
    # Calculate power
    power, error = chi_test.calculate_power(contingency_table=contingency_table)
    if error:
        print(f"Power calculation error: {error}")
    
    print(f"Results:")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Power: {power:.6f}" if power else "  Power: Could not calculate")
    print(f"  Interpretation: {'Significant association' if p_value < 0.05 else 'No significant association'} (α = 0.05)")
    print()


def demo_goodness_of_fit_test():
    """Demonstrate chi-square goodness-of-fit test."""
    print("=== Chi-Square Goodness-of-Fit Test Demo ===")
    print("Testing whether observed frequencies match expected distribution")
    
    # Example: Testing if treatment allocation follows expected 1:1:1 ratio
    # Categories: Treatment A, Treatment B, Control
    observed = [[20, 25, 15]]  # Observed frequencies
    expected = [[20, 20, 20]]  # Expected frequencies (equal allocation)
    
    print("Observed vs Expected Frequencies:")
    print("Category     Observed  Expected")
    print(f"Treatment A      {observed[0][0]}        {expected[0][0]}")
    print(f"Treatment B      {observed[0][1]}        {expected[0][1]}")
    print(f"Control          {observed[0][2]}        {expected[0][2]}")
    print()
    
    # Get chi-square test from factory
    factory = get_factory()
    chi_test = factory.get_test('chi_square')
    
    # Calculate p-value for goodness-of-fit
    p_value, error = chi_test.calculate_p_value(
        contingency_table=observed,
        expected_frequencies=expected
    )
    
    if error:
        print(f"Error: {error}")
        return
    
    print(f"Results:")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Interpretation: {'Significantly different from expected' if p_value < 0.05 else 'No significant difference from expected'} (α = 0.05)")
    print()


def demo_factory_aliases():
    """Demonstrate different ways to access chi-square test."""
    print("=== Factory Aliases Demo ===")
    print("Chi-square test can be accessed using multiple aliases:")
    
    factory = get_factory()
    aliases = ['chi_square', 'chi2', 'categorical_test', 'independence_test', 'goodness_of_fit']
    
    for alias in aliases:
        test = factory.get_test(alias)
        print(f"  '{alias}' -> {test.__class__.__name__}")
    
    print()


def demo_api_integration():
    """Demonstrate API integration for chi-square routing."""
    print("=== API Integration Demo ===")
    print("Testing API routing for categorical studies:")
    
    from api import map_study_type_to_test, perform_statistical_calculations
    
    # Test different study type mappings
    study_types = ['chi_square_test', 'categorical_analysis', 'association_test']
    
    for study_type in study_types:
        test_type = map_study_type_to_test(study_type)
        print(f"  '{study_type}' maps to: '{test_type}'")
    
    # Test statistical calculation through API
    params = {
        'contingency_table': [[20, 30], [40, 50]],
        'alpha': 0.05
    }
    
    result = perform_statistical_calculations('categorical_test', params)
    print(f"\nAPI calculation result:")
    print(f"  P-value: {result['calculated_p_value']:.6f}")
    print(f"  Power: {result['calculated_power']:.6f}")
    print(f"  Test used: {result['statistical_test_used']}")
    print()


def demo_edge_cases():
    """Demonstrate edge case handling."""
    print("=== Edge Cases Demo ===")
    
    factory = get_factory()
    chi_test = factory.get_test('chi_square')
    
    # Test validation
    print("Testing validation:")
    
    # Valid case
    valid, error = chi_test.validate_params(contingency_table=[[10, 20], [30, 40]])
    print(f"  Valid 2x2 table: {valid}")
    
    # Invalid case - negative frequencies
    valid, error = chi_test.validate_params(contingency_table=[[10, -5], [20, 30]])
    print(f"  Negative frequencies: {valid} ({'Error: ' + error if error else 'No error'})")
    
    # Low expected frequencies warning
    low_freq_table = [[2, 1], [1, 2]]
    p_value, warning = chi_test.calculate_p_value(contingency_table=low_freq_table)
    print(f"  Low expected frequencies: P-value = {p_value:.6f}")
    if warning:
        print(f"    {warning}")
    
    print()


def main():
    """Run all demonstrations."""
    print("Chi-Square Test Implementation - Phase 2.1 Demo")
    print("=" * 50)
    print()
    
    demo_independence_test()
    demo_goodness_of_fit_test()
    demo_factory_aliases()
    demo_api_integration()
    demo_edge_cases()
    
    print("Demo completed successfully!")
    print("\nKey Features Implemented:")
    print("✓ Independence tests for categorical data analysis")
    print("✓ Goodness-of-fit tests with expected frequencies")
    print("✓ Power analysis using Cramér's V effect size")
    print("✓ Comprehensive parameter validation")
    print("✓ Factory pattern integration with multiple aliases")
    print("✓ API routing for LLM-suggested categorical studies")
    print("✓ Proper error handling and warnings")
    print("✓ Full backwards compatibility with existing functionality")


if __name__ == "__main__":
    main()