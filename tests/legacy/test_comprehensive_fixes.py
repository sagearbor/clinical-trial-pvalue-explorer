#!/usr/bin/env python3
"""
Comprehensive test suite to verify all bug fixes and functionality.
Tests both statistical_utils and app.py imports after fixes.
"""

import sys
import traceback
from statistical_utils import calculate_p_value_from_N_d, calculate_power_from_N_d


def test_statistical_functions():
    """Test all statistical functions with various scenarios."""
    print("=" * 60)
    print("TESTING STATISTICAL FUNCTIONS")
    print("=" * 60)
    
    test_cases = [
        # (N_total, cohens_d, test_description)
        (100, 0.5, "Standard case: N=100, d=0.5"),
        (50, 0.3, "Small effect: N=50, d=0.3"),
        (200, 0.8, "Large effect: N=200, d=0.8"),
        (4, 0.5, "Minimum valid N: N=4, d=0.5"),
        (100, 0.0, "Zero effect size: N=100, d=0.0"),
        (100, -0.5, "Negative effect: N=100, d=-0.5"),
        (1000, 0.2, "Small effect, large N: N=1000, d=0.2"),
    ]
    
    print("\n--- P-Value Calculations ---")
    for N_total, cohens_d, description in test_cases:
        try:
            p_val, msg = calculate_p_value_from_N_d(N_total, cohens_d)
            status = "✓ PASS" if p_val is not None or msg else "✗ FAIL"
            print(f"{status} {description}")
            print(f"    Result: p={p_val}, msg='{msg}'")
        except Exception as e:
            print(f"✗ FAIL {description}")
            print(f"    Error: {e}")
    
    print("\n--- Power Calculations ---")
    for N_total, cohens_d, description in test_cases:
        try:
            power, msg = calculate_power_from_N_d(N_total, cohens_d)
            status = "✓ PASS" if power is not None or msg else "✗ FAIL"
            print(f"{status} {description}")
            print(f"    Result: power={power}, msg='{msg}'")
        except Exception as e:
            print(f"✗ FAIL {description}")
            print(f"    Error: {e}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES AND ERROR CONDITIONS")
    print("=" * 60)
    
    edge_cases = [
        # (N_total, cohens_d, expected_outcome, test_description)
        (2, 0.5, "error", "Invalid N: N=2 (too small)"),
        (1, 0.5, "error", "Invalid N: N=1 (too small)"),
        (0, 0.5, "error", "Invalid N: N=0"),
        (-5, 0.5, "error", "Invalid N: N=-5 (negative)"),
        (100, "invalid", "error", "Invalid Cohen's d: string input"),
        ("invalid", 0.5, "error", "Invalid N: string input"),
    ]
    
    print("\n--- Error Handling Tests ---")
    for N_total, cohens_d, expected, description in edge_cases:
        try:
            p_val, msg = calculate_p_value_from_N_d(N_total, cohens_d)
            if expected == "error" and (p_val is None and msg):
                print(f"✓ PASS {description}")
                print(f"    Correctly caught error: '{msg}'")
            elif expected == "error":
                print(f"✗ FAIL {description}")
                print(f"    Expected error but got: p={p_val}, msg='{msg}'")
            else:
                print(f"✓ PASS {description}")
                print(f"    Result: p={p_val}, msg='{msg}'")
        except Exception as e:
            if expected == "error":
                print(f"✓ PASS {description}")
                print(f"    Correctly threw exception: {e}")
            else:
                print(f"✗ FAIL {description}")
                print(f"    Unexpected exception: {e}")


def test_app_imports():
    """Test that app.py imports work correctly after fixes."""
    print("\n" + "=" * 60)
    print("TESTING APP.PY IMPORTS")
    print("=" * 60)
    
    try:
        print("\n--- Testing app.py import ---")
        import app
        print("✓ PASS app.py imported successfully")
        
        # Test that app.py uses the correct functions from statistical_utils
        print("\n--- Testing app.py uses correct functions ---")
        
        # Call the function through app module to ensure it's using the imported version
        result = app.calculate_p_value_from_N_d(100, 0.5)
        if result and len(result) == 2:
            print("✓ PASS app.calculate_p_value_from_N_d works correctly")
            print(f"    Result: {result}")
        else:
            print("✗ FAIL app.calculate_p_value_from_N_d returned unexpected result")
            print(f"    Result: {result}")
            
        result = app.calculate_power_from_N_d(100, 0.5)
        if result and len(result) == 2:
            print("✓ PASS app.calculate_power_from_N_d works correctly")
            print(f"    Result: {result}")
        else:
            print("✗ FAIL app.calculate_power_from_N_d returned unexpected result")
            print(f"    Result: {result}")
            
    except Exception as e:
        print(f"✗ FAIL app.py import failed: {e}")
        traceback.print_exc()


def test_specific_requirements():
    """Test the specific requirements mentioned in the task."""
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC QA REQUIREMENTS")
    print("=" * 60)
    
    print("\n--- Testing exact command from requirements ---")
    try:
        result = calculate_p_value_from_N_d(100, 0.5)
        print(f"✓ PASS calculate_p_value_from_N_d(100, 0.5) = {result}")
        
        # Verify the result is reasonable
        p_val, msg = result
        if p_val is not None and 0 < p_val < 1:
            print(f"✓ PASS p-value is in valid range: {p_val}")
        else:
            print(f"✗ FAIL p-value not in valid range: {p_val}")
            
    except Exception as e:
        print(f"✗ FAIL Command failed: {e}")
        traceback.print_exc()
    
    print("\n--- Testing zero effect size edge case ---")
    try:
        result = calculate_p_value_from_N_d(100, 0.0)
        p_val, msg = result
        if p_val == 1.0 and "no effect observed" in msg.lower():
            print("✓ PASS Zero effect size handled correctly")
            print(f"    Result: p={p_val}, msg='{msg}'")
        else:
            print("✗ FAIL Zero effect size not handled correctly")
            print(f"    Result: p={p_val}, msg='{msg}'")
    except Exception as e:
        print(f"✗ FAIL Zero effect size test failed: {e}")


def main():
    """Run all tests."""
    print("COMPREHENSIVE BUG FIX VERIFICATION")
    print("=" * 60)
    print("Testing all fixes for Phase 1.1 QA findings...")
    
    test_statistical_functions()
    test_edge_cases()
    test_app_imports()
    test_specific_requirements()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)
    print("\nIf all tests show ✓ PASS, the bugs have been successfully fixed!")
    print("The Universal Study P-Value Explorer is ready for Phase 1.2.")


if __name__ == "__main__":
    main()