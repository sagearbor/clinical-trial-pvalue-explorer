#!/usr/bin/env python3
"""Test p-value extraction patterns"""

from src.research_intelligence import _extract_p_value

def test_pvalue_patterns():
    """Test various p-value patterns"""
    
    test_cases = [
        ("The results were significant (p < 0.001).", 0.001),
        ("We found p = 0.045 for the main effect.", 0.045),
        ("P < 0.0001 was observed.", 0.0001),
        ("The difference was significant with p-value < 0.05.", 0.05),
        ("Results showed p = 2.3e-10 for the interaction.", 2.3e-10),
        ("Not significant (p = 0.34).", 0.34),
        ("Multiple comparisons: p < 0.01 for A, p = 0.02 for B.", 0.01),  # Should return smallest
        ("The test yielded P < .001", 0.001),
        ("Statistical significance at p ≤ 0.05 level", None),  # ≤ not supported yet
        ("No p-value mentioned in this text.", None),
    ]
    
    print("=" * 80)
    print("TESTING P-VALUE EXTRACTION PATTERNS")
    print("=" * 80)
    
    for text, expected in test_cases:
        result = _extract_p_value(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} Text: {text[:50]}...")
        print(f"   Expected: {expected}, Got: {result}")
        if result != expected and expected is not None:
            print(f"   ERROR: Mismatch!")
        print()

if __name__ == "__main__":
    test_pvalue_patterns()