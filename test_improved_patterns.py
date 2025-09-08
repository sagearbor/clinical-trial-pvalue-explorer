#!/usr/bin/env python3
"""Test improved patterns on previously missed cases"""

from src.research_intelligence import _extract_sample_size

# Test cases that were previously missed
test_cases = [
    ("markers. A total of 100 gastric cancer patients", 100),
    ("-humans. A total of 22 articles were analyzed", 22),
    ("this review analyzed 36 peer-reviewed journals", 36),
]

print("Testing improved patterns on previously missed cases:")
print("=" * 60)

for text, expected in test_cases:
    result = _extract_sample_size(text)
    status = "✅" if result == expected else "❌"
    print(f"{status} Text: '{text[:40]}...'")
    print(f"   Expected: {expected}, Got: {result}")
    print()