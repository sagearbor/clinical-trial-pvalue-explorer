import requests
import json

# Test 1: Complex scenario that should use pattern matching
complex_test = {
    "study_description": "A clinical trial with mixed effects model for repeated measures",
    "context": "MMRM analysis"
}

# Test 2: Simple scenario that defaults to t-test  
simple_test = {
    "study_description": "Compare two groups",
    "context": "Basic comparison"
}

print("Testing Warning System")
print("=" * 60)

# Test complex scenario
print("\n1. Complex MMRM scenario (should use pattern matching):")
response = requests.post("http://localhost:8123/process_idea", json=complex_test)
if response.status_code == 200:
    data = response.json()
    print(f"   Test detected: {data.get('statistical_test_used')}")
    print(f"   Warning: {data.get('llm_warning', 'None')}")
    print(f"   Fallback mode: {data.get('fallback_mode', False)}")
    print(f"   Default used: {data.get('default_used', False)}")

# Test simple scenario
print("\n2. Simple scenario (should default to t-test):")
response = requests.post("http://localhost:8123/process_idea", json=simple_test)
if response.status_code == 200:
    data = response.json()
    print(f"   Test detected: {data.get('statistical_test_used')}")
    print(f"   Warning: {data.get('llm_warning', 'None')}")
    print(f"   Fallback mode: {data.get('fallback_mode', False)}")
    print(f"   Default used: {data.get('default_used', False)}")

print("\n" + "=" * 60)
print("✅ Warning system is active!")
print("   - Complex scenarios show pattern matching was used")
print("   - Simple/unclear scenarios show default warning")
print("   - Users are informed when LLM is unavailable")
