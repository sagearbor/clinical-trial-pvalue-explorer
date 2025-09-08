import requests
import json

# Your complex clinical trial example
complex_trial = {
    "study_description": """A clinical trial evaluating the efficacy of a new antihypertensive drug compared to a placebo in patients with moderate hypertension might employ a mediumly complex statistical setup. The trial could use a randomized, double-blind, parallel-group design with a primary endpoint of mean change in systolic blood pressure over 12 weeks. Statistical analysis might involve a mixed-effects model for repeated measures (MMRM) to account for baseline blood pressure, time, treatment group, and interaction effects, with adjustments for covariates like age, sex, and BMI. Secondary endpoints, such as diastolic blood pressure and adverse event rates, could be analyzed using ANCOVA and logistic regression, respectively, with multiplicity adjustments via Bonferroni correction.""",
    "context": "Phase 3 RCT with repeated measures"
}

print("Testing Complex Clinical Trial Scenario")
print("=" * 60)
print("This should detect MMRM/Mixed Effects Model, NOT t-test!")
print("=" * 60)

response = requests.post("http://localhost:8123/process_idea", json=complex_trial, timeout=30)
if response.status_code == 200:
    result = response.json()
    detected = result.get('statistical_test_used')
    if detected in ['mixed_effects', 'mmrm', 'mixed_model']:
        print(f"\n✅ SUCCESS! Correctly detected: {detected}")
        print(f"   Rationale: {result.get('rationale')}")
    else:
        print(f"\n❌ PROBLEM: Detected Test = {detected}")
        print(f"   Should be: mixed_effects or MMRM")
    print(f"\nFull response: {json.dumps(result, indent=2)[:500]}")
else:
    print(f"Error: {response.status_code}")

# Now test available tests
print("\n" + "=" * 60)
print("Checking Available Tests Endpoint:")
response = requests.get("http://localhost:8123/available_tests")
if response.status_code == 200:
    tests = response.json()
    print(f"✅ Found {len(tests)} available tests!")
    print("\nCategories available:")
    categories = set(t['category'] for t in tests)
    for cat in sorted(categories):
        cat_tests = [t['label'] for t in tests if t['category'] == cat]
        print(f"  • {cat}: {', '.join(cat_tests[:3])}...")
else:
    print(f"❌ Available tests endpoint error: {response.status_code}")
