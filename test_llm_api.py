import requests
import json

# Test LLM processing on the correct port
url = "http://localhost:8123/process_idea"
data = {
    "idea": "I want to test if a new drug reduces blood pressure more effectively than the standard treatment. We have 100 patients in each group, measuring systolic BP reduction after 12 weeks.",
    "context": "Phase 3 clinical trial for hypertension medication"
}

print("Testing LLM Idea Processing...")
print("-" * 50)
print(f"Research Idea: {data['idea']}")
print("-" * 50)

try:
    response = requests.post(url, json=data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        print("\n✅ LLM Analysis Successful!")
        print(f"\nSuggested Study Type: {result.get('suggested_study_type', 'N/A')}")
        print(f"Statistical Test: {result.get('statistical_test_used', 'N/A')}")
        print(f"Calculated P-value: {result.get('calculated_p_value', 'N/A')}")
        print(f"Statistical Power: {result.get('calculated_power', 'N/A')}")
        print(f"Effect Size: {result.get('effect_size', 'N/A')}")
        
        if 'llm_interpretation' in result:
            print(f"\nLLM Interpretation:\n{result['llm_interpretation'][:500]}...")
    else:
        print(f"Error: Status {response.status_code}")
        print(response.text[:500])
except Exception as e:
    print(f"Error: {e}")
