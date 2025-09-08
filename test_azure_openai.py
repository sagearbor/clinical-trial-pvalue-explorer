import requests
import json

# Test Azure OpenAI with your complex trial
complex_trial = {
    "study_description": """A clinical trial evaluating the efficacy of a new antihypertensive drug compared to a placebo in patients with moderate hypertension might employ a mediumly complex statistical setup. The trial could use a randomized, double-blind, parallel-group design with a primary endpoint of mean change in systolic blood pressure over 12 weeks. Statistical analysis might involve a mixed-effects model for repeated measures (MMRM) to account for baseline blood pressure, time, treatment group, and interaction effects, with adjustments for covariates like age, sex, and BMI.""",
    "context": "Phase 3 RCT with repeated measures"
}

print("🧪 Testing Azure OpenAI Integration")
print("=" * 60)

response = requests.post("http://localhost:8123/process_idea", json=complex_trial, timeout=30)
if response.status_code == 200:
    data = response.json()
    
    # Check if LLM was actually used
    if data.get("llm_used") or not data.get("fallback_mode"):
        print("✅ AZURE OPENAI IS WORKING!")
        print(f"   Provider: {data.get('llm_provider_used', 'Unknown')}")
    else:
        print("⚠️ Still using pattern matching fallback")
        
    print(f"\n📊 Analysis Results:")
    print(f"   Test detected: {data.get('statistical_test_used')}")
    print(f"   Rationale: {data.get('rationale', '')[:100]}...")
    print(f"   P-value: {data.get('calculated_p_value', 'N/A')}")
    print(f"   Power: {data.get('calculated_power', 0)*100:.1f}%")
    
    if data.get("llm_warning"):
        print(f"\n⚠️ Warning: {data.get('llm_warning')}")
    
    # Show raw response for debugging
    print(f"\n🔍 Debug Info:")
    print(f"   Fallback mode: {data.get('fallback_mode', False)}")
    print(f"   Default used: {data.get('default_used', False)}")
    print(f"   LLM Provider: {data.get('llm_provider_used')}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text[:500])
