import requests
import json

# Test with correct field names
url = "http://localhost:8123/process_idea"
data = {
    "study_description": "I want to test if a new drug reduces blood pressure more effectively than the standard treatment. We have 100 patients in each group, measuring systolic BP reduction after 12 weeks. The control group receives the current standard ACE inhibitor, while the treatment group receives our novel compound.",
    "context": "Phase 3 randomized controlled trial for hypertension"
}

print("🧪 Testing LLM-Powered Statistical Analysis")
print("=" * 60)
print(f"Study Description:\n{data['study_description']}")
print("=" * 60)

try:
    response = requests.post(url, json=data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        print("\n✅ LLM ANALYSIS COMPLETE!\n")
        print("📊 Statistical Results:")
        print(f"  • Suggested Test: {result.get('statistical_test_used', 'N/A')}")
        print(f"  • P-value: {result.get('calculated_p_value', 'N/A'):.4f}" if result.get('calculated_p_value') else "  • P-value: N/A")
        print(f"  • Statistical Power: {result.get('calculated_power', 'N/A'):.2%}" if result.get('calculated_power') else "  • Statistical Power: N/A")
        print(f"  • Effect Size: {result.get('effect_size', 'N/A'):.3f}" if result.get('effect_size') else "  • Effect Size: N/A")
        print(f"  • Sample Size: {result.get('parameters', {}).get('n_total', 'N/A')}")
        
        if 'llm_interpretation' in result:
            print(f"\n🤖 AI Interpretation:")
            print(f"{result['llm_interpretation'][:300]}...")
            
        if 'recommendations' in result:
            print(f"\n💡 Recommendations:")
            for rec in result['recommendations'][:3]:
                print(f"  • {rec}")
                
        print("\n✨ The LLM successfully analyzed your research idea!")
        print("   You can now upload documents or enter more complex study designs.")
        
    else:
        print(f"❌ Error: Status {response.status_code}")
        print(response.text[:500])
except requests.exceptions.Timeout:
    print("⏱️ Request timed out - LLM may be processing. Check API keys in .env file.")
except Exception as e:
    print(f"❌ Error: {e}")
