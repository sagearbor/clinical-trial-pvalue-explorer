#!/usr/bin/env python3
"""
Test script for Phase 1.1 Enhanced LLM Integration
Tests both the new enhanced analysis endpoint and backwards compatibility
"""

import requests
import json
import sys
import time

# Configuration
BASE_URL = "http://localhost:8000"
ENHANCED_ENDPOINT = f"{BASE_URL}/analyze_study"
LEGACY_ENDPOINT = f"{BASE_URL}/process_idea"
HEALTH_ENDPOINT = f"{BASE_URL}/health"

# Test data
TEST_RESEARCH_IDEAS = [
    {
        "name": "Simple Two-Group Comparison",
        "text_idea": "I want to test if a new cognitive behavioral therapy intervention reduces anxiety scores compared to standard care in patients with generalized anxiety disorder."
    },
    {
        "name": "ANOVA Design",
        "text_idea": "I'm planning a study to compare blood pressure reduction across four different medication dosages (0mg, 5mg, 10mg, 20mg) in hypertensive patients."
    },
    {
        "name": "Survival Analysis",
        "text_idea": "I want to investigate whether a new cancer treatment extends progression-free survival time compared to the current standard treatment."
    },
    {
        "name": "Categorical Outcome",
        "text_idea": "I'm studying whether a patient education program increases medication adherence rates (yes/no) compared to standard care."
    },
    {
        "name": "Observational Study",
        "text_idea": "I want to examine the relationship between daily exercise duration and BMI in a population-based cohort study."
    }
]

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed - Provider: {data.get('provider', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_enhanced_endpoint(idea_data):
    """Test the new enhanced analysis endpoint"""
    print(f"\n🧪 Testing Enhanced Endpoint: {idea_data['name']}")
    print(f"Research Idea: {idea_data['text_idea'][:100]}...")
    
    payload = {"text_idea": idea_data["text_idea"]}
    
    try:
        response = requests.post(ENHANCED_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for error in response
            if data.get("error"):
                print(f"❌ Enhanced endpoint returned error: {data['error']}")
                return False
            
            # Validate enhanced response structure
            required_fields = ['suggested_study_type', 'rationale', 'data_type', 'study_design']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            
            # Display results
            print(f"✅ Suggested Study Type: {data.get('suggested_study_type')}")
            print(f"✅ Data Type: {data.get('data_type')}")
            print(f"✅ Study Design: {data.get('study_design')}")
            print(f"✅ Confidence Level: {data.get('confidence_level', 'N/A')}")
            
            # Check parameters
            parameters = data.get('parameters', {})
            if parameters:
                print(f"✅ Sample Size (N): {parameters.get('total_n', 'N/A')}")
                print(f"✅ Effect Size: {parameters.get('effect_size_value', 'N/A')} ({parameters.get('effect_size_type', 'N/A')})")
            
            # Check alternatives
            alternatives = data.get('alternative_tests', [])
            if alternatives:
                print(f"✅ Alternative Tests: {', '.join(alternatives[:3])}")
            
            # Check backwards compatibility fields
            if data.get('initial_N') and data.get('initial_cohens_d'):
                print(f"✅ Backwards Compatibility: N={data['initial_N']}, d={data['initial_cohens_d']}")
            
            print(f"✅ Provider Used: {data.get('llm_provider_used', 'Unknown')}")
            return True
            
        else:
            print(f"❌ Enhanced endpoint failed - Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced endpoint error: {e}")
        return False

def test_legacy_endpoint(idea_data):
    """Test backwards compatibility with legacy endpoint"""
    print(f"\n🔄 Testing Legacy Compatibility: {idea_data['name']}")
    
    payload = {"text_idea": idea_data["text_idea"]}
    
    try:
        response = requests.post(LEGACY_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for error in response
            if data.get("error"):
                print(f"❌ Legacy endpoint returned error: {data['error']}")
                return False
            
            # Validate legacy response structure
            if data.get('initial_N') and data.get('initial_cohens_d'):
                print(f"✅ Legacy format working: N={data['initial_N']}, d={data['initial_cohens_d']}")
                print(f"✅ Provider Used: {data.get('llm_provider_used', 'Unknown')}")
                return True
            else:
                print(f"❌ Legacy endpoint missing required fields")
                print(f"Response: {json.dumps(data, indent=2)}")
                return False
                
        else:
            print(f"❌ Legacy endpoint failed - Status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Legacy endpoint error: {e}")
        return False

def main():
    """Main test runner"""
    print("🚀 Starting Phase 1.1 Enhanced LLM Integration Tests")
    print("=" * 60)
    
    # Test health check first
    if not test_health_check():
        print("❌ Health check failed - make sure the API server is running")
        sys.exit(1)
    
    # Test results tracking
    enhanced_results = []
    legacy_results = []
    
    # Test each research idea
    for idea in TEST_RESEARCH_IDEAS:
        print("\n" + "=" * 60)
        
        # Test enhanced endpoint
        enhanced_success = test_enhanced_endpoint(idea)
        enhanced_results.append(enhanced_success)
        
        time.sleep(1)  # Small delay between requests
        
        # Test legacy endpoint for backwards compatibility
        legacy_success = test_legacy_endpoint(idea)
        legacy_results.append(legacy_success)
        
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    enhanced_passed = sum(enhanced_results)
    legacy_passed = sum(legacy_results)
    total_tests = len(TEST_RESEARCH_IDEAS)
    
    print(f"Enhanced Endpoint: {enhanced_passed}/{total_tests} tests passed")
    print(f"Legacy Endpoint: {legacy_passed}/{total_tests} tests passed")
    
    if enhanced_passed == total_tests and legacy_passed == total_tests:
        print("✅ ALL TESTS PASSED - Phase 1.1 implementation successful!")
        return True
    else:
        print("❌ Some tests failed - check the logs above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)