#!/usr/bin/env python3
"""Test full API integration with field propagation"""

import requests
import json

def test_api_integration():
    """Test that fields propagate through the API correctly"""
    
    print("=" * 80)
    print("TESTING API INTEGRATION")
    print("=" * 80)
    
    # Test request
    payload = {
        "study_description": "hearing aid adoption rates",
        "include_research": True,
        "max_papers": 5,
        "pubmed_papers": 3,
        "clinicaltrials_papers": 2
    }
    
    print(f"\nRequest: {json.dumps(payload, indent=2)}")
    
    # Make API request
    response = requests.post(
        "http://localhost:8123/process_idea",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Check research papers data
        papers = data.get("research_papers_data", [])
        print(f"\nFound {len(papers)} papers in response")
        
        has_sample_method = 0
        has_pvalue = 0
        
        for i, paper in enumerate(papers, 1):
            print(f"\nPaper {i}: {paper.get('title', '')[:50]}...")
            print(f"  Sample Size: {paper.get('sample_size')}")
            print(f"  Sample Size Method: {paper.get('sample_size_method')}")
            print(f"  P-value: {paper.get('p_value')}")
            print(f"  Journal: {paper.get('journal', '')[:30]}...")
            
            if paper.get('sample_size_method'):
                has_sample_method += 1
                print(f"  ✅ Has sample_size_method: {paper['sample_size_method']}")
            
            if paper.get('p_value'):
                has_pvalue += 1
                print(f"  ✅ Has p_value: {paper['p_value']}")
        
        print(f"\nSummary:")
        print(f"  Papers with sample_size_method: {has_sample_method}/{len(papers)}")
        print(f"  Papers with p_value: {has_pvalue}/{len(papers)}")
        
        if has_sample_method == 0:
            print("  ❌ ERROR: No papers have sample_size_method field!")
        else:
            print("  ✅ Field propagation working!")
            
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api_integration()