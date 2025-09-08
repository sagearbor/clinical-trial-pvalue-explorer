#!/usr/bin/env python3
"""Test if PMID 40913255 now extracts N correctly"""

import requests
import json

# Test with the specific query that should return PMID 40913255
response = requests.post("http://localhost:8123/process_idea", json={
    "study_description": "hearing aid adoption rates",
    "include_research": True,
    "max_papers": 5,
    "pubmed_papers": 5
})

if response.status_code == 200:
    data = response.json()
    papers = data.get('research_papers_data', [])
    
    print(f"Found {len(papers)} papers\n")
    
    # Look for the specific paper
    found_40913255 = False
    for p in papers:
        if "Hearing Aid Adoption Rates" in p.get('title', ''):
            found_40913255 = True
            print(f"✅ Found PMID 40913255:")
            print(f"   Title: {p.get('title')[:60]}...")
            print(f"   Journal: {p.get('journal')}")
            print(f"   N: {p.get('sample_size')}")
            print(f"   Method: {p.get('sample_size_method')}")
            print(f"   p-value: {p.get('p_value')}")
            
            if p.get('sample_size') == 321:
                print("   ✅✅ Successfully extracted N=321!")
            else:
                print(f"   ❌ Expected N=321, got {p.get('sample_size')}")
                
    if not found_40913255:
        print("❌ PMID 40913255 not in results")
        print("\nPapers returned:")
        for p in papers:
            print(f"  - {p.get('title', 'No title')[:50]}...")
else:
    print(f"Error: {response.status_code}")
