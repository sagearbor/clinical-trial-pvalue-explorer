#!/usr/bin/env python3
"""
Test to verify that references are NOT hallucinated by LLM
and only come from actual web searches
"""

import requests
import json
import time

BASE_URL = "http://localhost:8123"

print("=" * 70)
print("TESTING: No Hallucinated References")
print("=" * 70)

test_cases = [
    {
        "description": "Antihypertensive medication study",
        "study": "Comparing ACE inhibitors vs ARBs for blood pressure control in elderly patients"
    },
    {
        "description": "Hearing aids effectiveness",
        "study": "Evaluating hearing aid effectiveness in elderly with presbycusis"
    },
    {
        "description": "Diabetes management",  
        "study": "Comparing metformin vs lifestyle intervention for type 2 diabetes prevention"
    }
]

print("\n1️⃣ Testing WITHOUT literature search (should have NO references)")
print("-" * 70)

for test in test_cases:
    print(f"\nTest: {test['description']}")
    
    # Test WITHOUT literature search
    payload = {
        "study_description": test['study'],
        "include_research": False,  # Explicitly disable research
    }
    
    response = requests.post(f"{BASE_URL}/process_idea", json=payload)
    if response.status_code == 200:
        data = response.json()
        refs = data.get('references', [])
        papers = data.get('research_papers_data', [])
        
        if refs or papers:
            print(f"   ❌ FAIL: Found {len(refs)} references when research was DISABLED!")
            if refs:
                print(f"      Hallucinated references: {refs[0][:100]}...")
        else:
            print(f"   ✅ PASS: No references (as expected)")
            
        # Check for specific hallucinated reference
        if refs:
            refs_str = str(refs)
            if "Visceral Fat" in refs_str:
                print(f"   ❌ CRITICAL: Found 'Visceral Fat' hallucination!")
    else:
        print(f"   ❌ API Error: {response.status_code}")
    
    time.sleep(0.5)

print("\n" + "=" * 70)
print("2️⃣ Testing WITH literature search (should have REAL references)")
print("-" * 70)

# Test one case WITH literature search enabled
test = test_cases[0]
print(f"\nTest: {test['description']} WITH research enabled")

payload = {
    "study_description": test['study'],
    "include_research": True,  # Enable research
    "max_papers": 3,
    "pubmed_papers": 2,
    "clinicaltrials_papers": 1
}

response = requests.post(f"{BASE_URL}/process_idea", json=payload)
if response.status_code == 200:
    data = response.json()
    refs = data.get('references', [])
    papers = data.get('research_papers_data', [])
    debug = data.get('research_debug', {})
    
    if refs or papers:
        print(f"   ✅ Found {len(refs)} references from web searches")
        print(f"   📊 Sources: PubMed={debug.get('pubmed',{}).get('count',0)}, " +
              f"ClinicalTrials={debug.get('clinicaltrials',{}).get('count',0)}, " +
              f"arXiv={debug.get('arxiv',{}).get('count',0)}")
        
        # Check that references have proper structure (URLs, journals, etc)
        if papers:
            p = papers[0]
            has_url = bool(p.get('url'))
            has_journal = bool(p.get('journal'))
            has_year = bool(p.get('year'))
            
            if has_url and has_journal and has_year:
                print(f"   ✅ References have proper structure (URLs, journals, years)")
            else:
                print(f"   ⚠️  References missing data: url={has_url}, journal={has_journal}, year={has_year}")
                
        # Check for hallucinated content
        refs_str = str(refs) + str(papers)
        if "Visceral Fat and Cardiometabolic Risk" in refs_str:
            print(f"   ❌ WARNING: Still found hallucinated 'Visceral Fat' reference!")
    else:
        print(f"   ⚠️  No references found (web search may have failed)")
else:
    print(f"   ❌ API Error: {response.status_code}")

print("\n" + "=" * 70)
print("📊 SUMMARY")
print("=" * 70)
print("""
✅ FIX APPLIED:
1. Removed reference extraction from LLM responses
2. References now ONLY come from research_intelligence.py web searches
3. LLM prompts never ask for references
4. validate_and_extract_enhanced_response() no longer extracts references

🔍 TO VERIFY:
1. Run this test - should show NO references when research is disabled
2. Check API logs for debug messages about literature search
3. When research IS enabled, references should have real URLs and journals

⚠️ IMPORTANT:
- The LLM should NEVER generate references
- All references must come from actual PubMed/ClinicalTrials/arXiv searches
- If you see "Visceral Fat and Cardiometabolic Risk" anywhere, it's hallucinated!
""")