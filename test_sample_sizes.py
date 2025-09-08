#!/usr/bin/env python3
"""
Test that sample sizes (N) are properly extracted from all sources
"""

import requests
import json

BASE_URL = "http://localhost:8123"

print("=" * 70)
print("TESTING: Sample Size Extraction & ClinicalTrials.gov")
print("=" * 70)

# Test the hearing query
query = "How to help older adults increase uptake of hearing devices to increase quality of life"

payload = {
    "study_description": query,
    "include_research": True,
    "max_papers": 9,
    "pubmed_papers": 3,
    "clinicaltrials_papers": 4,  # Prioritize CT.gov since they have enrollment data
    "arxiv_papers": 2
}

print(f"\nQuery: {query[:60]}...")
print(f"Requesting: {payload['pubmed_papers']} PubMed, {payload['clinicaltrials_papers']} CT.gov, {payload['arxiv_papers']} arXiv")
print("-" * 70)

response = requests.post(f"{BASE_URL}/process_idea", json=payload)

if response.status_code == 200:
    data = response.json()
    papers = data.get('research_papers_data', [])
    debug = data.get('research_debug', {})
    
    print(f"\n📊 Results Summary:")
    print(f"   PubMed: {debug.get('pubmed', {}).get('count', 0)} papers")
    print(f"   ClinicalTrials: {debug.get('clinicaltrials', {}).get('count', 0)} papers")
    print(f"   arXiv: {debug.get('arxiv', {}).get('count', 0)} papers")
    print(f"   Total: {len(papers)} papers")
    
    if papers:
        print(f"\n📚 Papers with Sample Sizes:")
        print("-" * 70)
        
        # Track which sources provide N
        papers_with_n = 0
        by_source = {'PubMed': [], 'ClinicalTrials.gov': [], 'arXiv': []}
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'No title')
            journal = paper.get('journal', 'Unknown')
            year = paper.get('year', 'Unknown')
            url = paper.get('url', 'No URL')
            sample_size = paper.get('sample_size')
            
            print(f"\n{i}. {title[:60]}...")
            print(f"   Source: {journal}, Year: {year}")
            
            if sample_size:
                print(f"   ✅ N = {sample_size}")
                papers_with_n += 1
                
                # Track by source
                if 'ClinicalTrials' in journal:
                    by_source['ClinicalTrials.gov'].append(sample_size)
                elif 'PubMed' in journal:
                    by_source['PubMed'].append(sample_size)
                elif 'arXiv' in journal:
                    by_source['arXiv'].append(sample_size)
            else:
                print(f"   ❌ No sample size extracted")
            
            # Check if it's a hearing-related paper
            title_lower = title.lower()
            if any(term in title_lower for term in ['hearing', 'deaf', 'auditory', 'cochlear']):
                print(f"   🎯 HEARING-RELATED")
                
        print(f"\n" + "=" * 70)
        print(f"📊 SAMPLE SIZE ANALYSIS:")
        print(f"   Papers with N: {papers_with_n}/{len(papers)} ({papers_with_n/len(papers)*100:.0f}%)")
        print(f"\n   By Source:")
        print(f"   - ClinicalTrials.gov: {len(by_source['ClinicalTrials.gov'])} papers with N")
        if by_source['ClinicalTrials.gov']:
            print(f"     Sample sizes: {by_source['ClinicalTrials.gov']}")
        print(f"   - PubMed: {len(by_source['PubMed'])} papers with N")
        if by_source['PubMed']:
            print(f"     Sample sizes: {by_source['PubMed']}")
        print(f"   - arXiv: {len(by_source['arXiv'])} papers with N")
        if by_source['arXiv']:
            print(f"     Sample sizes: {by_source['arXiv']}")
            
    else:
        print("\n⚠️ No papers returned")
else:
    print(f"\n❌ API Error: {response.status_code}")

print(f"\n" + "=" * 70)
print("EXPECTED BEHAVIOR:")
print("=" * 70)
print("""
✅ ClinicalTrials.gov papers should:
   - Return hearing-related trials
   - ALL should have enrollment numbers (N)

✅ PubMed papers should:
   - Extract N from abstracts when mentioned
   - Common patterns: "N=123", "123 patients", "enrolled 456"

⚠️ arXiv papers:
   - May not always have N (preprints often lack final numbers)
""")