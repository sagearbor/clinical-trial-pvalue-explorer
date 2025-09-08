#!/usr/bin/env python3
"""
Test that hearing-related queries return relevant papers
"""

import requests
import json

BASE_URL = "http://localhost:8123"

print("=" * 70)
print("TESTING: Hearing Device Search Relevance")
print("=" * 70)

# Test the exact query the user mentioned
query = "How to help older adults increase uptake of hearing devices to increase quality of life"

payload = {
    "study_description": query,
    "include_research": True,
    "max_papers": 9,  # Total requested
    "pubmed_papers": 4,
    "clinicaltrials_papers": 3,
    "arxiv_papers": 2
}

print(f"\nQuery: {query}")
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
        print(f"\n📚 Papers Found:")
        print("-" * 70)
        
        # Track relevance
        hearing_related = 0
        irrelevant_papers = []
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'No title')
            journal = paper.get('journal', 'Unknown')
            year = paper.get('year', 'Unknown')
            url = paper.get('url', 'No URL')
            
            print(f"\n{i}. {title[:80]}...")
            print(f"   Source: {journal}, Year: {year}")
            if url != 'No URL':
                print(f"   URL: {url[:80]}...")
            
            # Check relevance
            title_lower = title.lower()
            relevant_terms = ['hearing', 'deaf', 'auditory', 'cochlear', 'presbycusis', 
                            'hearing aid', 'audiolog', 'sound', 'ear']
            
            if any(term in title_lower for term in relevant_terms):
                print(f"   ✅ RELEVANT: Contains hearing-related terms")
                hearing_related += 1
            else:
                print(f"   ❌ IRRELEVANT: No hearing-related terms")
                irrelevant_papers.append(title)
                
                # Check for known problematic papers
                if "Visceral Fat" in title:
                    print(f"   🚨 CRITICAL: Found 'Visceral Fat' paper!")
                elif "Mothers and CareGivers" in title:
                    print(f"   🚨 CRITICAL: Found 'Mothers and CareGivers' paper!")
                elif "Success in Health" in title:
                    print(f"   🚨 CRITICAL: Found 'Success in Health' paper!")
        
        print(f"\n" + "=" * 70)
        print(f"📊 RELEVANCE ANALYSIS:")
        print(f"   Relevant papers: {hearing_related}/{len(papers)} ({hearing_related/len(papers)*100:.0f}%)")
        print(f"   Irrelevant papers: {len(irrelevant_papers)}/{len(papers)}")
        
        if irrelevant_papers:
            print(f"\n❌ Irrelevant papers that shouldn't appear:")
            for title in irrelevant_papers[:5]:  # Show first 5
                print(f"   - {title[:60]}...")
                
    else:
        print("\n⚠️ No papers returned")
else:
    print(f"\n❌ API Error: {response.status_code}")

print(f"\n" + "=" * 70)
print("EXPECTED BEHAVIOR:")
print("=" * 70)
print("""
✅ Papers should be about:
   - Hearing aids/devices
   - Hearing loss in elderly
   - Presbycusis
   - Auditory rehabilitation
   - Quality of life with hearing impairment

❌ Papers should NOT be about:
   - Visceral fat
   - Obesity in children
   - General caregiving (unless hearing-related)
   - Unrelated health topics
""")