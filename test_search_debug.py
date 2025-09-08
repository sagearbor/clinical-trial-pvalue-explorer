#!/usr/bin/env python3
"""
Debug script to see exactly what papers are being returned by searches
"""

import requests
import json

BASE_URL = "http://localhost:8123"

print("=" * 70)
print("DEBUGGING: What papers are actually returned by searches?")
print("=" * 70)

# Different search queries
queries = [
    "Antihypertensive medication ACE inhibitors vs ARBs blood pressure",
    "Hearing aids effectiveness elderly presbycusis",
    "Diabetes metformin lifestyle intervention prevention"
]

for i, query in enumerate(queries, 1):
    print(f"\n{i}. Query: {query[:60]}...")
    print("-" * 70)
    
    payload = {
        "study_description": query,
        "include_research": True,
        "max_papers": 3,
        "pubmed_papers": 2,
        "clinicaltrials_papers": 1
    }
    
    response = requests.post(f"{BASE_URL}/process_idea", json=payload)
    if response.status_code == 200:
        data = response.json()
        papers = data.get('research_papers_data', [])
        
        if papers:
            print(f"Found {len(papers)} papers:")
            for j, paper in enumerate(papers, 1):
                title = paper.get('title', 'No title')
                journal = paper.get('journal', 'Unknown')
                year = paper.get('year', 'Unknown')
                url = paper.get('url', 'No URL')
                
                print(f"\n   Paper {j}:")
                print(f"   Title: {title[:80]}...")
                print(f"   Journal: {journal}")
                print(f"   Year: {year}")
                print(f"   URL: {url[:80] if url != 'No URL' else 'No URL'}")
                
                # Check for the suspicious "Visceral Fat" paper
                if "Visceral Fat" in title:
                    print(f"   ⚠️ WARNING: Found 'Visceral Fat' paper!")
        else:
            print("No papers returned")
    else:
        print(f"API Error: {response.status_code}")

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("""
If you see the SAME paper titles across DIFFERENT queries, then:
1. The search engine might be caching results incorrectly
2. The search queries might be too broad
3. There might be a fallback returning default papers

If you see DIFFERENT papers for each query with relevant titles, then:
✅ The search is working correctly
✅ Papers are real from PubMed/ClinicalTrials
✅ No hallucination is occurring
""")