#!/usr/bin/env python3
"""Test the full pipeline with a query that should return papers with N"""

import requests
import json

# Use a query that should return older papers with sample sizes
response = requests.post("http://localhost:8123/process_idea", json={
    "study_description": "randomized controlled trial hearing aids elderly 2020-2023",
    "include_research": True,
    "max_papers": 6,
    "pubmed_papers": 3,
    "clinicaltrials_papers": 3
})

if response.status_code == 200:
    data = response.json()
    papers = data.get('research_papers_data', [])
    
    print(f"="*70)
    print(f"FULL PIPELINE TEST RESULTS")
    print(f"="*70)
    print(f"Found {len(papers)} papers\n")
    
    papers_with_n = 0
    papers_with_p = 0
    
    for i, p in enumerate(papers, 1):
        title = p.get('title', 'No title')[:50]
        n = p.get('sample_size')
        n_method = p.get('sample_size_method')
        p_value = p.get('p_value')
        journal = p.get('journal', 'Unknown')
        
        print(f"{i}. {title}...")
        print(f"   Source: {journal}")
        
        if n:
            papers_with_n += 1
            indicator = "¹" if n_method == "text_extraction" else ""
            print(f"   ✅ N = {n}{indicator} (method: {n_method})")
        else:
            print(f"   ❌ N = None")
            
        if p_value:
            papers_with_p += 1
            print(f"   ✅ p = {p_value}")
        else:
            print(f"   ❌ p = None")
            
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Papers with N: {papers_with_n}/{len(papers)} ({papers_with_n/len(papers)*100:.0f}%)")
    print(f"  Papers with p-values: {papers_with_p}/{len(papers)} ({papers_with_p/len(papers)*100:.0f}%)")
    
    # Check if frontend would show indicators
    has_text_extracted = any(p.get('sample_size_method') == 'text_extraction' for p in papers)
    if has_text_extracted:
        print(f"\n✅ Frontend should show: '¹ Sample size extracted from abstract text'")
else:
    print(f"Error: {response.status_code}")
