#!/usr/bin/env python3
"""Test what abstract text we're actually getting from PubMed"""

import sys
sys.path.insert(0, '/dcri/sasusers/home/scb2/gitRepos/clinical-trial-pvalue-explorer/src')

import asyncio
from research_intelligence import PubMedSearcher

async def test():
    searcher = PubMedSearcher()
    results = await searcher.search("hearing aids elderly", max_results=2)
    
    for i, paper in enumerate(results, 1):
        print(f"\n{'='*70}")
        print(f"Paper {i}: {paper.get('title', 'No title')[:60]}...")
        print(f"PMID: {paper.get('pmid')}")
        print(f"Abstract length: {len(paper.get('abstract', ''))}")
        abstract = paper.get('abstract', '')
        
        # Show first 500 chars
        print(f"Abstract preview: {abstract[:500]}...")
        
        # Check for N patterns
        if 'n=' in abstract.lower() or 'n =' in abstract.lower():
            print("✅ Contains 'n=' pattern")
        else:
            print("❌ No 'n=' pattern found")
            
        # Check what was extracted
        print(f"Extracted N: {paper.get('sample_size')}")
        print(f"Extraction method: {paper.get('sample_size_method')}")
        print(f"Extracted p-value: {paper.get('p_value')}")

asyncio.run(test())
