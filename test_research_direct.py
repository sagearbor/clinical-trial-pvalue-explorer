#!/usr/bin/env python3
"""Test research engine directly"""

import sys
sys.path.insert(0, '/dcri/sasusers/home/scb2/gitRepos/clinical-trial-pvalue-explorer/src')

import asyncio
from research_intelligence import ResearchIntelligenceEngine

async def test():
    engine = ResearchIntelligenceEngine()
    summary = await engine.search("hearing aids elderly", max_results=3, clinicaltrials_papers=2, pubmed_papers=1)
    
    print(f"Found {len(summary.papers_analyzed)} papers:")
    for p in summary.papers_analyzed:
        print(f"\n- {p.title[:50]}...")
        print(f"  sample_size: {p.sample_size}")
        print(f"  sample_size_method: {p.sample_size_method}")
        print(f"  p_value: {p.p_value}")
        print(f"  journal: {p.journal}")

asyncio.run(test())
