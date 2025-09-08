#!/usr/bin/env python3
"""Test ClinicalTrials.gov structured data extraction"""

import asyncio
from src.research_intelligence import ResearchIntelligenceEngine

async def test_clinicaltrials():
    """Test that ClinicalTrials.gov papers have structured sample sizes"""
    engine = ResearchIntelligenceEngine()
    
    # Search for hearing aids in ClinicalTrials.gov
    summary = await engine.analyze_research_topic(
        idea="hearing aids elderly",
        max_papers=5,
        pubmed_papers=0,
        arxiv_papers=0,
        clinicaltrials_papers=5
    )
    
    print("=" * 80)
    print("TESTING CLINICALTRIALS.GOV STRUCTURED DATA")
    print("=" * 80)
    
    if summary and summary.papers_analyzed:
        structured_count = 0
        for i, paper in enumerate(summary.papers_analyzed, 1):
            print(f"\nPaper {i}: {paper.title[:60]}...")
            print(f"  NCT ID: {paper.pmid or 'N/A'}")
            print(f"  Sample Size: {paper.sample_size}")
            print(f"  Sample Size Method: {paper.sample_size_method}")
            print(f"  Journal: {paper.journal}")
            
            if paper.sample_size_method == "structured":
                structured_count += 1
                print("  ✅ Has structured sample size")
            elif paper.sample_size_method == "text_extraction":
                print("  ⚠️  Text extraction (should be structured)")
            else:
                print("  ❌ No sample size method")
        
        print(f"\n{structured_count}/{len(summary.papers_analyzed)} papers have structured sample sizes")
        
        if structured_count == 0:
            print("⚠️  WARNING: No structured sample sizes found. Check CT.gov API parsing.")
    else:
        print("No ClinicalTrials.gov papers found!")

if __name__ == "__main__":
    asyncio.run(test_clinicaltrials())