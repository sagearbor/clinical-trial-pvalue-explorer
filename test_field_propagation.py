#!/usr/bin/env python3
"""Test to verify sample_size_method and p_value propagation"""

import asyncio
import json
from src.research_intelligence import ResearchIntelligenceEngine

async def test_field_propagation():
    """Test that sample_size_method and p_value fields propagate correctly"""
    engine = ResearchIntelligenceEngine()
    
    # Search for a known paper with sample size
    summary = await engine.analyze_research_topic(
        idea="hearing aid adoption rates",
        max_papers=3,
        pubmed_papers=3,
        arxiv_papers=0,
        clinicaltrials_papers=0
    )
    
    print("=" * 80)
    print("TESTING FIELD PROPAGATION")
    print("=" * 80)
    
    if summary and summary.papers_analyzed:
        for i, paper in enumerate(summary.papers_analyzed, 1):
            print(f"\nPaper {i}: {paper.title[:60]}...")
            print(f"  PMID: {paper.pmid}")
            print(f"  Sample Size: {paper.sample_size}")
            print(f"  Sample Size Method: {paper.sample_size_method}")  # Should not be None
            print(f"  P-value: {paper.p_value}")  # Should not be None if found
            
            # Check the raw object attributes
            print(f"  Has sample_size_method attr: {hasattr(paper, 'sample_size_method')}")
            print(f"  Has p_value attr: {hasattr(paper, 'p_value')}")
            
            # Convert to dict as API does
            paper_dict = {
                "title": paper.title,
                "authors": paper.authors,
                "year": paper.year,
                "journal": paper.journal,
                "sample_size": paper.sample_size,
                "sample_size_method": paper.sample_size_method,
                "p_value": paper.p_value,
                "study_signal": paper.study_signal,
                "url": paper.url,
                "extras": paper.extras if hasattr(paper, 'extras') else None,
            }
            
            print(f"\n  Dict conversion:")
            print(f"    sample_size_method in dict: {paper_dict.get('sample_size_method')}")
            print(f"    p_value in dict: {paper_dict.get('p_value')}")
    else:
        print("No papers found!")

if __name__ == "__main__":
    asyncio.run(test_field_propagation())