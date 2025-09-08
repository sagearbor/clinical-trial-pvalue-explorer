#!/usr/bin/env python3
"""Test specific PMID 40913255 extraction"""

import asyncio
from src.research_intelligence import PubMedSearcher

async def test_pmid_40913255():
    """Test extraction for PMID 40913255"""
    searcher = PubMedSearcher()
    
    # Fetch details for specific PMID
    papers = await searcher._fetch_details(["40913255"])
    
    print("=" * 80)
    print("TESTING PMID 40913255")
    print("=" * 80)
    
    if papers:
        paper = papers[0]
        print(f"Title: {paper.get('title', '')[:60]}...")
        print(f"Journal: {paper.get('journal', '')}")
        print(f"Sample Size: {paper.get('sample_size', 'None')}")
        print(f"Sample Size Method: {paper.get('sample_size_method', 'None')}")
        print(f"P-value: {paper.get('p_value', 'None')}")
        
        # Print part of abstract to verify
        abstract = paper.get('abstract', '')
        if abstract:
            # Look for the critical part
            idx = abstract.find("Patients were given")
            if idx > -1:
                print(f"\nAbstract excerpt:\n{abstract[idx:idx+200]}...")
            
            idx2 = abstract.find("Our sample")
            if idx2 > -1:
                print(f"\nStudy sample excerpt:\n{abstract[idx2:idx2+100]}...")
                
            # Look for p-values
            if "p <" in abstract or "p =" in abstract or "P <" in abstract:
                print("\nP-value mentions found in abstract")
                for i, line in enumerate(abstract.split('.')):
                    if 'p <' in line.lower() or 'p =' in line.lower():
                        print(f"  Line {i}: {line.strip()}")
    else:
        print("No paper found!")

if __name__ == "__main__":
    asyncio.run(test_pmid_40913255())