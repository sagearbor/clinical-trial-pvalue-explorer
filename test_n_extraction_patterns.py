#!/usr/bin/env python3
"""Find papers with N in abstract but not extracted, to improve patterns"""

import asyncio
import re
from src.research_intelligence import ResearchIntelligenceEngine, _extract_sample_size

async def find_missed_n_patterns():
    """Search for papers and identify missed N patterns"""
    engine = ResearchIntelligenceEngine()
    
    # Get a larger sample of papers
    summary = await engine.analyze_research_topic(
        idea="randomized controlled trial",
        max_papers=20,
        pubmed_papers=20,
        arxiv_papers=0,
        clinicaltrials_papers=0
    )
    
    print("=" * 80)
    print("ANALYZING N EXTRACTION PATTERNS")
    print("=" * 80)
    
    missed_patterns = []
    extracted_count = 0
    
    if summary and summary.papers_analyzed:
        for paper in summary.papers_analyzed:
            abstract = paper.abstract
            extracted_n = paper.sample_size
            
            # Look for potential N patterns in abstract
            potential_n_patterns = [
                r"[Nn]\s*=\s*[\d,]+",
                r"[\d,]+\s*(?:patients|participants|subjects|adults|children|infants)",
                r"(?:enrolled|included|recruited|randomized|analyzed)\s*[\d,]+",
                r"[\d,]+\s*(?:were|was)\s*(?:enrolled|included|recruited|randomized|analyzed)",
                r"sample\s*size.*?[\d,]+",
                r"total\s*of\s*[\d,]+",
                r"[\d,]+\s*(?:men|women|males|females)",
                r"cohort\s*of\s*[\d,]+",
                r"data\s*from\s*[\d,]+",
                r"[\d,]+\s*completed\s*the\s*study",
            ]
            
            found_in_text = []
            for pattern in potential_n_patterns:
                matches = re.findall(pattern, abstract, re.IGNORECASE)
                if matches:
                    found_in_text.extend(matches)
            
            if found_in_text and not extracted_n:
                # Found N in text but didn't extract it
                print(f"\n❌ MISSED: {paper.title[:50]}...")
                print(f"   PMID: {paper.pmid}")
                print(f"   Found in text: {found_in_text[:3]}")  # Show first 3 matches
                print(f"   Abstract excerpt:")
                
                # Show the context around the first match
                first_match = found_in_text[0]
                idx = abstract.lower().find(first_match.lower())
                if idx > -1:
                    start = max(0, idx - 50)
                    end = min(len(abstract), idx + len(first_match) + 50)
                    print(f"   ...{abstract[start:end]}...")
                
                missed_patterns.append({
                    'pmid': paper.pmid,
                    'title': paper.title,
                    'patterns': found_in_text,
                    'abstract_excerpt': abstract[start:end] if idx > -1 else None
                })
            elif extracted_n:
                extracted_count += 1
                print(f"✅ Extracted N={extracted_n}: {paper.title[:50]}...")
    
    print(f"\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Papers with extracted N: {extracted_count}/{len(summary.papers_analyzed)}")
    print(f"  Papers with missed N: {len(missed_patterns)}")
    
    if missed_patterns:
        print(f"\nMOST COMMON MISSED PATTERNS:")
        pattern_counts = {}
        for mp in missed_patterns:
            for p in mp['patterns']:
                # Generalize the pattern
                generalized = re.sub(r'[\d,]+', 'N', p)
                pattern_counts[generalized] = pattern_counts.get(generalized, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {count}x: {pattern}")

if __name__ == "__main__":
    asyncio.run(find_missed_n_patterns())