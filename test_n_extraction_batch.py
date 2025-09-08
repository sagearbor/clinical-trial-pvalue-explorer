#!/usr/bin/env python3
"""Batch N extraction tester - processes papers in manageable batches"""

import asyncio
import csv
import re
import json
from datetime import datetime
from typing import List, Dict, Any
from src.research_intelligence import ResearchIntelligenceEngine, _extract_sample_size, _extract_p_value

class BatchNExtractionTester:
    def __init__(self):
        self.engine = ResearchIntelligenceEngine()
        self.log_file = "n_extraction_batch_log.csv"
        self.detailed_log = "n_extraction_detailed.txt"
        
    def init_log(self):
        """Initialize CSV log file"""
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Run', 'PMID/ID', 'Source', 'Title', 
                'API_sample_size', 'API_sample_size_method', 'API_p_value',
                'Current_Extraction', 'Numbers_in_Abstract', 'Abstract_Excerpt',
                'Issue', 'Pattern_Match'
            ])
    
    def log_entry(self, run: int, paper: Dict, current_extraction: Any, 
                  numbers_found: List, excerpt: str, issue: str, pattern: str):
        """Add entry to CSV log"""
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                run,
                paper.get('pmid') or paper.get('id', '')[:20],
                paper.get('source', 'Unknown'),
                paper.get('title', '')[:60],
                paper.get('sample_size'),
                paper.get('sample_size_method'),
                paper.get('p_value'),
                current_extraction,
                ','.join(map(str, numbers_found[:3])),
                excerpt[:100],
                issue,
                pattern
            ])
    
    def find_numbers_in_text(self, text: str) -> List[Dict]:
        """Find all potential sample sizes in text"""
        patterns = [
            ('n_equals', r'\b[Nn]\s*=\s*([\d,]+)'),
            ('parentheses_n', r'\([Nn]\s*=\s*([\d,]+)\)'),
            ('num_patients', r'([\d,]+)\s+patients?\b'),
            ('num_participants', r'([\d,]+)\s+participants?\b'),
            ('num_subjects', r'([\d,]+)\s+subjects?\b'),
            ('were_enrolled', r'([\d,]+)\s+were\s+enrolled'),
            ('were_included', r'([\d,]+)\s+were\s+included'),
            ('were_randomized', r'([\d,]+)\s+were\s+randomized'),
            ('total_of', r'total\s+of\s+([\d,]+)'),
            ('enrolled_num', r'enrolled\s+([\d,]+)'),
            ('included_num', r'included\s+([\d,]+)'),
            ('randomized_num', r'randomized\s+([\d,]+)'),
            ('analyzed_num', r'analyzed\s+([\d,]+)'),
            ('completed_num', r'([\d,]+)\s+completed'),
            ('sample_of', r'sample\s+of\s+([\d,]+)'),
            ('cohort_of', r'cohort\s+of\s+([\d,]+)'),
            ('num_men_women', r'([\d,]+)\s+(?:men|women)'),
            ('num_children', r'([\d,]+)\s+children'),
            ('aged_num', r'aged\s+([\d,]+)'),
            ('mean_age', r'mean\s+age[^,\.]*?([\d,]+)'),
        ]
        
        found = []
        for pattern_name, pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    num = int(match.replace(',', ''))
                    if 10 <= num <= 100000:  # Reasonable range
                        # Find context
                        idx = text.lower().find(match.lower())
                        if idx > -1:
                            start = max(0, idx - 20)
                            end = min(len(text), idx + len(match) + 20)
                            context = text[start:end]
                            found.append({
                                'number': num,
                                'pattern': pattern_name,
                                'context': context,
                                'match': match
                            })
                except:
                    pass
        
        return found
    
    async def fetch_batch(self, query: str, source: str, count: int) -> List[Dict]:
        """Fetch a batch of papers from specified source"""
        config = {
            'pubmed': {"pubmed_papers": count, "arxiv_papers": 0, "clinicaltrials_papers": 0},
            'arxiv': {"pubmed_papers": 0, "arxiv_papers": count, "clinicaltrials_papers": 0},
            'clinicaltrials': {"pubmed_papers": 0, "arxiv_papers": 0, "clinicaltrials_papers": count}
        }
        
        papers = []
        try:
            summary = await self.engine.analyze_research_topic(
                idea=query,
                max_papers=count,
                **config[source]
            )
            
            if summary and summary.papers_analyzed:
                for p in summary.papers_analyzed:
                    papers.append({
                        'pmid': p.pmid,
                        'id': p.pmid or p.arxiv_id or p.title[:30],
                        'title': p.title,
                        'abstract': p.abstract,
                        'journal': p.journal,
                        'sample_size': p.sample_size,
                        'sample_size_method': p.sample_size_method,
                        'p_value': p.p_value,
                        'source': source.title(),
                        'url': p.url
                    })
        except Exception as e:
            print(f"Error fetching {source}: {e}")
        
        return papers
    
    async def run_batch_test(self):
        """Run the batch test"""
        self.init_log()
        
        queries = [
            "randomized controlled trial",
            "clinical trial",
            "cohort study",
            "case control study",
            "systematic review"
        ]
        
        sources = ['pubmed', 'clinicaltrials', 'arxiv']
        batch_size = 10
        
        print("=" * 80)
        print("BATCH N EXTRACTION TEST")
        print("=" * 80)
        
        with open(self.detailed_log, 'w') as detail_f:
            detail_f.write("N EXTRACTION DETAILED ANALYSIS\n")
            detail_f.write("=" * 80 + "\n\n")
            
            for run in range(1, 6):  # 5 runs with improvements
                print(f"\n{'='*80}")
                print(f"RUN {run}")
                print(f"{'='*80}")
                
                detail_f.write(f"\nRUN {run}\n")
                detail_f.write("-" * 40 + "\n")
                
                papers_without_n = []
                papers_with_n = []
                total_papers = 0
                
                # Fetch papers from each source
                for source in sources:
                    query = queries[(run - 1) % len(queries)]
                    print(f"\nFetching {batch_size} papers from {source} (query: {query})...")
                    
                    papers = await self.fetch_batch(query, source, batch_size)
                    total_papers += len(papers)
                    
                    for paper in papers:
                        # Test current extraction
                        current_n = _extract_sample_size(paper['abstract'])
                        current_p = _extract_p_value(paper['abstract'])
                        
                        # Find all numbers in abstract
                        numbers = self.find_numbers_in_text(paper['abstract'])
                        
                        # Determine issue
                        issue = ""
                        if paper['sample_size'] and not paper['sample_size_method']:
                            issue = "SIZE_WITHOUT_METHOD"
                        elif paper['sample_size_method'] and not paper['sample_size']:
                            issue = "METHOD_WITHOUT_SIZE"
                        elif not paper['sample_size'] and numbers:
                            issue = "MISSED_EXTRACTION"
                        elif not paper['sample_size'] and not numbers:
                            issue = "NO_N_IN_TEXT"
                        
                        # Get excerpt
                        excerpt = ""
                        pattern_used = ""
                        if numbers:
                            excerpt = numbers[0]['context']
                            pattern_used = numbers[0]['pattern']
                        
                        # Log to CSV
                        self.log_entry(
                            run, paper, current_n,
                            [n['number'] for n in numbers],
                            excerpt, issue, pattern_used
                        )
                        
                        # Track for analysis
                        if paper['sample_size']:
                            papers_with_n.append(paper)
                        elif numbers:
                            papers_without_n.append((paper, numbers))
                        
                        # Detailed logging for missed extractions
                        if issue == "MISSED_EXTRACTION":
                            detail_f.write(f"\nMISSED: {paper['title'][:60]}\n")
                            detail_f.write(f"  Source: {paper['source']}\n")
                            detail_f.write(f"  PMID: {paper.get('pmid', 'N/A')}\n")
                            detail_f.write(f"  Numbers found: {[n['number'] for n in numbers[:3]]}\n")
                            detail_f.write(f"  Patterns: {[n['pattern'] for n in numbers[:3]]}\n")
                            detail_f.write(f"  Context: {numbers[0]['context'] if numbers else 'N/A'}\n")
                
                # Summary for this run
                print(f"\n📊 Run {run} Summary:")
                print(f"  Total papers: {total_papers}")
                print(f"  With N extracted: {len(papers_with_n)} ({100*len(papers_with_n)/max(1,total_papers):.1f}%)")
                print(f"  Missed (has numbers): {len(papers_without_n)}")
                
                detail_f.write(f"\n\nRun {run} Statistics:\n")
                detail_f.write(f"  Extraction rate: {len(papers_with_n)}/{total_papers}\n")
                
                # Analyze missed patterns
                if papers_without_n:
                    print(f"\n🔍 Analyzing missed patterns...")
                    pattern_counts = {}
                    for paper, numbers in papers_without_n[:5]:  # Analyze first 5
                        for num_info in numbers:
                            pattern = num_info['pattern']
                            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                    
                    print(f"  Most common missed patterns:")
                    for pattern, count in sorted(pattern_counts.items(), 
                                                key=lambda x: x[1], reverse=True)[:3]:
                        print(f"    {pattern}: {count}x")
                    
                    # Make improvements for next run
                    await self.improve_patterns(run, pattern_counts, detail_f)
                
                await asyncio.sleep(1)  # Rate limiting
        
        print(f"\n✅ Test complete!")
        print(f"📊 Results saved to: {self.log_file}")
        print(f"📝 Detailed log: {self.detailed_log}")
    
    async def improve_patterns(self, run: int, pattern_counts: Dict, detail_f):
        """Improve extraction patterns based on findings"""
        print(f"\n🔧 Improving patterns for Run {run + 1}...")
        
        improvements = []
        
        # Determine what patterns to add based on frequency
        if pattern_counts.get('num_men_women', 0) >= 2:
            improvements.append("Strengthen gender-specific patterns")
        
        if pattern_counts.get('analyzed_num', 0) >= 2:
            improvements.append("Add 'data analyzed from X patients' pattern")
        
        if pattern_counts.get('aged_num', 0) >= 1:
            improvements.append("Improve age-related number extraction")
        
        detail_f.write(f"\nSuggested improvements for Run {run + 1}:\n")
        for imp in improvements:
            detail_f.write(f"  - {imp}\n")
            print(f"    - {imp}")
        
        # Note: In a real scenario, we would modify the patterns in
        # research_intelligence.py here

async def main():
    tester = BatchNExtractionTester()
    await tester.run_batch_test()

if __name__ == "__main__":
    asyncio.run(main())