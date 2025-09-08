#!/usr/bin/env python3
"""Comprehensive N extraction test harness - tests 100+ articles from each source"""

import asyncio
import csv
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.research_intelligence import ResearchIntelligenceEngine, _extract_sample_size

class ComprehensiveNExtractionTester:
    def __init__(self):
        self.engine = ResearchIntelligenceEngine()
        self.log_file = "n_extraction_log.csv"
        self.run_number = 1
        self.papers_cache = []
        
    async def fetch_papers(self, queries: List[str], papers_per_query: int = 25) -> List[Dict]:
        """Fetch papers from all three sources using multiple queries"""
        all_papers = []
        
        for query in queries:
            print(f"\n📚 Fetching papers for query: '{query}'")
            
            # Fetch from each source
            for source_config in [
                {"pubmed_papers": papers_per_query, "arxiv_papers": 0, "clinicaltrials_papers": 0},
                {"pubmed_papers": 0, "arxiv_papers": papers_per_query, "clinicaltrials_papers": 0},
                {"pubmed_papers": 0, "arxiv_papers": 0, "clinicaltrials_papers": papers_per_query},
            ]:
                try:
                    summary = await self.engine.analyze_research_topic(
                        idea=query,
                        max_papers=papers_per_query,
                        **source_config
                    )
                    
                    if summary and summary.papers_analyzed:
                        for paper in summary.papers_analyzed:
                            # Determine source
                            source = "Unknown"
                            if "ClinicalTrials.gov" in (paper.journal or ""):
                                source = "ClinicalTrials.gov"
                            elif paper.pmid:
                                source = "PubMed"
                            elif paper.arxiv_id or 'arxiv' in (paper.url or '').lower():
                                source = "arXiv"
                            
                            paper_dict = {
                                'id': paper.pmid or paper.arxiv_id or paper.title[:30],
                                'title': paper.title,
                                'source': source,
                                'journal': paper.journal,
                                'abstract': paper.abstract,
                                'sample_size': paper.sample_size,
                                'sample_size_method': paper.sample_size_method,
                                'p_value': paper.p_value,
                                'url': paper.url,
                                'query': query
                            }
                            all_papers.append(paper_dict)
                            
                except Exception as e:
                    print(f"  ❌ Error fetching papers: {e}")
                    
                await asyncio.sleep(1)  # Rate limiting
        
        return all_papers
    
    def analyze_abstract_for_n(self, abstract: str) -> Dict[str, Any]:
        """Analyze abstract for potential N patterns"""
        potential_patterns = {
            'n_equals': r"[Nn]\s*=\s*([\d,]+)",
            'n_parentheses': r"\([Nn]\s*=\s*([\d,]+)\)",
            'num_patients': r"([\d,]+)\s*(?:patients?|participants?|subjects?)",
            'enrolled': r"(?:enrolled|included|recruited|randomized)\s*([\d,]+)",
            'were_enrolled': r"([\d,]+)\s*(?:were|was)\s*(?:enrolled|included)",
            'total_of': r"total\s*(?:of\s*)?([\d,]+)",
            'sample_size': r"sample\s*size\s*(?:of\s*)?([\d,]+)",
            'completed': r"([\d,]+)\s*completed",
            'cohort': r"cohort\s*of\s*([\d,]+)",
            'men_women': r"([\d,]+)\s*(?:men|women|males|females)",
            'children': r"([\d,]+)\s*(?:children|infants|neonates|adolescents)",
            'aged': r"([\d,]+)\s*(?:aged|years old)",
            'data_from': r"data\s*from\s*([\d,]+)",
            'analyzed': r"(?:analyzed|analysed)\s*([\d,]+)",
            'treated': r"([\d,]+)\s*(?:treated|received treatment)",
            'assigned': r"([\d,]+)\s*(?:assigned|allocated)",
            'evaluated': r"(?:evaluated|assessed)\s*([\d,]+)",
            'consecutive': r"([\d,]+)\s*consecutive",
            'prospective': r"([\d,]+)\s*prospective",
            'retrospective': r"([\d,]+)\s*retrospective",
        }
        
        found_patterns = {}
        all_numbers = []
        
        for pattern_name, pattern in potential_patterns.items():
            matches = re.findall(pattern, abstract, re.IGNORECASE)
            if matches:
                # Clean up numbers
                clean_matches = []
                for match in matches:
                    try:
                        num = int(match.replace(',', ''))
                        if 10 <= num <= 100000:  # Reasonable range
                            clean_matches.append(num)
                            all_numbers.append(num)
                    except:
                        pass
                        
                if clean_matches:
                    found_patterns[pattern_name] = clean_matches
        
        # Find the most likely N (prefer smaller, more specific numbers)
        likely_n = None
        if all_numbers:
            # Sort and take the smallest reasonable number
            all_numbers.sort()
            likely_n = all_numbers[0]
        
        return {
            'found_patterns': found_patterns,
            'all_numbers': all_numbers,
            'likely_n': likely_n,
            'has_numbers': len(all_numbers) > 0
        }
    
    def init_log_file(self):
        """Initialize CSV log file with headers"""
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'run', 'paper_id', 'source', 'title', 'sample_size', 'sample_size_method',
                'p_value', 'has_numbers_in_abstract', 'likely_n', 'patterns_found',
                'abstract_excerpt', 'anomaly', 'notes'
            ])
            writer.writeheader()
    
    def log_paper(self, paper: Dict, analysis: Dict, run: int, notes: str = ""):
        """Log paper analysis to CSV"""
        # Check for anomalies
        anomaly = ""
        if paper['sample_size_method'] and not paper['sample_size']:
            anomaly = "Method without size"
        elif paper['sample_size'] and not paper['sample_size_method']:
            anomaly = "Size without method"
        
        # Get abstract excerpt with numbers
        excerpt = ""
        if analysis['has_numbers'] and paper['abstract']:
            # Find first occurrence of a number
            for num in analysis['all_numbers'][:1]:
                pattern = str(num)
                idx = paper['abstract'].lower().find(pattern)
                if idx > -1:
                    start = max(0, idx - 30)
                    end = min(len(paper['abstract']), idx + len(pattern) + 30)
                    excerpt = f"...{paper['abstract'][start:end]}..."
                    break
        
        row = {
            'run': run,
            'paper_id': paper['id'],
            'source': paper['source'],
            'title': paper['title'][:80],
            'sample_size': paper['sample_size'],
            'sample_size_method': paper['sample_size_method'],
            'p_value': paper['p_value'],
            'has_numbers_in_abstract': analysis['has_numbers'],
            'likely_n': analysis['likely_n'],
            'patterns_found': json.dumps(list(analysis['found_patterns'].keys()))[:100],
            'abstract_excerpt': excerpt[:150],
            'anomaly': anomaly,
            'notes': notes
        }
        
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
    
    async def run_test(self):
        """Main test runner"""
        print("=" * 80)
        print("COMPREHENSIVE N EXTRACTION TEST")
        print("=" * 80)
        
        # Initialize log
        self.init_log_file()
        
        # Define diverse queries to get varied paper types
        queries = [
            "randomized controlled trial",
            "clinical trial phase 3",
            "cohort study prospective",
            "systematic review meta-analysis",
            "case control study",
            "cross sectional survey",
            "longitudinal study follow-up",
            "pilot study feasibility"
        ]
        
        # Fetch papers (aim for ~100 per source)
        print("\n🔍 Fetching papers from all sources...")
        all_papers = await self.fetch_papers(queries, papers_per_query=15)
        
        # Group by source
        papers_by_source = {"PubMed": [], "ClinicalTrials.gov": [], "arXiv": []}
        for paper in all_papers:
            if paper['source'] in papers_by_source:
                papers_by_source[paper['source']].append(paper)
        
        print(f"\n📊 Papers collected:")
        for source, papers in papers_by_source.items():
            print(f"  {source}: {len(papers)} papers")
        
        # Process in batches of 10 from each source
        batch_size = 10
        self.run_number = 1
        
        while True:
            print(f"\n{'='*80}")
            print(f"RUN {self.run_number}")
            print(f"{'='*80}")
            
            papers_without_n = []
            papers_with_n = []
            
            # Analyze current state
            for source, papers in papers_by_source.items():
                for paper in papers[:batch_size * self.run_number]:  # Look at papers up to current batch
                    analysis = self.analyze_abstract_for_n(paper['abstract'])
                    self.log_paper(paper, analysis, self.run_number)
                    
                    if not paper['sample_size']:
                        if analysis['has_numbers']:
                            papers_without_n.append((paper, analysis))
                    else:
                        papers_with_n.append(paper)
            
            print(f"\n📈 Current extraction rates:")
            total = sum(len(papers) for papers in papers_by_source.values())
            with_n = len(papers_with_n)
            print(f"  Overall: {with_n}/{total} ({100*with_n/total:.1f}%)")
            
            if not papers_without_n or self.run_number >= 10:
                break
            
            # Analyze missed patterns
            print(f"\n🔍 Analyzing {len(papers_without_n)} papers without extracted N...")
            pattern_frequency = {}
            for paper, analysis in papers_without_n[:10]:  # Focus on 10 at a time
                print(f"\n  Paper: {paper['title'][:60]}...")
                print(f"    Source: {paper['source']}")
                print(f"    Likely N: {analysis['likely_n']}")
                print(f"    Patterns: {list(analysis['found_patterns'].keys())[:3]}")
                
                for pattern in analysis['found_patterns'].keys():
                    pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
            
            # Improve patterns based on findings
            if pattern_frequency:
                print(f"\n📊 Most common unextracted patterns:")
                for pattern, count in sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    {pattern}: {count} occurrences")
                
                # Add improvements to extraction logic
                await self.improve_extraction_patterns(pattern_frequency)
            
            self.run_number += 1
            await asyncio.sleep(1)
        
        print(f"\n✅ Test complete! Results logged to {self.log_file}")
        await self.generate_final_report()
    
    async def improve_extraction_patterns(self, pattern_frequency: Dict[str, int]):
        """Improve extraction patterns based on findings"""
        print(f"\n🔧 Improving extraction patterns based on findings...")
        
        # This is where we would modify the patterns in research_intelligence.py
        # For now, we'll note what improvements would be made
        improvements = []
        
        if 'aged' in pattern_frequency and pattern_frequency['aged'] > 2:
            improvements.append("Add pattern for 'X aged Y years'")
        
        if 'treated' in pattern_frequency and pattern_frequency['treated'] > 2:
            improvements.append("Add pattern for 'X patients treated'")
        
        if 'assigned' in pattern_frequency and pattern_frequency['assigned'] > 2:
            improvements.append("Add pattern for 'X assigned to treatment'")
        
        if improvements:
            print(f"  Improvements to make:")
            for imp in improvements:
                print(f"    - {imp}")
        
        # In reality, we would update the patterns here
        # For demonstration, we'll continue with analysis
    
    async def generate_final_report(self):
        """Generate final summary report"""
        print(f"\n{'='*80}")
        print("FINAL REPORT")
        print(f"{'='*80}")
        
        # Read the log file and analyze
        import pandas as pd
        df = pd.read_csv(self.log_file)
        
        # Overall statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  Total papers analyzed: {len(df)}")
        print(f"  Papers with N extracted: {df['sample_size'].notna().sum()}")
        print(f"  Papers with numbers but no N: {(df['has_numbers_in_abstract'] & df['sample_size'].isna()).sum()}")
        
        # By source
        print(f"\n📊 By Source:")
        for source in df['source'].unique():
            source_df = df[df['source'] == source]
            extracted = source_df['sample_size'].notna().sum()
            total = len(source_df)
            print(f"  {source}: {extracted}/{total} ({100*extracted/total:.1f}%)")
        
        # Anomalies
        anomalies = df[df['anomaly'] != '']
        if not anomalies.empty:
            print(f"\n⚠️  Anomalies found: {len(anomalies)}")
            for _, row in anomalies.head(5).iterrows():
                print(f"    {row['paper_id']}: {row['anomaly']}")
        
        # Most successful patterns
        print(f"\n✅ Most successful extraction patterns:")
        # This would require more detailed tracking
        
        print(f"\n📄 Full results saved to: {self.log_file}")

async def main():
    tester = ComprehensiveNExtractionTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())