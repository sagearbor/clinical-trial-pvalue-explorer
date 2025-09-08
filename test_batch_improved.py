#!/usr/bin/env python3
"""Extended batch test with improved patterns"""

import asyncio
import csv
import re
from datetime import datetime
from src.research_intelligence import ResearchIntelligenceEngine, _extract_sample_size

class ImprovedBatchTester:
    def __init__(self):
        self.engine = ResearchIntelligenceEngine()
        self.results_file = "improved_extraction_results.csv"
        
    async def test_diverse_queries(self):
        """Test with diverse medical queries to get varied abstract styles"""
        
        queries = [
            # Different study types
            "randomized controlled trial diabetes",
            "cohort study cardiovascular",
            "case control cancer",
            "cross sectional survey mental health",
            "systematic review meta-analysis",
            "pilot study feasibility",
            "longitudinal follow-up",
            "retrospective analysis",
            
            # Different medical domains
            "clinical trial oncology",
            "pediatric study obesity",
            "geriatric intervention dementia",
            "surgical outcomes",
            "drug efficacy safety",
            "vaccine immunogenicity",
            "diagnostic accuracy",
            "therapeutic intervention"
        ]
        
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Source', 'Title', 'N_Found', 'Method', 'P_value', 'Success'])
        
        total_papers = 0
        successful_extractions = 0
        by_source = {'PubMed': {'total': 0, 'success': 0}, 
                     'ClinicalTrials.gov': {'total': 0, 'success': 0},
                     'arXiv': {'total': 0, 'success': 0}}
        
        print("=" * 80)
        print("IMPROVED PATTERN EXTRACTION TEST")
        print("=" * 80)
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Testing query: '{query}'")
            
            # Test each source
            for source_name, config in [
                ('PubMed', {"pubmed_papers": 5, "arxiv_papers": 0, "clinicaltrials_papers": 0}),
                ('ClinicalTrials.gov', {"pubmed_papers": 0, "arxiv_papers": 0, "clinicaltrials_papers": 5}),
                ('arXiv', {"pubmed_papers": 0, "arxiv_papers": 5, "clinicaltrials_papers": 0})
            ]:
                try:
                    summary = await self.engine.analyze_research_topic(
                        idea=query,
                        max_papers=5,
                        **config
                    )
                    
                    if summary and summary.papers_analyzed:
                        for paper in summary.papers_analyzed:
                            total_papers += 1
                            by_source[source_name]['total'] += 1
                            
                            success = paper.sample_size is not None
                            if success:
                                successful_extractions += 1
                                by_source[source_name]['success'] += 1
                            
                            # Log result
                            with open(self.results_file, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    query[:30],
                                    source_name,
                                    paper.title[:50],
                                    paper.sample_size,
                                    paper.sample_size_method,
                                    paper.p_value,
                                    'Y' if success else 'N'
                                ])
                            
                            # Print inline progress
                            status = "✅" if success else "❌"
                            print(f"  {status} {source_name}: N={paper.sample_size}")
                            
                except Exception as e:
                    print(f"  ⚠️ Error with {source_name}: {e}")
            
            await asyncio.sleep(1)  # Rate limiting
            
            # Print running totals
            if i % 4 == 0:
                print(f"\n📊 Progress Report (after {i} queries):")
                print(f"  Overall: {successful_extractions}/{total_papers} ({100*successful_extractions/max(1,total_papers):.1f}%)")
                for src, stats in by_source.items():
                    if stats['total'] > 0:
                        rate = 100 * stats['success'] / stats['total']
                        print(f"  {src}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Final summary
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"\n📊 Overall Extraction Rate: {successful_extractions}/{total_papers} ({100*successful_extractions/max(1,total_papers):.1f}%)")
        
        print("\n📊 By Source:")
        for src, stats in by_source.items():
            if stats['total'] > 0:
                rate = 100 * stats['success'] / stats['total']
                print(f"  {src}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        print(f"\n📄 Detailed results saved to: {self.results_file}")
        
        # Analyze patterns in failures
        await self.analyze_failures()
    
    async def analyze_failures(self):
        """Analyze papers that failed extraction"""
        print("\n🔍 Analyzing extraction failures...")
        
        failures = []
        with open(self.results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Success'] == 'N':
                    failures.append(row)
        
        if failures:
            print(f"  Found {len(failures)} failed extractions")
            
            # Group by source
            by_source = {}
            for fail in failures:
                src = fail['Source']
                by_source[src] = by_source.get(src, 0) + 1
            
            print("  Failures by source:")
            for src, count in by_source.items():
                print(f"    {src}: {count}")
            
            # Sample some failures for manual review
            print("\n  Sample failures for review:")
            for fail in failures[:5]:
                print(f"    - {fail['Title'][:50]} ({fail['Source']})")

async def main():
    tester = ImprovedBatchTester()
    await tester.test_diverse_queries()

if __name__ == "__main__":
    asyncio.run(main())