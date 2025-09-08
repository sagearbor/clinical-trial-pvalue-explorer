#!/usr/bin/env python3
"""Get the arXiv URL for the tumor paper"""

import requests
import json

response = requests.post("http://localhost:8123/process_idea", json={
    "study_description": "How to help older adults increase uptake of hearing devices to increase quality of life",
    "include_research": True,
    "max_papers": 9,
    "pubmed_papers": 3,
    "clinicaltrials_papers": 3,
    "arxiv_papers": 3
})

if response.status_code == 200:
    data = response.json()
    papers = data.get('research_papers_data', [])
    
    for paper in papers:
        if 'tumor' in paper.get('title', '').lower() or 'seven-step' in paper.get('title', '').lower():
            print(f"Title: {paper.get('title')}")
            print(f"URL: {paper.get('url')}")
            print(f"Sample size: {paper.get('sample_size')}")
            print(f"Abstract preview: {paper.get('abstract', '')[:200]}...")
            break