#!/usr/bin/env python3
"""Find a PubMed article that definitely has N in the abstract"""

import requests
import xml.etree.ElementTree as ET

# Search for hearing aid studies that likely have sample sizes
search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {
    "db": "pubmed",
    "term": "hearing aids randomized controlled trial",
    "retmax": 5,
    "retmode": "json"
}

response = requests.get(search_url, params=params)
pmids = response.json()["esearchresult"]["idlist"]

print(f"Found {len(pmids)} PMIDs, checking for N values...\n")

# Fetch details for each
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

for pmid in pmids[:3]:
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    response = requests.get(fetch_url, params=params)
    
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        title = root.findtext(".//ArticleTitle") or "No title"
        
        # Get all abstract text
        abstract_texts = root.findall(".//AbstractText")
        abstract = " ".join([elem.text or "" for elem in abstract_texts])
        
        print(f"PMID: {pmid}")
        print(f"Title: {title[:60]}...")
        print(f"Abstract length: {len(abstract)}")
        
        # Check for N patterns
        import re
        n_patterns = [
            r"\bn\s*=\s*(\d+)",
            r"\(n\s*=\s*(\d+)\)",
            r"(\d+)\s+patients",
            r"(\d+)\s+participants",
            r"enrolled\s+(\d+)"
        ]
        
        found_n = False
        for pattern in n_patterns:
            matches = re.findall(pattern, abstract.lower())
            if matches:
                print(f"  ✅ Found N values: {matches}")
                found_n = True
                break
        
        if not found_n:
            print(f"  ❌ No N found")
        
        # Show a snippet where N might be
        if "n=" in abstract.lower() or "n =" in abstract.lower():
            idx = abstract.lower().find("n")
            print(f"  Context: ...{abstract[max(0,idx-20):min(len(abstract),idx+50)]}...")
        
        print("-" * 70)
