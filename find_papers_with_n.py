#!/usr/bin/env python3
"""Search for papers that explicitly mention sample sizes"""

import requests
import xml.etree.ElementTree as ET
import re

# Search specifically for papers that mention sample size in abstract
search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
params = {
    "db": "pubmed",
    "term": '"n = " hearing aids',  # Look for explicit n = mentions
    "retmax": 5,
    "retmode": "json"
}

response = requests.get(search_url, params=params)
data = response.json()
pmids = data["esearchresult"]["idlist"]

print(f"Searching for papers with 'n =' in abstract...")
print(f"Found {len(pmids)} potential matches\n")

if pmids:
    # Fetch the first one in detail
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmids[0],
        "retmode": "xml"
    }
    
    response = requests.get(fetch_url, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        title = root.findtext(".//ArticleTitle") or "No title"
        pmid = pmids[0]
        
        # Get abstract
        abstract_texts = root.findall(".//AbstractText")
        abstract = " ".join([elem.text or "" for elem in abstract_texts if elem.text])
        
        print(f"Example paper with N:")
        print(f"PMID: {pmid}")
        print(f"Title: {title}")
        print(f"Abstract preview: {abstract[:300]}...")
        
        # Extract N values
        n_pattern = re.compile(r"\bn\s*=\s*([\d,]+)", re.IGNORECASE)
        matches = n_pattern.findall(abstract)
        if matches:
            print(f"\n✅ Found N values: {matches}")
            print(f"Largest N: {max([int(m.replace(',','')) for m in matches])}")
            
        # Also check for p-values
        p_pattern = re.compile(r"p\s*[<=]\s*([\d.]+)", re.IGNORECASE)
        p_matches = p_pattern.findall(abstract)
        if p_matches:
            print(f"✅ Found p-values: {p_matches}")
else:
    print("No papers found with 'n =' pattern")

# Try alternative search
print("\n" + "="*70)
print("Trying search for clinical trials with sample sizes...")

params2 = {
    "db": "pubmed",
    "term": "hearing aids clinical trial 321 patients",  # The specific example user mentioned
    "retmax": 3,
    "retmode": "json"
}

response2 = requests.get(search_url, params=params2)
pmids2 = response2.json()["esearchresult"]["idlist"]

if pmids2:
    print(f"Found {len(pmids2)} papers")
    # Check first one
    params = {
        "db": "pubmed",
        "id": pmids2[0],
        "retmode": "xml"
    }
    response = requests.get(fetch_url, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        abstract = " ".join([elem.text or "" for elem in root.findall(".//AbstractText") if elem.text])
        title = root.findtext(".//ArticleTitle") or "No title"
        
        print(f"\nFirst result:")
        print(f"Title: {title[:60]}...")
        
        # Check for any number that could be N
        numbers = re.findall(r"\b(\d{2,4})\b", abstract)
        if numbers:
            print(f"Numbers found in abstract: {numbers[:5]}")
