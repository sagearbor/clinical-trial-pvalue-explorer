#!/usr/bin/env python3
"""Test with a specific older paper that should have N"""

import requests
import xml.etree.ElementTree as ET
import re

# Try an older, well-established RCT
fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# PMID 25270860 is a hearing aid RCT from 2014
params = {
    "db": "pubmed", 
    "id": "25270860",
    "retmode": "xml"
}

response = requests.get(fetch_url, params=params)
if response.status_code == 200:
    root = ET.fromstring(response.text)
    
    # Get title
    title = root.findtext(".//ArticleTitle") or "No title"
    print(f"Title: {title}")
    
    # Get abstract - handle both regular and labeled sections
    abstract_parts = []
    for elem in root.findall(".//AbstractText"):
        label = elem.get("Label", "")
        text = elem.text or ""
        if label:
            abstract_parts.append(f"{label}: {text}")
        else:
            abstract_parts.append(text)
    
    abstract = " ".join(abstract_parts)
    print(f"\nAbstract length: {len(abstract)}")
    
    if abstract:
        print(f"\nAbstract preview:\n{abstract[:500]}...")
        
        # Look for N
        n_pattern = re.compile(r"\bn\s*=\s*([\d,]+)", re.IGNORECASE)
        matches = n_pattern.findall(abstract)
        if matches:
            print(f"\n✅ Found N values: {matches}")
        
        # Also try other patterns
        patient_pattern = re.compile(r"(\d+)\s+(?:patients|participants|subjects)", re.IGNORECASE)
        patient_matches = patient_pattern.findall(abstract)
        if patient_matches:
            print(f"✅ Found patient counts: {patient_matches}")
    else:
        print("No abstract found!")
        
    # Try structured fields
    print("\n" + "="*70)
    print("Checking structured data...")
    
    # Sometimes N is in other fields
    for field in ["DataBankList", "MeshHeadingList", "KeywordList"]:
        elements = root.findall(f".//{field}")
        if elements:
            print(f"Found {field}")
