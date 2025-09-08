#!/usr/bin/env python3
"""Test if journal names are being extracted correctly"""

import requests
import xml.etree.ElementTree as ET

# Fetch PMID 40913255 directly
response = requests.get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params={
    'db': 'pubmed',
    'id': '40913255',
    'retmode': 'xml'
})

root = ET.fromstring(response.text)
art = root.find('.//PubmedArticle')

# Get journal info
journal_title = art.findtext(".//Journal/Title")
journal_abbrev = art.findtext(".//Journal/ISOAbbreviation")

print(f"Journal Title: {journal_title}")
print(f"Journal Abbreviation: {journal_abbrev}")

# Get abstract with proper structure
abstract_parts = []
for elem in art.findall('.//AbstractText'):
    label = elem.get('Label', '')
    text = elem.text or ''
    if label and text:
        abstract_parts.append(f"{label}: {text}")
    elif text:
        abstract_parts.append(text)

abstract = " ".join(abstract_parts)
print(f"\nAbstract length: {len(abstract)}")

# Check for N=321
if "n = 321" in abstract:
    print("✅ Found 'n = 321' in abstract")
    idx = abstract.find("n = 321")
    print(f"Context: ...{abstract[max(0,idx-50):min(len(abstract),idx+50)]}...")
    
# Check for p-values
import re
p_pattern = re.compile(r"p\s*[<=]\s*([\d.]+)", re.IGNORECASE)
p_matches = p_pattern.findall(abstract)
if p_matches:
    print(f"\n✅ Found p-values: {p_matches}")
else:
    print("\n❌ No p-values found")
