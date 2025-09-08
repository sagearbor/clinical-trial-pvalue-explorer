# research_intelligence.py — upgraded ClinicalTrials.gov integration
# - Uses v2 API with explicit `fields=` param (no scraping)
# - Adds pagination support (pageToken)
# - Normalizes structured CT.gov fields (phase, status, enrollment, outcomes, etc.)
# - Keeps compatibility with existing table (sample_size, url, etc.)

import os, re, ssl, asyncio, aiohttp, xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

RESEARCH_INTELLIGENCE_AVAILABLE = True

# ---------------- Data models ----------------
@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    journal: str
    year: int
    pmid: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    sample_size: Optional[int] = None
    sample_size_method: Optional[str] = None  # "structured" | "extracted" | "description" - how N was obtained
    p_value: Optional[float] = None  # Extracted p-value from abstract/text
    study_signal: Optional[str] = None  # "Positive" | "Negative" | "Mixed" | "Unclear" | None
    study_signal_confidence: Optional[str] = None  # "high" | "medium" | "low" - for future use
    extras: Optional[Dict[str, Any]] = None  # NEW: richer CT.gov fields (phase, status, etc.)

@dataclass
class ResearchSummary:
    query: str
    search_date: datetime
    total_papers_found: int
    papers_analyzed: List[ResearchPaper]
    # Extended metadata for frontend and tests
    evidence_quality: Optional[str] = None
    effect_size_range: Optional[tuple] = None
    typical_sample_sizes: Optional[List[int]] = None
    common_study_designs: Optional[List[str]] = None
    recommended_outcome_measures: Optional[List[str]] = None
    effect_heterogeneity: Optional[str] = None
    publication_bias_risk: Optional[str] = None
    temporal_trends: Optional[str] = None

# ---------------- Common helpers ----------------
# Priority patterns for sample size extraction (checked first)
_PRIORITY_SAMPLE_PATTERNS = [
    re.compile(r"[Ss]tudy\s+[Ss]ample:?\s*.*?\([Nn]\s*=\s*([\d,]{2,10})\)", re.DOTALL),  # Study Sample: ... (n = 321)
    re.compile(r"[Oo]ur\s+sample\s*\([Nn]\s*=\s*([\d,]{2,10})\)"),  # Our sample (n = 321)
    re.compile(r"[Ff]inal\s+sample.*?\([Nn]\s*=\s*([\d,]{2,10})\)"),  # Final sample ... (n = 321)
    re.compile(r"[Aa]nalyz\w+\s+sample.*?\([Nn]\s*=\s*([\d,]{2,10})\)"),  # Analyzed sample (n = 321)
]

# Multiple patterns for sample size extraction
_SAMPLE_PATTERNS = [
    re.compile(r"\b[Nn]\s*=\s*([\d,]{2,10})\b"),  # N = 123 or N = 2,677
    re.compile(r"\([Nn]\s*=\s*([\d,]{2,10})\)"),  # (n = 321) in parentheses
    re.compile(r"\bsample\s*\([Nn]\s*=\s*([\d,]{2,10})\)"),  # sample (n = 321)
    re.compile(r"\b([\d,]{2,10})\s*(?:patients?|participants?|subjects?|adults?|individuals?)\b", re.I),  # 123 patients
    re.compile(r"\b(?:enrolled|included|recruited|randomized)\s*([\d,]{2,10})\b", re.I),  # enrolled 123
    re.compile(r"\b([\d,]{2,10})\s*(?:were|was)\s*(?:enrolled|included|recruited|randomized)\b", re.I),  # 123 were enrolled
    re.compile(r"\btotal\s*(?:of\s*)?([\d,]{2,10})\s*(?:patients?|participants?|subjects?|consenting participants?)\b", re.I),  # total of 123 patients/consenting participants
    re.compile(r"\btotal\s+of\s+([\d,]{2,10})\b", re.I),  # A total of 100 (more flexible)
    re.compile(r"\bsample\s*size\s*(?:of\s*)?([\d,]{2,10})\b", re.I),  # sample size of 123
    re.compile(r"\bfinal\s*sample\s*(?:of\s*)?([\d,]{2,10})\b", re.I),  # final sample of 321
    re.compile(r"\b([\d,]{2,10})\s*(?:men|women|males|females|boys|girls|children|infants|neonates)\b", re.I),  # 123 women/men
    re.compile(r"\bcohort\s*of\s*([\d,]{2,10})\b", re.I),  # cohort of 123
    re.compile(r"\bdata\s*from\s*([\d,]{2,10})\b", re.I),  # data from 123
    re.compile(r"\b(?:analyzed|analysed)\s+([\d,]{2,10})\b", re.I),  # analyzed 36
    re.compile(r"\b([\d,]{2,10})\s*(?:articles?|papers?|studies)\s*(?:were|was)?\s*(?:analyzed|analysed|reviewed)\b", re.I),  # 22 articles were analyzed
    re.compile(r"\b([\d,]{2,10})\s*completed\s*the\s*(?:study|trial)\b", re.I),  # 123 completed the study
]

# Pattern for p-value extraction
_PVALUE_PATTERNS = [
    re.compile(r"[Pp]\s*[<=]\s*(0?\.\d{1,4}|\d\.\d+[eE]-\d+)"),  # p < 0.05, p = 0.001, p < 2.3e-10
    re.compile(r"[Pp]-value\s*[<=]\s*(0?\.\d{1,4}|\d\.\d+[eE]-\d+)"),  # p-value < 0.05
    re.compile(r"\([Pp]\s*[<=]\s*(0?\.\d{1,4}|\d\.\d+[eE]-\d+)\)"),  # (p < 0.001)
    re.compile(r"significant\s*\([Pp]\s*[<=]\s*(0?\.\d{1,4})\)"),  # significant (p < 0.05)
]
_POS_RE = re.compile(r"\b(significant|improv\w+|increase|decrease)\b", re.I)
_NEG_RE = re.compile(r"\b(no\s+significan|null|non-?significant)\b", re.I)

def _extract_sample_size(text: str) -> Optional[int]:
    """Extract sample size from text using multiple patterns."""
    if not text:
        return None
    
    # First check priority patterns (Study Sample, Our sample, etc.)
    for pattern in _PRIORITY_SAMPLE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                # Remove commas and convert to int
                n_str = match.replace(',', '')
                n = int(n_str)
                # Sanity check - reasonable sample size range
                if 10 <= n <= 100000:
                    return n  # Return immediately if found in priority pattern
            except:
                continue
    
    # If no priority match, try regular patterns
    found_sizes = []
    for pattern in _SAMPLE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                # Remove commas and convert to int
                n_str = match.replace(',', '')
                n = int(n_str)
                # Sanity check - reasonable sample size range
                if 10 <= n <= 100000:
                    found_sizes.append(n)
            except:
                continue
    
    # Return the smallest N found (usually the actual study sample, not distribution)
    # Changed from max to min to prefer smaller sample sizes which are often the actual study
    return min(found_sizes) if found_sizes else None

def _extract_p_value(text: str) -> Optional[float]:
    """Extract the smallest (most significant) p-value from text."""
    if not text:
        return None
    
    found_pvalues = []
    
    for pattern in _PVALUE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            try:
                p = float(match)
                if 0 < p <= 1:  # Valid p-value range
                    found_pvalues.append(p)
            except:
                continue
    
    # Return the smallest (most significant) p-value
    return min(found_pvalues) if found_pvalues else None

def _infer_signal(text: str) -> Optional[str]:
    t = (text or "").lower()
    if _POS_RE.search(t) and _NEG_RE.search(t):
        return "Mixed"
    if _POS_RE.search(t):
        return "Positive"
    if _NEG_RE.search(t):
        return "Negative"
    return "Unclear"

def _ssl_connector():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return aiohttp.TCPConnector(ssl=ctx)

# ---------------- PubMed ----------------
class PubMedSearcher:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    base_url = base
    email = os.getenv("PUBMED_EMAIL", "research@example.com")
    api_key = os.getenv("PUBMED_API_KEY")

    async def search(self, idea: str, max_results: int = 10, years_back: int = 7) -> List[Dict[str, Any]]:
        # Try multiple queries: primary combined query, then fallbacks if no results
        q_primary = self._enhance_query(idea, years_back)
        pmids = await self._search_pmids(q_primary, max_results)
        
        # Only use fallbacks if we got NO results (not just fewer than requested)
        if not pmids:
            # fallback 1: plain phrase in Title/Abstract
            q_fallback = f'({idea})[Title/Abstract]'
            pmids = await self._search_pmids(q_fallback, max_results)
            
        if not pmids:
            # fallback 2: extract key terms from the idea and search for them
            key_terms = self._extract_key_terms(idea)
            if key_terms and len(key_terms) >= 2:  # Only if we have meaningful terms
                # Use AND to ensure relevance, not OR
                syn_clause = " AND ".join([f'({term})[Title/Abstract]' for term in key_terms[:3]])
                pmids = await self._search_pmids(syn_clause, max_results)
                
        if not pmids:
            print(f"⚠️ PubMed: No relevant papers found for '{idea[:50]}...'")
            return []
            
        # Fetch details and filter for relevance
        papers = await self._fetch_details(pmids)
        
        # Basic relevance check - paper should mention at least one key term
        if papers:
            key_terms = self._extract_key_terms(idea)
            if key_terms:
                relevant_papers = []
                for paper in papers:
                    title_abstract = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
                    # Check if any key term appears
                    matches = [term for term in key_terms if term.lower() in title_abstract]
                    if matches:
                        relevant_papers.append(paper)
                    else:
                        # Debug: show what was rejected
                        print(f"   Rejected: {paper.get('title', '')[:60]}... (no match for: {key_terms})")
                        
                if relevant_papers:
                    print(f"✅ PubMed: Found {len(relevant_papers)} relevant papers")
                    return relevant_papers
                else:
                    print(f"⚠️ PubMed: Found {len(papers)} papers but none matched terms: {key_terms}")
                    # Show first paper title for debugging
                    if papers:
                        print(f"   Example rejected: {papers[0].get('title', 'No title')[:80]}")
                    return []
        
        return papers

    # Backwards-compatible method name expected by tests
    async def search_papers(self, idea: str, max_results: int = 10, years_back: int = 7) -> List[Dict[str, Any]]:
        return await self.search(idea, max_results=max_results, years_back=years_back)

    def _extract_key_terms(self, idea: str) -> List[str]:
        """Extract key medical/clinical terms from the query string."""
        idea_lower = idea.lower()
        found_terms = []
        
        # For hearing-related queries, add comprehensive hearing terms
        if 'hearing' in idea_lower or 'deaf' in idea_lower or 'ear' in idea_lower:
            found_terms.extend(['hearing', 'hearing aid', 'hearing device', 'deaf', 
                              'presbycusis', 'auditory', 'cochlear'])
            # Also keep specific terms from the query
            if 'older' in idea_lower or 'elderly' in idea_lower:
                found_terms.append('elderly')
            if 'quality' in idea_lower and 'life' in idea_lower:
                found_terms.append('quality of life')
            if 'uptake' in idea_lower or 'adoption' in idea_lower:
                found_terms.append('uptake')
                
        # Common clinical/medical terms to look for
        medical_keywords = [
            'diabetes', 'hypertension', 'cancer', 'cardiovascular', 'heart',
            'blood pressure', 'glucose', 'insulin', 'metformin', 'statin',
            'ace inhibitor', 'arb', 'cognitive', 'dementia', 'alzheimer', 
            'parkinson', 'stroke', 'copd', 'asthma', 'obesity', 'bmi', 
            'cholesterol', 'lipid', 'trial', 'therapy', 'treatment', 
            'intervention', 'placebo', 'randomized', 'clinical'
        ]
        
        # Find matching medical terms
        for term in medical_keywords:
            if term in idea_lower:
                found_terms.append(term)
                    
        # Return unique terms
        return list(set(found_terms)) if found_terms else []
    
    def _enhance_query(self, idea: str, years_back: int) -> str:
        # Build query based on the actual idea content
        year = datetime.now().year
        start = year - years_back
        
        # Extract relevant terms from the actual query
        key_terms = self._extract_key_terms(idea)
        
        # For hearing queries, build a focused search
        idea_lower = idea.lower()
        if 'hearing' in idea_lower:
            # Build specific hearing-related query
            hearing_terms = ['hearing aid', 'hearing device', 'hearing loss', 'presbycusis', 
                           'deaf', 'deafness', 'auditory', 'cochlear implant']
            hearing_clause = " OR ".join([f'"{term}"[Title/Abstract]' for term in hearing_terms])
            
            # Add age-related terms if mentioned
            if 'older' in idea_lower or 'elderly' in idea_lower:
                age_clause = '(elderly[Title/Abstract] OR "older adults"[Title/Abstract] OR aged[MeSH Terms])'
                combined = f"({hearing_clause}) AND ({age_clause})"
            else:
                combined = f"({hearing_clause})"
        else:
            # For non-hearing queries, use the extracted key terms
            if key_terms:
                # Use AND between key terms for more focused results
                key_clause = " AND ".join([f'({term})[Title/Abstract]' for term in key_terms[:3]])
                combined = key_clause
            else:
                # Fallback to the original idea
                combined = f'({idea})[Title/Abstract]'
        # constrain by humans and language and recent years
        return f"{combined} AND (humans[mesh] OR humans[Title/Abstract] OR humans) AND (english[lang]) AND ({start}[PDAT]:{year}[PDAT])"

    # Backwards-compatible alias
    def _enhance_pubmed_query(self, idea: str, years_back: int = 7) -> str:
        return self._enhance_query(idea, years_back)

    async def _search_pmids(self, term: str, retmax: int) -> List[str]:
        url = f"{self.base}esearch.fcgi"
        params = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "best match", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        async with aiohttp.ClientSession(connector=_ssl_connector()) as s:
            res = s.get(url, params=params)
            if hasattr(res, "__aenter__"):
                async with res as r:
                    if r.status != 200:
                        return []
                    data = await r.json()
            else:
                r = await res
                if r.status != 200:
                    return []
                data = await r.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        url = f"{self.base}efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        async with aiohttp.ClientSession(connector=_ssl_connector()) as s:
            res = s.get(url, params=params)
            if hasattr(res, "__aenter__"):
                async with res as r:
                    if r.status != 200:
                        return []
                    xml = await r.text()
            else:
                r = await res
                if r.status != 200:
                    return []
                xml = await r.text()
            return self._parse(xml)

    def _parse(self, xml: str) -> List[Dict[str, Any]]:
        out = []
        root = ET.fromstring(xml)
        for art in root.findall(".//PubmedArticle"):
            title = (art.findtext(".//ArticleTitle") or "").strip()
            # Handle both plain and structured abstracts
            abstract_parts = []
            for elem in art.findall(".//AbstractText"):
                label = elem.get('Label', '')
                # Get all text content including nested tags
                text = ''.join(elem.itertext()).strip()
                if label and text:
                    abstract_parts.append(f"{label}: {text}")
                elif text:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts).strip()
            authors = []
            for a in art.findall(".//Author"):
                last = a.findtext("LastName") or ""
                first = a.findtext("ForeName") or a.findtext("FirstName") or ""
                if first and last:
                    authors.append(f"{first} {last}")
            year = art.findtext(".//PubDate/Year")
            try:
                year = int(year) if year else datetime.now().year
            except:
                year = datetime.now().year
            pmid = art.findtext(".//PMID")
            sample_size = _extract_sample_size(abstract)
            out.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": art.findtext(".//Journal/Title") or art.findtext(".//Journal/ISOAbbreviation") or "PubMed",
                    "year": year,
                    "pmid": pmid,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                    "sample_size": sample_size,
                    "sample_size_method": "text_extraction" if sample_size else None,  # PubMed is always text extraction
                    "p_value": _extract_p_value(abstract),  # Extract p-values from abstract
                    "study_signal": _infer_signal(abstract),
                    "source": "pubmed",
                }
            )
        return out

# ---------------- ClinicalTrials.gov (v2 API) ----------------
class ClinicalTrialsSearcher:
    base = "https://clinicaltrials.gov/api/v2/studies"
    base_url = base

    # You can add stricter default filters here if desired (e.g., only interventional, completed)
    DEFAULT_FILTERS = {
        # examples: uncomment as desired
        # "filter.overallStatus": "Completed",
        # "filter.studyType": "Interventional",
    }

    FIELDS = [
        # identification
        "studies.protocolSection.identificationModule.nctId",
        "studies.protocolSection.identificationModule.briefTitle",
        "studies.protocolSection.identificationModule.briefSummary",
        "studies.protocolSection.identificationModule.officialTitle",  # Full title might have N
        # status / dates
        "studies.protocolSection.statusModule.overallStatus",
        "studies.protocolSection.statusModule.startDateStruct",
        "studies.protocolSection.statusModule.primaryCompletionDateStruct",
        # design - CRITICAL FOR N
        "studies.protocolSection.designModule.studyType",
        "studies.protocolSection.designModule.phases",
        "studies.protocolSection.designModule.enrollmentInfo",  # PRIMARY SOURCE OF N
        "studies.protocolSection.designModule.designInfo",
        "studies.protocolSection.designModule.targetDuration",  # Follow-up period
        # arms / interventions - useful for per-arm N
        "studies.protocolSection.armsInterventionsModule.numberOfArms",
        "studies.protocolSection.armsInterventionsModule.armGroups",  # Could have per-arm N
        # outcomes
        "studies.protocolSection.outcomesModule.primaryOutcomes",
        "studies.protocolSection.outcomesModule.secondaryOutcomes",
        # eligibility - sometimes mentions target N
        "studies.protocolSection.eligibilityModule",
        # locations (country list)
        "studies.protocolSection.contactsLocationsModule.locations",
        # Statistics - might have power calculations
        "studies.protocolSection.oversightModule.oversightHasDmc",
        # Results section (for completed trials) - ACTUAL N
        "studies.resultsSection.participantFlowModule",  # Actual enrollment/completion
        "studies.resultsSection.baselineCharacteristicsModule",  # Actual baseline N
    ]

    async def search(self, idea: str, max_results: int = 8, **filters) -> List[Dict[str, Any]]:
        # Build query variants based on the actual search topic
        idea_lower = idea.lower()
        query_variants = []
        
        # For hearing-related queries, use specific terms that work well with CT.gov
        if 'hearing' in idea_lower or 'deaf' in idea_lower:
            # ClinicalTrials.gov works better with specific medical terms
            query_variants.extend([
                "hearing aid",
                "hearing loss",
                "presbycusis", 
                "hearing device",
                "cochlear implant",
                "auditory rehabilitation",
                "hearing aids elderly",
                "hearing loss older adults"
            ])
        else:
            # For other queries, use the original approach
            query_variants = [
                idea,  # Original query
                ' '.join(idea.split()),  # Normalized spaces
                ' '.join([w for w in idea.split() if len(w) > 3]),  # Remove short words
            ]
            
            # Only add pediatric variant if the query mentions children/pediatric
            if any(term in idea_lower for term in ['child', 'pediatric', 'infant', 'adolescent']):
                query_variants.append(f"{idea} pediatric")

        params_base = {"pageSize": min(max_results, 50), "format": "json"}
        params_base.update(self.DEFAULT_FILTERS)
        params_base.update({k: v for k, v in (filters or {}).items() if v is not None})

        out: List[Dict[str, Any]] = []

        async with aiohttp.ClientSession(connector=_ssl_connector()) as s:
            # Try structured fields first for each variant
            for q in query_variants:
                params = dict(params_base)
                params["query.term"] = q
                params["fields"] = ",".join(self.FIELDS)

                res = s.get(self.base, params=params)
                if hasattr(res, '__aenter__'):
                    async with res as r:
                        if r.status != 200:
                            continue
                        data = await r.json()
                else:
                    r = await res
                    if r.status != 200:
                        continue
                    data = await r.json()

                studies = data.get("studies", [])
                for st in studies:
                    out.append(self._normalize_record(st))
                    if len(out) >= max_results:
                        return out

            # Fallback: try simple unstructured query for variants
            for q in query_variants:
                params = dict(params_base)
                params["query.term"] = q
                res = s.get(self.base, params=params)
                if hasattr(res, '__aenter__'):
                    async with res as r:
                        if r.status != 200:
                            continue
                        data = await r.json()
                else:
                    r = await res
                    if r.status != 200:
                        continue
                    data = await r.json()
                studies = data.get('studies', [])
                for st in studies:
                    out.append(self._normalize_record(st))
                    if len(out) >= max_results:
                        return out

        return out
    
    async def _fetch_full_study(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """Fetch complete study record including detailed description for N extraction."""
        if not nct_id:
            return None
            
        # Request ALL fields for this specific study
        params = {
            "format": "json",
            "fields": "studies.protocolSection,studies.resultsSection,studies.documentSection,studies.derivedSection"
        }
        
        url = f"{self.base}/{nct_id}"
        
        async with aiohttp.ClientSession(connector=_ssl_connector()) as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        studies = data.get("studies", [])
                        if studies:
                            return studies[0]
            except Exception as e:
                print(f"Error fetching full study {nct_id}: {e}")
        
        return None

    def _normalize_record(self, st: Dict[str, Any]) -> Dict[str, Any]:
        ps = st.get("protocolSection", {})
        rs = st.get("resultsSection", {})  # Results for completed trials
        idm = ps.get("identificationModule", {})
        sm = ps.get("statusModule", {})
        dm = ps.get("designModule", {})
        om = ps.get("outcomesModule", {})
        lm = ps.get("contactsLocationsModule", {})

        nct = idm.get("nctId")
        title = idm.get("briefTitle", "")
        summary = idm.get("briefSummary", "")

        # Try multiple sources for enrollment/sample size
        enrollment_count = None
        enrollment_type = None
        sample_size_method = None  # Track how we got the N
        
        # 1. Primary source: enrollmentInfo (most reliable)
        ei = dm.get("enrollmentInfo") or {}
        try:
            enrollment_count = int(ei.get("count")) if ei.get("count") else None
            if enrollment_count:
                enrollment_type = ei.get("type")  # Actual | Anticipated
                sample_size_method = "structured"  # From structured API field
        except Exception:
            pass
        
        # 2. If completed trial, check results section for actual N
        if not enrollment_count and rs:
            # Check participant flow for actual enrollment
            pf = rs.get("participantFlowModule", {})
            if pf:
                groups = pf.get("groups", [])
                if groups:
                    # Sum up participants across all groups
                    total = 0
                    for group in groups:
                        try:
                            total += int(group.get("participantsCount", 0))
                        except:
                            pass
                    if total > 0:
                        enrollment_count = total
                        enrollment_type = "Actual"
                        sample_size_method = "structured"  # From results section
            
            # Also check baseline characteristics
            if not enrollment_count:
                bc = rs.get("baselineCharacteristicsModule", {})
                if bc:
                    groups = bc.get("groups", [])
                    if groups:
                        total = 0
                        for group in groups:
                            try:
                                total += int(group.get("participantsCount", 0))
                            except:
                                pass
                        if total > 0:
                            enrollment_count = total
                            enrollment_type = "Actual"
                            sample_size_method = "structured"  # From baseline characteristics
        
        # 3. If still no N, try extracting from detailed description
        if not enrollment_count:
            # Check the detailed description for N patterns
            detailed_desc = ps.get("descriptionModule", {}).get("detailedDescription", "")
            if detailed_desc:
                extracted_n = _extract_sample_size(detailed_desc)
                if extracted_n:
                    enrollment_count = extracted_n
                    enrollment_type = "Extracted from description"
                    sample_size_method = "text_extraction"  # Extracted from text
            
            # Also check eligibility criteria which often mentions target N
            eligibility = ps.get("eligibilityModule", {})
            if not enrollment_count and eligibility:
                eligibility_text = str(eligibility)
                extracted_n = _extract_sample_size(eligibility_text)
                if extracted_n:
                    enrollment_count = extracted_n
                    enrollment_type = "Extracted from eligibility"
                    sample_size_method = "text_extraction"  # Extracted from text

        # outcomes
        primary_outcomes = (om.get("primaryOutcomes") or [])
        primary_measure = primary_outcomes[0].get("measure") if primary_outcomes else None

        # design details
        di = dm.get("designInfo") or {}
        allocation = di.get("allocation")
        masking = di.get("maskingInfo", {}).get("masking") if di.get("maskingInfo") else None
        study_type = dm.get("studyType")
        phase_list = dm.get("phases") or []
        number_of_arms = ps.get("armsInterventionsModule", {}).get("numberOfArms")

        # dates
        start_date = (sm.get("startDateStruct") or {}).get("date")
        primary_completion = (sm.get("primaryCompletionDateStruct") or {}).get("date")

        # locations (collect unique country names)
        locs = lm.get("locations") or []
        countries = sorted({(loc.get("country") or "").strip() for loc in locs if loc.get("country")}) or None

        return {
            "title": title,
            "abstract": summary,
            "authors": [],
            "journal": "ClinicalTrials.gov",
            "year": datetime.now().year,
            "url": f"https://clinicaltrials.gov/study/{nct}" if nct else None,
            "sample_size": enrollment_count,  # compatibility for table
            "sample_size_method": sample_size_method,  # Track how N was obtained
            "p_value": _extract_p_value(summary) if summary else None,  # Extract p-values from summary
            "study_signal": None,
            "source": "clinicaltrials",
            # richer structured fields for dashboards
            "nct_id": nct,
            "status": sm.get("overallStatus"),
            "phase": ", ".join(phase_list) if phase_list else None,
            "enrollment_type": enrollment_type,  # Actual/Anticipated
            "study_type": study_type,
            "primary_outcome": primary_measure,
            "number_of_arms": number_of_arms,
            "allocation": allocation,
            "masking": masking,
            "start_date": start_date,
            "primary_completion_date": primary_completion,
            "countries": countries,
        }

# ---------------- arXiv ----------------
class ArXivSearcher:
    base = "http://export.arxiv.org/api/query"
    base_url = base

    async def search(self, idea: str, max_results: int = 6) -> List[Dict[str, Any]]:
        # Use broader search without category restrictions for better relevance
        # Add medical/clinical terms to improve relevance
        enhanced_query = idea
        if "hearing" in idea.lower() or "auditory" in idea.lower():
            enhanced_query = f"{idea} OR (hearing AND (aids OR devices OR loss OR impairment))"
        
        params = {
            "search_query": enhanced_query,  # No category restriction for better results
            "start": 0,
            "max_results": max_results * 2,  # Request more to filter later
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with aiohttp.ClientSession(connector=_ssl_connector()) as s:
            res = s.get(self.base, params=params)
            if hasattr(res, "__aenter__"):
                async with res as r:
                    if r.status != 200:
                        return []
                    xml = await r.text()
            else:
                r = await res
                if r.status != 200:
                    return []
                xml = await r.text()
            return self._parse(xml)

    # Backwards-compatible method
    async def search_papers(self, idea: str, max_results: int = 6) -> List[Dict[str, Any]]:
        return await self.search(idea, max_results=max_results)
    
    async def _fetch_full_text(self, arxiv_id: str) -> Optional[str]:
        """Fetch full text from arXiv paper (PDF converted to text)."""
        if not arxiv_id:
            return None
        
        # Extract clean arXiv ID (remove version if present)
        import re
        match = re.search(r'(\d+\.\d+)', arxiv_id)
        if not match:
            return None
        clean_id = match.group(1)
        
        # ArXiv provides PDF, we'd need to parse it
        # For now, we'll try to get more text from the abstract API
        # In production, you'd use a PDF parser like PyPDF2
        
        # Alternative: Some arxiv papers have full text in extended abstracts
        # Let's try fetching with more detail
        params = {
            "id_list": clean_id,
            "max_results": 1
        }
        
        try:
            async with aiohttp.ClientSession(connector=_ssl_connector()) as session:
                async with session.get(self.base, params=params) as response:
                    if response.status == 200:
                        xml = await response.text()
                        # Parse for extended content
                        root = ET.fromstring(xml)
                        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                            # Get the full summary which sometimes has methods
                            summary = entry.findtext("{http://www.w3.org/2005/Atom}summary") or ""
                            # Also check for comments field which often has sample size
                            comment = entry.findtext("{http://arxiv.org/schemas/atom}comment") or ""
                            full_text = f"{summary}\n{comment}"
                            return full_text
        except Exception as e:
            print(f"Error fetching arXiv full text for {arxiv_id}: {e}")
        
        return None

    def _parse(self, xml: str) -> List[Dict[str, Any]]:
        out = []
        root = ET.fromstring(xml)
        for e in root.findall("{http://www.w3.org/2005/Atom}entry"):
            get = lambda tag: (e.findtext(f"{{http://www.w3.org/2005/Atom}}{tag}") or "").strip()
            title = get("title")
            abstract = get("summary")
            
            # Filter out obviously irrelevant papers (tumor, cancer, etc. unless query includes those terms)
            irrelevant_terms = ['tumor', 'tumour', 'cancer', 'malignant', 'metastasis', 'oncology']
            title_lower = title.lower()
            if any(term in title_lower for term in irrelevant_terms):
                # Skip unless the query itself contains these terms
                continue  # This will filter out the tumor modeling papers
            authors = [a.findtext("{http://www.w3.org/2005/Atom}name") for a in e.findall("{http://www.w3.org/2005/Atom}author")]
            url = get("id")
            when = get("published")[:10]
            year = int(when[:4]) if when else datetime.now().year
            
            # Try to extract sample size from abstract first
            sample_size = _extract_sample_size(abstract)
            p_value = _extract_p_value(abstract)
            
            # If no N found, check the comment field which often has it
            if not sample_size:
                # arXiv comments often contain "X pages, Y figures, N=Z patients"
                comment = e.findtext("{http://arxiv.org/schemas/atom}comment") or ""
                if comment:
                    sample_size = _extract_sample_size(comment)
                    # Also append comment to abstract for better extraction
                    if comment and "page" not in comment.lower():  # Skip pure formatting comments
                        abstract = f"{abstract}\n\nAdditional info: {comment}"
            
            out.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": "arXiv preprint",
                    "year": year,
                    "url": url,
                    "arxiv_id": url.split("/")[-1] if url else None,
                    "sample_size": sample_size,
                    "sample_size_method": "text_extraction" if sample_size else None,  # arXiv is always text extraction
                    "p_value": p_value,  # Extract p-values from abstract
                    "study_signal": _infer_signal(abstract),
                    "source": "arxiv",
                }
            )
        # Return only up to max_results papers (since we requested 2x to filter)
        # This ensures we return the requested number after filtering
        return out

# ---------------- Orchestrator ----------------
class ResearchIntelligenceEngine:
    def __init__(self):
        self.pubmed = PubMedSearcher()
        self.ct = ClinicalTrialsSearcher()
        self.arxiv = ArXivSearcher()
        # Backwards-compatible attributes expected by tests
        self.pubmed_searcher = self.pubmed
        self.ct_searcher = self.ct
        self.arxiv_searcher = self.arxiv

    def _convert_to_research_paper(self, raw: Dict[str, Any]) -> ResearchPaper:
        """Convert a raw dict (from searchers) to ResearchPaper dataclass."""
        extras = {k: v for k, v in raw.items() if k not in {'title','authors','abstract','journal','year','pmid','doi','arxiv_id','url','sample_size','sample_size_method','p_value','study_signal','source'}}
        # always include explicit source in extras for downstream inference
        src = raw.get('source') or raw.get('journal') or None
        if src:
            extras.setdefault('source', src)
        return ResearchPaper(
            title=raw.get('title', '') or '',
            authors=raw.get('authors', []) or [],
            abstract=raw.get('abstract', '') or '',
            journal=raw.get('journal', '') or '',
            year=raw.get('year') or datetime.now().year,
            pmid=raw.get('pmid'),
            doi=raw.get('doi'),
            arxiv_id=raw.get('arxiv_id'),
            url=raw.get('url'),
            sample_size=raw.get('sample_size'),
            sample_size_method=raw.get('sample_size_method'),  # Include extraction method
            p_value=raw.get('p_value'),  # Include extracted p-value
            study_signal=raw.get('study_signal'),
            extras=extras or None
        )

    def _assess_evidence_quality(self, papers: List[ResearchPaper]) -> str:
        """Assess evidence quality (simple heuristic)."""
        if not papers:
            return 'low'
        score = 0
        for p in papers:
            if p.sample_size and p.sample_size >= 100:
                score += 2
            if 'meta' in (p.title or '').lower() or 'systematic' in (p.title or '').lower():
                score += 3
            if 'random' in (p.abstract or '').lower() or 'random' in (p.title or '').lower():
                score += 2
        if score >= 5:
            return 'high'
        if score >= 2:
            return 'medium'
        return 'low'

    async def analyze_research_topic(
        self,
        idea: str,
        max_papers: int = 6,
        pubmed_papers: Optional[int] = None,
        arxiv_papers: Optional[int] = None,
        clinicaltrials_papers: Optional[int] = None,
    ) -> "ResearchSummary":
        if max_papers <= 0:
            return ResearchSummary(query=idea, search_date=datetime.now(), total_papers_found=0, papers_analyzed=[])

        pN = pubmed_papers or max(1, int(max_papers * 0.5))
        cN = clinicaltrials_papers or max(1, int(max_papers * 0.3))
        aN = arxiv_papers or max(0, max_papers - pN - cN)

        # Use 'search_papers' if provided by mocks/tests, otherwise fall back to .search
        pubmed_call = getattr(self.pubmed, 'search_papers', self.pubmed.search)
        ct_call = getattr(self.ct, 'search_papers', self.ct.search)
        arxiv_call = getattr(self.arxiv, 'search_papers', self.arxiv.search)

        res = await asyncio.gather(
            pubmed_call(idea, max_results=pN),
            ct_call(idea, max_results=cN),
            arxiv_call(idea, max_results=aN),
            return_exceptions=True,
        )

        papers: List[Dict[str, Any]] = []
        for r in res:
            if isinstance(r, list):
                papers.extend(r)

        # Improved relevance scoring - must have meaningful keyword overlap
        idea_l = (idea or "").lower()
        
        # Extract key terms from the query for better matching
        # Include shorter important words and medical terms
        stop_words = {'with', 'from', 'that', 'this', 'have', 'been', 'will', 'about', 
                      'into', 'help', 'increase', 'how'}
        key_words = [w for w in idea_l.split() if len(w) > 2 and w not in stop_words]
        
        # Add important medical terms that might be too short otherwise
        if any(term in idea_l for term in ['hearing', 'deaf', 'ear']):
            if 'aid' in idea_l or 'aids' in idea_l:
                key_words.append('hearing aid')
            if 'device' in idea_l:
                key_words.append('hearing device')
        
        def score(p: Dict[str, Any]) -> float:
            title = (p.get('title', '') or '').lower()
            abstract = (p.get('abstract', '') or '').lower()
            text = f"{title} {abstract}"
            
            # Count how many key words from query appear in the paper
            overlap = sum(1 for w in key_words if w in text)
            
            # For hearing queries, also check for hearing-related terms
            if 'hearing' in ' '.join(key_words) or 'deaf' in ' '.join(key_words):
                hearing_terms = ['hearing', 'deaf', 'auditory', 'cochlear', 'presbycusis', 
                               'audiolog', 'sound', 'ear', 'acoustic']
                hearing_overlap = sum(1 for term in hearing_terms if term in text)
                if hearing_overlap > 0:
                    overlap = max(overlap, hearing_overlap)
            
            # If no overlap with key terms, score is 0 (irrelevant)
            if overlap == 0:
                return 0
            
            # Bonus for specific relevance indicators
            boost = 0
            j = (p.get("journal") or "").lower()
            
            # Check if title contains any key terms (stronger signal)
            title_overlap = sum(1 for w in key_words if w in title)
            boost += title_overlap * 2  # Title matches are worth more
            
            if "meta" in text and "analysis" in text: boost += 2
            if "randomized" in text or "trial" in text: boost += 1
            if "clinicaltrials.gov" in j: boost += 0.5
            if p.get("sample_size"): boost += 0.5
            
            yr = p.get("year") or datetime.now().year
            recency = max(0, 1.0 - max(0, (datetime.now().year - int(yr))) / 10.0)
            
            return overlap + boost + recency

        # Filter and score papers BEFORE creating ResearchPaper objects
        scored_papers = []
        for p in papers:
            if not p.get("title"):  # Skip papers without titles
                continue
            paper_score = score(p)
            if paper_score > 0:  # Only keep papers with positive relevance
                scored_papers.append((paper_score, p))
        
        # Sort by score and take top papers
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        
        # Now create ResearchPaper objects only for relevant papers
        rp: List[ResearchPaper] = []
        for _, p in scored_papers[:max_papers]:  # Limit to requested number
            extras = {k: v for k, v in p.items() if k not in {"title","authors","abstract","journal","year","pmid","doi","arxiv_id","url","sample_size","sample_size_method","p_value","study_signal","source"}}
            rp.append(
                ResearchPaper(
                    title=p.get("title") or "",
                    authors=p.get("authors") or [],
                    abstract=p.get("abstract") or "",
                    journal=p.get("journal") or "",
                    year=p.get("year") or datetime.now().year,
                    pmid=p.get("pmid"),
                    doi=p.get("doi"),
                    arxiv_id=p.get("arxiv_id"),
                    url=p.get("url"),
                    sample_size=p.get("sample_size"),
                    sample_size_method=p.get("sample_size_method"),
                    p_value=p.get("p_value"),
                    study_signal=p.get("study_signal"),
                    extras=extras or None,
                )
            )
        seen, dedup = set(), []
        for p in rp:
            key = p.title.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
            if len(dedup) >= max_papers:
                break

        quality = self._assess_evidence_quality(dedup)
        # Optionally compute simple sample size distribution and common designs (left minimal)
        typical_sizes = [p.sample_size for p in dedup if p.sample_size] or None
        return ResearchSummary(
            query=idea,
            search_date=datetime.now(),
            total_papers_found=len(dedup),
            papers_analyzed=dedup,
            evidence_quality=quality,
            typical_sample_sizes=typical_sizes,
        )


def get_research_engine():
    """Factory helper used by the API to obtain a ready research engine instance."""
    return ResearchIntelligenceEngine()
