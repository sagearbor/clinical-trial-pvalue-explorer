"""
Research Intelligence Module - Stage 1 LLM Pipeline

This module implements the first stage of the enhanced two-stage LLM pipeline:
1. Takes user study description
2. Searches PubMed, Google Scholar, and arXiv for relevant research
3. Extracts quantitative data (effect sizes, sample sizes, outcomes)
4. Summarizes evidence quality and provides structured research summary
5. Passes enriched data to Stage 2 LLM for statistical design recommendations

Key Components:
- PubMedSearcher: Interface to PubMed API
- ScholarSearcher: Interface to Google Scholar (via SerpAPI or similar)
- ArXivSearcher: Interface to arXiv API
- ResearchSummarizer: LLM-based research synthesis
- EvidenceExtractor: Quantitative data extraction from abstracts/papers
"""

import os
import json
import asyncio
import aiohttp
import ssl
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Structured representation of a research paper with extracted data."""
    title: str
    authors: List[str]
    abstract: str
    journal: str
    year: int
    pmid: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    
    # Extracted quantitative data
    sample_size: Optional[int] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None  # 'cohens_d', 'odds_ratio', etc.
    p_value: Optional[float] = None
    confidence_interval: Optional[str] = None
    study_design: Optional[str] = None
    intervention_type: Optional[str] = None
    outcome_measure: Optional[str] = None
    
    # Quality indicators
    is_rct: bool = False
    is_meta_analysis: bool = False
    is_systematic_review: bool = False
    quality_score: float = 0.0  # 0-10 scale
    
    # Study outcomes
    study_signal: str = "Unknown"  # "Positive", "Negative", "Mixed", "Unclear", "Unknown"

@dataclass 
class ResearchSummary:
    """Comprehensive summary of research evidence for a topic."""
    query: str
    search_date: datetime
    total_papers_found: int
    papers_analyzed: List[ResearchPaper]
    
    # Synthesized insights
    evidence_quality: str  # 'high', 'medium', 'low'
    effect_size_range: Tuple[float, float]  # (min, max) observed
    typical_sample_sizes: List[int]
    common_study_designs: List[str]
    recommended_outcome_measures: List[str]
    
    # Uncertainty assessment
    effect_heterogeneity: str  # 'low', 'medium', 'high'
    publication_bias_risk: str  # 'low', 'medium', 'high'
    temporal_trends: str  # description of how effects changed over time
    
    # LLM-generated insights
    key_findings: str
    methodological_considerations: str
    sample_size_rationale: str
    power_recommendations: str


class PubMedSearcher:
    """Interface to PubMed E-utilities API for biomedical literature search."""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.email = os.getenv("PUBMED_EMAIL", "researcher@example.com")
        self.api_key = os.getenv("PUBMED_API_KEY")  # Optional but recommended
        
    async def search_papers(self, query: str, max_results: int = 20, years_back: int = 5) -> List[Dict]:
        """
        Search PubMed for papers matching the query.
        
        Args:
            query: Search query (will be enhanced with filters)
            max_results: Maximum papers to return
            years_back: Only include papers from last N years
            
        Returns:
            List of paper metadata dictionaries
        """
        # Enhance query with filters for higher quality studies
        enhanced_query = self._enhance_pubmed_query(query, years_back)
        
        try:
            # Step 1: Search for PMIDs
            pmids = await self._search_pmids(enhanced_query, max_results)
            if not pmids:
                return []
            
            # Step 2: Fetch detailed paper information
            papers = await self._fetch_paper_details(pmids)
            return papers
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
    
    def _enhance_pubmed_query(self, query: str, years_back: int) -> str:
        """Enhance query with PubMed-specific filters and MeSH terms."""
        current_year = datetime.now().year
        start_year = current_year - years_back
        
        # Add date range and study type filters
        enhanced = f"({query}) AND ({start_year}[PDAT]:{current_year}[PDAT])"
        
        # Prioritize high-quality study types
        study_types = [
            "randomized controlled trial[pt]",
            "meta-analysis[pt]", 
            "systematic review[pt]",
            "clinical trial[pt]"
        ]
        
        enhanced += f" AND ({' OR '.join(study_types)})"
        
        return enhanced
    
    async def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search PubMed and return list of PMIDs."""
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        # Create SSL context that's more permissive
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('esearchresult', {}).get('idlist', [])
                else:
                    logger.warning(f"PubMed search failed: {response.status}")
                    return []
    
    async def _fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for a list of PMIDs."""
        if not pmids:
            return []
            
        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        # Create SSL context that's more permissive
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(fetch_url, params=params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    return self._parse_pubmed_xml(xml_data)
                else:
                    logger.warning(f"PubMed fetch failed: {response.status}")
                    return []
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict]:
        """Parse PubMed XML response into structured paper data."""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._extract_paper_from_xml(article)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            
        return papers
    
    def _extract_paper_from_xml(self, article_xml) -> Optional[Dict]:
        """Extract paper information from a single PubMed article XML element."""
        try:
            # Basic metadata
            title_elem = article_xml.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "Unknown Title"
            
            # Abstract
            abstract_elem = article_xml.find('.//AbstractText')
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Authors
            authors = []
            for author in article_xml.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName') or author.find('FirstName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Journal and year
            journal_elem = article_xml.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else "Unknown Journal"
            
            year_elem = article_xml.find('.//PubDate/Year')
            year = int(year_elem.text) if year_elem is not None else datetime.now().year
            
            # PMID
            pmid_elem = article_xml.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # DOI
            doi = None
            for article_id in article_xml.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            # Publication types (for quality assessment)
            pub_types = []
            for pub_type in article_xml.findall('.//PublicationType'):
                pub_types.append(pub_type.text.lower())
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'journal': journal,
                'year': year,
                'pmid': pmid,
                'doi': doi,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                'publication_types': pub_types,
                'source': 'pubmed'
            }
            
        except Exception as e:
            logger.error(f"Error extracting paper data: {e}")
            return None


class ArXivSearcher:
    """Interface to arXiv API for preprint and academic paper search."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum papers to return
            
        Returns:
            List of paper metadata dictionaries
        """
        # Focus on relevant categories for clinical research - more restrictive
        categories = ['q-bio.QM', 'q-bio.PE', 'q-bio.TO', 'stat.AP']  # Quantitative Methods, Populations and Evolution, Tissues and Organs, Applications
        category_filter = ' OR '.join([f'cat:{cat}' for cat in categories])
        
        # Add medical keywords to filter
        medical_filter = 'clinical OR medical OR health OR intervention OR trial OR study'
        enhanced_query = f"({query}) AND ({category_filter}) AND ({medical_filter})"
        
        params = {
            'search_query': enhanced_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            # Create SSL context that's more permissive
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        return self._parse_arxiv_xml(xml_data)
                    else:
                        logger.warning(f"arXiv search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[Dict]:
        """Parse arXiv XML response into structured paper data."""
        papers = []
        
        try:
            root = ET.fromstring(xml_data)
            # Remove namespace for easier parsing
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}')[1]
            
            for entry in root.findall('entry'):
                paper = self._extract_arxiv_paper(entry)
                if paper:
                    papers.append(paper)
                    
        except ET.ParseError as e:
            logger.error(f"arXiv XML parsing error: {e}")
            
        return papers
    
    def _extract_arxiv_paper(self, entry_xml) -> Optional[Dict]:
        """Extract paper information from arXiv entry XML."""
        try:
            title_elem = entry_xml.find('title')
            title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
            
            summary_elem = entry_xml.find('summary')
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Authors
            authors = []
            for author in entry_xml.findall('author'):
                name_elem = author.find('name')
                if name_elem is not None:
                    authors.append(name_elem.text)
            
            # arXiv ID and URL
            id_elem = entry_xml.find('id')
            url = id_elem.text if id_elem is not None else None
            arxiv_id = url.split('/')[-1] if url else None
            
            # Publication date
            published_elem = entry_xml.find('published')
            year = datetime.now().year
            if published_elem is not None:
                try:
                    pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                    year = pub_date.year
                except:
                    pass
            
            # Categories
            categories = []
            for category in entry_xml.findall('category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            return {
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'journal': 'arXiv preprint',
                'year': year,
                'arxiv_id': arxiv_id,
                'url': url,
                'categories': categories,
                'source': 'arxiv'
            }
            
        except Exception as e:
            logger.error(f"Error extracting arXiv paper: {e}")
            return None


class ClinicalTrialsSearcher:
    """Interface to ClinicalTrials.gov API for clinical trial data."""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
    
    async def search_trials(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search ClinicalTrials.gov for trials matching the query.
        
        Args:
            query: Search query
            max_results: Maximum trials to return
            
        Returns:
            List of trial metadata dictionaries
        """
        # Use new API v2 parameters
        params = {
            'query.term': query,  # General search terms
            'pageSize': max_results,
            'format': 'json'
        }
        
        try:
            # Create SSL context that's more permissive
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_trials_response(data)
                    else:
                        logger.warning(f"ClinicalTrials.gov search failed: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"ClinicalTrials.gov search error: {e}")
            return []
    
    def _parse_trials_response(self, data: Dict) -> List[Dict]:
        """Parse ClinicalTrials.gov API v2 JSON response into structured trial data."""
        trials = []
        
        try:
            # API v2 format: data -> studies array
            studies_list = data.get('studies', [])
            
            for study in studies_list:
                trial = self._extract_trial_info_v2(study)
                if trial:
                    trials.append(trial)
                    
        except Exception as e:
            logger.error(f"Error parsing ClinicalTrials.gov v2 response: {e}")
            
        return trials
    
    def _extract_trial_info_v2(self, study: Dict) -> Optional[Dict]:
        """Extract trial information from API v2 study record."""
        try:
            protocol_section = study.get('protocolSection', {})
            identification_module = protocol_section.get('identificationModule', {})
            status_module = protocol_section.get('statusModule', {})
            design_module = protocol_section.get('designModule', {})
            arms_interventions_module = protocol_section.get('armsInterventionsModule', {})
            outcomes_module = protocol_section.get('outcomesModule', {})
            eligibility_module = protocol_section.get('eligibilityModule', {})
            
            # Extract enrollment count
            sample_size = None
            if design_module.get('enrollmentInfo'):
                try:
                    sample_size = int(design_module['enrollmentInfo'].get('count', 0))
                except:
                    pass
            
            # Extract completion date and year
            completion_date = status_module.get('primaryCompletionDateStruct', {}).get('date', '')
            year = datetime.now().year
            if completion_date:
                try:
                    parsed_date = datetime.strptime(completion_date, '%Y-%m-%d')
                    year = parsed_date.year
                except:
                    try:
                        year = int(completion_date[:4])
                    except:
                        pass
            
            # Extract interventions
            interventions = arms_interventions_module.get('interventions', [])
            intervention_names = [i.get('name', '') for i in interventions if i.get('name')]
            intervention_str = ', '.join(intervention_names[:3])  # First 3 interventions
            
            # Extract conditions
            conditions = identification_module.get('conditions', [])
            condition_str = ', '.join(conditions[:3]) if conditions else ''  # First 3 conditions
            
            # Extract outcomes
            primary_outcomes = outcomes_module.get('primaryOutcomes', [])
            primary_outcome_str = primary_outcomes[0].get('measure', '') if primary_outcomes else ''
            
            return {
                'title': identification_module.get('briefTitle', ''),
                'nct_id': identification_module.get('nctId', ''),
                'summary': identification_module.get('briefSummary', ''),
                'condition': condition_str,
                'intervention': intervention_str,
                'phase': design_module.get('phases', ['N/A'])[0] if design_module.get('phases') else 'N/A',
                'study_type': design_module.get('studyType', ''),
                'allocation': design_module.get('designInfo', {}).get('allocation', ''),
                'sample_size': sample_size,
                'primary_outcome': primary_outcome_str,
                'completion_date': completion_date,
                'year': year,
                'status': status_module.get('overallStatus', ''),
                'url': f"https://clinicaltrials.gov/study/{identification_module.get('nctId')}" if identification_module.get('nctId') else None,
                'source': 'clinicaltrials.gov'
            }
            
        except Exception as e:
            logger.error(f"Error extracting v2 trial data: {e}")
            return None
    
    def _extract_trial_info(self, study: Dict) -> Optional[Dict]:
        """Extract trial information from a single study record."""
        try:
            def get_field(field_name: str) -> str:
                """Helper to safely extract field values."""
                field_data = study.get(field_name, [])
                if isinstance(field_data, list) and field_data:
                    return field_data[0] if field_data[0] else ""
                return ""
            
            # Extract enrollment count and try to convert to integer
            enrollment_str = get_field('EnrollmentCount')
            sample_size = None
            if enrollment_str:
                try:
                    sample_size = int(enrollment_str)
                except:
                    pass
            
            # Extract completion date and convert to year
            completion_date = get_field('CompletionDate')
            year = datetime.now().year
            if completion_date:
                try:
                    # Try different date formats
                    for fmt in ['%B %Y', '%B %d, %Y', '%Y-%m-%d', '%m/%d/%Y']:
                        try:
                            parsed_date = datetime.strptime(completion_date, fmt)
                            year = parsed_date.year
                            break
                        except:
                            continue
                except:
                    pass
            
            return {
                'title': get_field('BriefTitle') or get_field('OfficialTitle'),
                'nct_id': get_field('NCTId'),
                'summary': get_field('BriefSummary') or get_field('DetailedDescription'),
                'condition': get_field('Condition'),
                'intervention': get_field('InterventionName'),
                'phase': get_field('Phase'),
                'study_type': get_field('StudyType'),
                'allocation': get_field('DesignAllocation'),
                'sample_size': sample_size,
                'enrollment_type': get_field('EnrollmentType'),
                'primary_outcome': get_field('PrimaryOutcomeMeasure'),
                'secondary_outcome': get_field('SecondaryOutcomeMeasure'),
                'completion_date': completion_date,
                'year': year,
                'status': get_field('OverallStatus'),
                'url': f"https://clinicaltrials.gov/study/{get_field('NCTId')}" if get_field('NCTId') else None,
                'source': 'clinicaltrials.gov'
            }
            
        except Exception as e:
            logger.error(f"Error extracting trial data: {e}")
            return None


class ResearchIntelligenceEngine:
    """
    Main orchestrator for the research intelligence pipeline.
    Coordinates search across multiple databases and synthesizes findings.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the research intelligence engine."""
        self.pubmed_searcher = PubMedSearcher()
        self.arxiv_searcher = ArXivSearcher()
        self.clinicaltrials_searcher = ClinicalTrialsSearcher()
    
    def _enhance_query_for_clinical_research(self, query: str) -> str:
        """
        Enhance search query to be more targeted for clinical/medical research.
        Extract key medical terms and add relevant MeSH-like keywords.
        """
        import re
        
        # Convert to lowercase for processing
        query_lower = query.lower()
        
        # Medical/clinical keywords that should be preserved and emphasized
        clinical_terms = {
            'obesity': ['obesity', 'overweight', 'BMI', 'weight loss', 'body mass index'],
            'diet': ['diet', 'dietary', 'nutrition', 'nutritional', 'food intake'],
            'smartphone': ['smartphone', 'mobile app', 'digital health', 'mHealth', 'mobile intervention'],
            'reminder': ['reminder', 'notification', 'prompt', 'behavioral intervention'],
            'epigenetic': ['epigenetic', 'DNA methylation', 'gene expression'],
            'diabetes': ['diabetes', 'diabetic', 'glucose', 'insulin', 'glycemic'],
            'hypertension': ['hypertension', 'blood pressure', 'cardiovascular'],
            'intervention': ['intervention', 'treatment', 'therapy', 'program'],
            'randomized': ['randomized', 'RCT', 'controlled trial', 'clinical trial'],
            'pre-post': ['pre-post', 'before-after', 'longitudinal', 'prospective'],
            'control': ['control group', 'comparison', 'placebo']
        }
        
        # Find relevant clinical terms in the query
        found_terms = []
        for main_term, related_terms in clinical_terms.items():
            if any(term in query_lower for term in related_terms):
                found_terms.extend(related_terms[:2])  # Add top 2 related terms
        
        # Create enhanced query
        if found_terms:
            # Use the most relevant terms
            core_terms = ' '.join(found_terms[:4])  # Limit to avoid overly complex queries
            enhanced = f"{core_terms} clinical trial OR intervention OR study"
        else:
            # Fallback: add general clinical research terms
            enhanced = f"{query} clinical trial OR intervention OR randomized OR study"
        
        return enhanced
    
    def _extract_sample_size_from_text(self, text: str) -> int:
        """
        Extract sample size from abstract or title text using regex patterns.
        """
        if not text:
            return None
            
        # Common patterns for sample size mentions
        patterns = [
            r'[Nn]\s*=\s*(\d+)',  # N=123
            r'[Nn]\s*:\s*(\d+)',  # N: 123  
            r'(\d+)\s+participants?',  # 123 participants
            r'(\d+)\s+subjects?',  # 123 subjects
            r'(\d+)\s+patients?',  # 123 patients
            r'(\d+)\s+individuals?',  # 123 individuals
            r'sample\s+of\s+(\d+)',  # sample of 123
            r'cohort\s+of\s+(\d+)',  # cohort of 123
            r'enrolled\s+(\d+)',  # enrolled 123
            r'recruited\s+(\d+)',  # recruited 123
            r'total\s+of\s+(\d+)\s+(?:participants?|subjects?|patients?)',  # total of 123 participants
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Get the first match and convert to int
                try:
                    sample_size = int(matches[0])
                    # Sanity check: reasonable sample size for clinical research
                    if 5 <= sample_size <= 100000:  # Between 5 and 100k participants
                        return sample_size
                except ValueError:
                    continue
                    
        return None
    
    def _detect_study_signal(self, title: str, abstract: str) -> str:
        """
        Detect whether a study found positive, negative, or mixed results.
        Important for identifying underpowered studies with null results.
        """
        text = (title + ' ' + abstract).lower() if title and abstract else ''
        if not text:
            return "Unknown"
        
        # Positive signal indicators
        positive_patterns = [
            r'significant\s+(?:improvement|reduction|decrease|effect|association|difference)',
            r'effective\s+(?:intervention|treatment|approach)',
            r'reduced\s+(?:bmi|weight|obesity|symptoms)',
            r'improved\s+(?:outcomes|health|weight\s+loss)',
            r'beneficial\s+effect',
            r'positive\s+(?:effect|association|outcome)',
            r'successfully\s+(?:reduced|improved|treated)',
            r'significant\s+weight\s+loss',
            r'statistically\s+significant.*(?:p\s*[<≤]\s*0\.05)',
            r'p\s*[<≤]\s*0\.0[0-5]',
            r'strongly\s+associated',
            r'dose[\-\s]response\s+relationship'
        ]
        
        # Negative signal indicators  
        negative_patterns = [
            r'no\s+significant\s+(?:difference|effect|improvement|association|change)',
            r'not\s+significant',
            r'failed\s+to\s+(?:show|demonstrate|find)',
            r'no\s+evidence\s+of',
            r'did\s+not\s+(?:show|demonstrate|result\s+in)',
            r'non[\-\s]significant',
            r'p\s*[>≥]\s*0\.05',
            r'null\s+(?:result|finding|effect)',
            r'inconclusive\s+(?:results|evidence)',
            r'no\s+(?:effect|impact|change|improvement)',
            r'unsuccessful\s+(?:intervention|treatment)',
            r'ineffective\s+(?:treatment|intervention)'
        ]
        
        # Mixed/qualified results indicators
        mixed_patterns = [
            r'modest\s+(?:improvement|effect|reduction)',
            r'limited\s+(?:evidence|effect|improvement)',
            r'mixed\s+(?:results|findings|evidence)',
            r'some\s+evidence',
            r'borderline\s+significant',
            r'trend\s+toward',
            r'marginally\s+significant',
            r'weak\s+(?:association|evidence)',
            r'inconsistent\s+(?:results|findings)'
        ]
        
        # Count matches for each category
        import re
        positive_count = sum(1 for pattern in positive_patterns if re.search(pattern, text))
        negative_count = sum(1 for pattern in negative_patterns if re.search(pattern, text))
        mixed_count = sum(1 for pattern in mixed_patterns if re.search(pattern, text))
        
        # Determine signal based on strongest pattern match
        if positive_count > negative_count and positive_count > mixed_count:
            return "Positive"
        elif negative_count > positive_count and negative_count > mixed_count:
            return "Negative" 
        elif mixed_count > 0:
            return "Mixed"
        else:
            return "Unclear"
        
    async def analyze_research_topic(
        self, 
        query: str, 
        max_papers: int = 30,
        pubmed_papers: int = None,
        arxiv_papers: int = None, 
        clinicaltrials_papers: int = None
    ) -> ResearchSummary:
        """
        Perform comprehensive research analysis for a given topic.
        
        Args:
            query: Research topic or study description
            max_papers: Maximum total papers to analyze
            
        Returns:
            ResearchSummary with synthesized findings and recommendations
        """
        logger.info(f"Starting research analysis for: {query}")
        
        # Enhance query for better medical/clinical relevance
        enhanced_query = self._enhance_query_for_clinical_research(query)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # Use specific counts if provided, otherwise distribute intelligently
        if pubmed_papers is None or arxiv_papers is None or clinicaltrials_papers is None:
            # Default distribution: PubMed gets 40%, ClinicalTrials gets 30%, arXiv gets 30%
            pubmed_papers = pubmed_papers or int(max_papers * 0.4)
            clinicaltrials_papers = clinicaltrials_papers or int(max_papers * 0.3) 
            arxiv_papers = arxiv_papers or (max_papers - pubmed_papers - clinicaltrials_papers)
        
        logger.info(f"Paper allocation: PubMed={pubmed_papers}, arXiv={arxiv_papers}, ClinicalTrials={clinicaltrials_papers}")
        
        # Search multiple databases in parallel with enhanced query
        search_tasks = [
            self.pubmed_searcher.search_papers(enhanced_query, pubmed_papers),
            self.clinicaltrials_searcher.search_trials(enhanced_query, clinicaltrials_papers),
            self.arxiv_searcher.search_papers(enhanced_query, arxiv_papers)
        ]
        
        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Combine results
            all_papers = []
            for result in search_results:
                if isinstance(result, list):
                    all_papers.extend(result)
                else:
                    logger.error(f"Search error: {result}")
            
            logger.info(f"Found {len(all_papers)} papers total")
            
            # Convert to ResearchPaper objects (simplified for now)
            research_papers = [self._convert_to_research_paper(p) for p in all_papers]
            
            # Synthesize findings
            research_summary = await self._synthesize_findings(query, research_papers)
            
            logger.info(f"Research analysis complete. Evidence quality: {research_summary.evidence_quality}")
            return research_summary
            
        except Exception as e:
            logger.error(f"Research analysis error: {e}")
            # Return empty summary on error
            return self._create_empty_summary(query)
    
    def _convert_to_research_paper(self, paper_dict: Dict) -> ResearchPaper:
        """Convert paper dictionary to ResearchPaper object."""
        # Handle ClinicalTrials.gov data differently
        if paper_dict.get('source') == 'clinicaltrials.gov':
            # For clinical trials, use the trial summary as abstract
            abstract = paper_dict.get('summary', '')
            if paper_dict.get('intervention'):
                abstract += f"\nIntervention: {paper_dict.get('intervention')}"
            if paper_dict.get('primary_outcome'):
                abstract += f"\nPrimary Outcome: {paper_dict.get('primary_outcome')}"
            
            # Detect study signal for clinical trials too
            study_signal = self._detect_study_signal(paper_dict.get('title', ''), abstract)
            
            return ResearchPaper(
                title=paper_dict.get('title', ''),
                authors=[],  # Clinical trials don't have traditional authors
                abstract=abstract,
                journal=f"ClinicalTrials.gov ({paper_dict.get('phase', 'N/A')})",
                year=paper_dict.get('year', datetime.now().year),
                sample_size=paper_dict.get('sample_size'),
                url=paper_dict.get('url'),
                # Store trial-specific data
                study_design=paper_dict.get('allocation', 'Unknown'),
                intervention_type=paper_dict.get('intervention', ''),
                outcome_measure=paper_dict.get('primary_outcome', ''),
                study_signal=study_signal
            )
        else:
            # Handle regular papers (PubMed, arXiv)
            abstract = paper_dict.get('abstract', '')
            
            # Extract sample size from abstract if available
            sample_size = self._extract_sample_size_from_text(abstract + ' ' + paper_dict.get('title', ''))
            
            # Detect study signal/effect
            study_signal = self._detect_study_signal(paper_dict.get('title', ''), abstract)
            
            return ResearchPaper(
                title=paper_dict.get('title', ''),
                authors=paper_dict.get('authors', []),
                abstract=abstract,
                journal=paper_dict.get('journal', ''),
                year=paper_dict.get('year', datetime.now().year),
                pmid=paper_dict.get('pmid'),
                doi=paper_dict.get('doi'),
                arxiv_id=paper_dict.get('arxiv_id'),
                url=paper_dict.get('url'),
                sample_size=sample_size,
                study_signal=study_signal
            )
    
    def _create_empty_summary(self, query: str) -> ResearchSummary:
        """Create empty research summary for error cases."""
        return ResearchSummary(
            query=query,
            search_date=datetime.now(),
            total_papers_found=0,
            papers_analyzed=[],
            evidence_quality='low',
            effect_size_range=(0.0, 0.0),
            typical_sample_sizes=[],
            common_study_designs=[],
            recommended_outcome_measures=[],
            effect_heterogeneity='high',
            publication_bias_risk='high',
            temporal_trends='insufficient data',
            key_findings='No research data found',
            methodological_considerations='Unable to assess',
            sample_size_rationale='Default recommendations apply',
            power_recommendations='Use conservative estimates'
        )
    
    async def _synthesize_findings(self, query: str, papers: List[ResearchPaper]) -> ResearchSummary:
        """Synthesize research findings into structured recommendations."""
        
        if not papers:
            return self._create_empty_summary(query)
        
        # Basic analysis - would be enhanced with LLM integration
        high_quality_count = len([p for p in papers if 'randomized' in p.title.lower() or 'meta-analysis' in p.title.lower()])
        
        # Simple evidence quality assessment
        if high_quality_count >= 3:
            evidence_quality = 'high'
        elif high_quality_count >= 1:
            evidence_quality = 'medium'
        else:
            evidence_quality = 'low'
        
        return ResearchSummary(
            query=query,
            search_date=datetime.now(),
            total_papers_found=len(papers),
            papers_analyzed=papers,
            evidence_quality=evidence_quality,
            effect_size_range=(0.0, 0.0),  # Would extract from papers
            typical_sample_sizes=[],  # Would extract from papers
            common_study_designs=['RCT', 'observational'],
            recommended_outcome_measures=[],
            effect_heterogeneity='medium',
            publication_bias_risk='medium',
            temporal_trends='stable',
            key_findings=f"Found {len(papers)} relevant papers ({high_quality_count} high-quality studies)",
            methodological_considerations='Standard considerations apply',
            sample_size_rationale=f'Based on {len(papers)} studies',
            power_recommendations=f'Evidence quality: {evidence_quality}'
        )


# Example usage
async def test_research_intelligence():
    """Test the research intelligence system."""
    engine = ResearchIntelligenceEngine()
    summary = await engine.analyze_research_topic("depression cognitive therapy", max_papers=10)
    print(f"Found {summary.total_papers_found} papers, quality: {summary.evidence_quality}")


if __name__ == "__main__":
    asyncio.run(test_research_intelligence())