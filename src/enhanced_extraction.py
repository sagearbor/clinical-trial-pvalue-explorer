#!/usr/bin/env python3
"""Enhanced N extraction with LLM support, confidence scoring, and rich output"""

import re
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
import aiohttp

# Import LLM modules
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class ExtractionResult:
    """Rich extraction result with confidence and context"""
    sample_size: Optional[int] = None
    confidence: float = 0.0
    method: str = "unknown"  # "regex", "llm", "structured"
    context: str = ""
    section: Optional[str] = None  # Methods, Results, etc.
    alternatives: List[Dict[str, Any]] = None
    per_group: Optional[Dict[str, int]] = None  # treatment/control groups
    validation_flags: List[str] = None  # warnings about unusual values
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.validation_flags is None:
            self.validation_flags = []
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PatternWithConfidence:
    """Regex pattern with associated confidence score"""
    def __init__(self, pattern: str, confidence: float, name: str, priority: bool = False):
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.confidence = confidence
        self.name = name
        self.priority = priority


class EnhancedNExtractor:
    """Enhanced N extraction with multiple methods and confidence scoring"""
    
    # Priority patterns with high confidence
    PRIORITY_PATTERNS = [
        PatternWithConfidence(
            r"[Ss]tudy\s+[Ss]ample:?\s*.*?\([Nn]\s*=\s*([\d,]{2,10})\)", 
            0.95, "study_sample", priority=True
        ),
        PatternWithConfidence(
            r"[Oo]ur\s+sample\s*\([Nn]\s*=\s*([\d,]{2,10})\)", 
            0.95, "our_sample", priority=True
        ),
        PatternWithConfidence(
            r"[Ff]inal\s+sample.*?\([Nn]\s*=\s*([\d,]{2,10})\)", 
            0.90, "final_sample", priority=True
        ),
        PatternWithConfidence(
            r"[Aa]nalyz\w+\s+sample.*?\([Nn]\s*=\s*([\d,]{2,10})\)", 
            0.90, "analyzed_sample", priority=True
        ),
    ]
    
    # Regular patterns with varying confidence
    REGULAR_PATTERNS = [
        PatternWithConfidence(r"\b[Nn]\s*=\s*([\d,]{2,10})\b", 0.85, "n_equals"),
        PatternWithConfidence(r"\([Nn]\s*=\s*([\d,]{2,10})\)", 0.85, "n_parentheses"),
        PatternWithConfidence(r"\btotal\s+of\s+([\d,]{2,10})\b", 0.75, "total_of"),
        PatternWithConfidence(r"\b([\d,]{2,10})\s*(?:were|was)\s*(?:enrolled|included|recruited|randomized)\b", 0.70, "were_enrolled"),
        PatternWithConfidence(r"\b(?:enrolled|included|recruited|randomized)\s*([\d,]{2,10})\b", 0.70, "enrolled_n"),
        PatternWithConfidence(r"\b([\d,]{2,10})\s*(?:patients?|participants?|subjects?)\b", 0.60, "n_patients"),
        PatternWithConfidence(r"\b(?:analyzed|analysed)\s+([\d,]{2,10})\b", 0.65, "analyzed_n"),
        PatternWithConfidence(r"\b([\d,]{2,10})\s*completed\s*the\s*(?:study|trial)\b", 0.70, "n_completed"),
        PatternWithConfidence(r"\bcohort\s+of\s+([\d,]{2,10})\b", 0.65, "cohort_of"),
        PatternWithConfidence(r"\bdata\s+from\s+([\d,]{2,10})\b", 0.60, "data_from"),
        PatternWithConfidence(r"\b([\d,]{2,10})\s*(?:men|women|males|females)\b", 0.55, "gender_specific"),
        PatternWithConfidence(r"\bsample\s+size\s+(?:of\s*)?([\d,]{2,10})\b", 0.80, "sample_size"),
    ]
    
    def __init__(self, llm_provider: str = None):
        self.llm_provider = llm_provider or os.getenv("ACTIVE_LLM_PROVIDER", "gemini").lower()
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client based on provider"""
        self.llm_client = None
        
        if self.llm_provider == "gemini" and genai:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel('gemini-1.5-flash')
        
        elif self.llm_provider == "openai" and openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.llm_client = openai.Client(api_key=api_key)
        
        elif self.llm_provider == "anthropic" and anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.llm_client = anthropic.Client(api_key=api_key)
    
    def extract_with_regex(self, text: str) -> ExtractionResult:
        """Extract N using regex patterns with confidence scoring"""
        if not text:
            return ExtractionResult(method="regex")
        
        # Check priority patterns first
        for pattern_conf in self.PRIORITY_PATTERNS:
            matches = pattern_conf.pattern.findall(text)
            for match in matches:
                try:
                    n = int(match.replace(',', ''))
                    if 10 <= n <= 100000:  # Reasonable range
                        # Find context
                        idx = text.lower().find(match.lower())
                        start = max(0, idx - 30)
                        end = min(len(text), idx + len(match) + 30)
                        context = text[start:end]
                        
                        return ExtractionResult(
                            sample_size=n,
                            confidence=pattern_conf.confidence,
                            method="regex",
                            context=context,
                            section=self._detect_section(text, idx)
                        )
                except:
                    continue
        
        # Try regular patterns, collect all matches
        all_matches = []
        for pattern_conf in self.REGULAR_PATTERNS:
            matches = pattern_conf.pattern.findall(text)
            for match in matches:
                try:
                    n = int(match.replace(',', ''))
                    if 10 <= n <= 100000:
                        idx = text.lower().find(match.lower())
                        start = max(0, idx - 30)
                        end = min(len(text), idx + len(match) + 30)
                        context = text[start:end]
                        
                        all_matches.append({
                            'n': n,
                            'confidence': pattern_conf.confidence,
                            'context': context,
                            'pattern': pattern_conf.name,
                            'section': self._detect_section(text, idx)
                        })
                except:
                    continue
        
        if not all_matches:
            return ExtractionResult(method="regex", confidence=0.0)
        
        # Smart selection from multiple matches
        selected = self._smart_select_n(all_matches, text)
        
        # Create alternatives list
        alternatives = [
            {
                'sample_size': m['n'],
                'confidence': m['confidence'],
                'context': m['context']
            }
            for m in all_matches if m != selected
        ][:3]  # Keep top 3 alternatives
        
        result = ExtractionResult(
            sample_size=selected['n'],
            confidence=selected['confidence'],
            method="regex",
            context=selected['context'],
            section=selected.get('section'),
            alternatives=alternatives
        )
        
        # Add validation flags
        self._validate_n(result, text)
        
        return result
    
    async def extract_with_llm(self, text: str, title: str = "") -> ExtractionResult:
        """Extract N using LLM with confidence scoring"""
        if not self.llm_client or not text:
            return ExtractionResult(method="llm", confidence=0.0)
        
        prompt = f"""Extract the sample size (N) from this research abstract. 

Title: {title}
Abstract: {text}

Instructions:
1. Find the ACTUAL study sample size (not survey distributions or screening numbers)
2. If multiple N values exist, identify the final analyzed sample
3. Provide confidence score (0.0 to 1.0)
4. Extract the context sentence containing N

Return JSON only:
{{
  "sample_size": <integer or null>,
  "confidence": <float 0-1>,
  "context": "<sentence with N>",
  "section": "<Methods/Results/etc or null>",
  "alternatives": [
    {{"n": <int>, "context": "<text>", "confidence": <float>}}
  ],
  "per_group": {{"treatment": <int or null>, "control": <int or null>}},
  "reasoning": "<brief explanation>"
}}"""
        
        try:
            response_text = await self._call_llm(prompt)
            response_data = json.loads(response_text)
            
            result = ExtractionResult(
                sample_size=response_data.get('sample_size'),
                confidence=response_data.get('confidence', 0.0),
                method="llm",
                context=response_data.get('context', ''),
                section=response_data.get('section'),
                alternatives=[
                    {
                        'sample_size': alt['n'],
                        'confidence': alt.get('confidence', 0.5),
                        'context': alt.get('context', '')
                    }
                    for alt in response_data.get('alternatives', [])
                ],
                per_group=response_data.get('per_group')
            )
            
            # Validate LLM output
            self._validate_n(result, text)
            
            return result
            
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return ExtractionResult(method="llm", confidence=0.0)
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM based on provider"""
        if self.llm_provider == "gemini" and self.llm_client:
            response = self.llm_client.generate_content(prompt)
            return response.text
        
        elif self.llm_provider == "openai" and self.llm_client:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "anthropic" and self.llm_client:
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            return response.content[0].text
        
        return "{}"
    
    def _detect_section(self, text: str, position: int) -> Optional[str]:
        """Detect which section of abstract contains the N"""
        # Look for section headers before the position
        text_before = text[:position].lower()
        
        sections = {
            'methods': ['method', 'procedure', 'participant', 'subject'],
            'results': ['result', 'finding', 'outcome'],
            'background': ['background', 'introduction', 'objective'],
            'conclusion': ['conclusion', 'discussion', 'implication']
        }
        
        for section, keywords in sections.items():
            for keyword in keywords:
                if keyword in text_before[-200:]:  # Look in last 200 chars
                    return section
        
        return None
    
    def _smart_select_n(self, matches: List[Dict], full_text: str) -> Dict:
        """Smart selection from multiple N values"""
        if len(matches) == 1:
            return matches[0]
        
        # Prioritize by section
        section_priority = {'methods': 3, 'results': 2, 'background': 0, 'conclusion': 1}
        for match in matches:
            match['section_score'] = section_priority.get(match.get('section', ''), 1)
        
        # Look for final/analyzed keywords
        for match in matches:
            context_lower = match['context'].lower()
            if any(word in context_lower for word in ['final', 'analyzed', 'analysed', 'completed']):
                match['confidence'] *= 1.2  # Boost confidence
            elif any(word in context_lower for word in ['screened', 'contacted', 'invited']):
                match['confidence'] *= 0.7  # Reduce confidence
        
        # Sort by confidence * section_score
        matches.sort(key=lambda x: x['confidence'] * x.get('section_score', 1), reverse=True)
        
        return matches[0]
    
    def _validate_n(self, result: ExtractionResult, text: str) -> None:
        """Add validation flags for unusual N values"""
        if not result.sample_size:
            return
        
        n = result.sample_size
        text_lower = text.lower()
        
        # Check for unusual values
        if 'randomized controlled trial' in text_lower and n < 20:
            result.validation_flags.append("Unusually small N for RCT")
        
        if 'systematic review' in text_lower and n < 100:
            result.validation_flags.append("N might be number of studies, not participants")
        
        if n > 50000:
            result.validation_flags.append("Very large N - verify if population study")
        
        if n == 100 or n == 1000 or n == 10000:
            result.validation_flags.append("Round number - verify if actual or target")
    
    async def extract_combined(self, text: str, title: str = "") -> Dict[str, Any]:
        """Extract using both methods and return combined result"""
        # Run both extractions
        regex_result = self.extract_with_regex(text)
        llm_result = await self.extract_with_llm(text, title)
        
        # Calculate quality scores
        regex_quality = (1 if regex_result.sample_size else 0) * regex_result.confidence
        llm_quality = (1 if llm_result.sample_size else 0) * llm_result.confidence
        
        # Determine which to use as primary
        if llm_quality > regex_quality:
            primary_result = llm_result
            secondary_result = regex_result
            primary_method = "llm"
        else:
            primary_result = regex_result
            secondary_result = llm_result
            primary_method = "regex"
        
        # Check if they differ
        differs = False
        if regex_result.sample_size and llm_result.sample_size:
            differs = regex_result.sample_size != llm_result.sample_size
        
        return {
            'sample_size': primary_result.sample_size,
            'confidence': primary_result.confidence,
            'method': primary_method,
            'context': primary_result.context,
            'section': primary_result.section,
            'alternatives': primary_result.alternatives,
            'per_group': primary_result.per_group,
            'validation_flags': primary_result.validation_flags,
            'regex_extraction': regex_result.to_dict(),
            'llm_extraction': llm_result.to_dict(),
            'extractions_differ': differs,
            'regex_quality_score': regex_quality,
            'llm_quality_score': llm_quality,
            'auto_selected': primary_method
        }


class UserFeedbackLogger:
    """Log user feedback about incorrect N extractions"""
    
    def __init__(self, log_file: str = "n_extraction_feedback.csv"):
        self.log_file = log_file
        self._init_log()
    
    def _init_log(self):
        """Initialize log file if it doesn't exist"""
        if not os.path.exists(self.log_file):
            import csv
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pmid', 'title', 'reported_n', 'correct_n',
                    'extraction_method', 'confidence', 'user_notes'
                ])
    
    def log_feedback(self, pmid: str, title: str, reported_n: Optional[int],
                     correct_n: Optional[int], method: str, confidence: float,
                     notes: str = "") -> bool:
        """Log user feedback, avoiding duplicates"""
        import csv
        
        # Check if this PMID already has recent feedback
        existing = []
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                reader = csv.DictReader(f)
                existing = list(reader)
        
        # Check for duplicate (same PMID and correct_n in last 24 hours)
        from datetime import datetime, timedelta
        now = datetime.now()
        for row in existing:
            if row['pmid'] == pmid and row['correct_n'] == str(correct_n):
                try:
                    log_time = datetime.fromisoformat(row['timestamp'])
                    if now - log_time < timedelta(hours=24):
                        return False  # Skip duplicate
                except:
                    pass
        
        # Append new feedback
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.isoformat(),
                pmid,
                title[:100],
                reported_n,
                correct_n,
                method,
                confidence,
                notes
            ])
        
        return True


# Backward compatibility functions
def extract_sample_size_enhanced(text: str, title: str = "") -> Dict[str, Any]:
    """Enhanced extraction with all features - synchronous wrapper"""
    extractor = EnhancedNExtractor()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(extractor.extract_combined(text, title))
    finally:
        loop.close()


def log_extraction_feedback(pmid: str, title: str, reported_n: Optional[int],
                           correct_n: Optional[int], method: str = "unknown",
                           confidence: float = 0.0, notes: str = "") -> bool:
    """Log user feedback about incorrect extraction"""
    logger = UserFeedbackLogger()
    return logger.log_feedback(pmid, title, reported_n, correct_n, method, confidence, notes)