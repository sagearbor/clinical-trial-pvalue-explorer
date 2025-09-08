# References Hallucination Fix - COMPLETE ✅

## Problem Identified
User reported that completely different search queries (antihypertensive vs hearing aids) were returning the same "Visceral Fat and Cardiometabolic Risk" reference, indicating LLM hallucination.

## Root Causes Found

### 1. **LLM Was Extracting References** ❌
- `validate_and_extract_enhanced_response()` was extracting `references` from LLM responses
- LLMs were never asked to provide references in prompts
- But the validation function was still trying to extract them
- This caused LLMs to hallucinate references

### 2. **PubMed Fallback Using Wrong Terms** ❌  
- When primary search failed, fallback was using hardcoded terms: "app", "mobile", "mhealth", "sms"
- These terms were irrelevant to most clinical studies
- Result: Same irrelevant papers appearing for all queries

## Fixes Applied

### Fix 1: Remove LLM Reference Extraction
```python
# BEFORE (api.py line 267):
"references": d.get("references") or [],

# AFTER:
# NEVER extract references from LLM - they hallucinate!
# References must only come from actual web searches via research_intelligence.py
```

### Fix 2: Improve PubMed Search Relevance
```python
# BEFORE: Hardcoded mobile app terms
synonyms = ["app", "mobile app", "smartphone app", "text messaging", "sms", "mhealth"]

# AFTER: Extract relevant medical terms from actual query
key_terms = self._extract_key_terms(idea)  # Extracts: diabetes, metformin, ACE inhibitor, etc.
```

### Fix 3: Add Relevance Filtering
- Papers must mention at least one key term from the query
- If no relevant papers found, return empty list (not random papers)
- Use AND instead of OR for fallback searches to ensure relevance

## Verification

### Test 1: No References Without Research
✅ When `include_research=False`, NO references are shown
✅ LLM responses no longer contain hallucinated references

### Test 2: Real References With Research  
✅ When `include_research=True`, references come from actual PubMed/ClinicalTrials/arXiv
✅ ClinicalTrials.gov returns relevant studies (e.g., "ACE Inhibitor" study for hypertension query)
✅ References have real URLs, PMIDs, and journal names

### Test 3: No More "Visceral Fat" Hallucination
✅ Different queries return different, relevant papers
✅ No more hardcoded or repeated irrelevant papers

## How It Works Now

1. **User enters study description** → "ACE inhibitors vs ARBs for hypertension"

2. **LLM analyzes for statistical test** (NO reference generation)
   - Suggests: Two-sample t-test
   - Provides: Sample size, effect size estimates
   - NO hallucinated references

3. **IF literature search enabled:**
   - PubMed searches for actual relevant papers
   - ClinicalTrials.gov finds relevant trials  
   - arXiv searches for relevant preprints
   - Papers are filtered for relevance
   - Only real, verifiable references shown

4. **IF literature search disabled:**
   - NO references shown at all
   - System works purely on statistical analysis

## Files Modified
1. `/src/api.py` - Removed reference extraction from LLM responses
2. `/src/research_intelligence.py` - Fixed PubMed search to use relevant terms
3. Added relevance filtering and better fallback logic

## User Impact
- ✅ No more hallucinated references
- ✅ References are real and verifiable (with URLs)
- ✅ Different queries return different, relevant papers
- ✅ Clean separation: LLM for statistics, Web search for references

---
*Fix completed successfully. References now come ONLY from actual web searches, never from LLM hallucination.*