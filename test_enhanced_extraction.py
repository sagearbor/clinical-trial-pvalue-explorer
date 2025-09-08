#!/usr/bin/env python3
"""Test the enhanced extraction system"""

import asyncio
from src.enhanced_extraction import EnhancedNExtractor, UserFeedbackLogger

async def test_enhanced_extraction():
    """Test enhanced extraction with both regex and LLM"""
    
    # Test cases with known issues
    test_abstracts = [
        {
            'title': 'Test 1: Multiple N values',
            'text': """Background: Previous studies with 5000 patients showed mixed results.
                      Methods: We screened 2,677 individuals via email survey.
                      Study Sample: Our sample (n = 321) included 182 female patients.
                      Results: Of the 321 participants, 295 completed the study."""
        },
        {
            'title': 'Test 2: Simple N',
            'text': """A randomized controlled trial with n = 100 patients was conducted.
                      Fifty patients were assigned to treatment and 50 to control."""
        },
        {
            'title': 'Test 3: Ambiguous N',
            'text': """We analyzed data from 36 peer-reviewed journals containing 
                      reports of 1,234 clinical trials with a total of 500,000 participants."""
        }
    ]
    
    extractor = EnhancedNExtractor()
    
    print("=" * 80)
    print("ENHANCED N EXTRACTION TEST")
    print("=" * 80)
    
    for test in test_abstracts:
        print(f"\n📝 {test['title']}")
        print("-" * 40)
        
        # Test regex extraction
        regex_result = extractor.extract_with_regex(test['text'])
        print(f"REGEX:")
        print(f"  N = {regex_result.sample_size}")
        print(f"  Confidence = {regex_result.confidence:.2f}")
        print(f"  Context: {regex_result.context[:60] if regex_result.context else 'None'}...")
        if regex_result.validation_flags:
            print(f"  Warnings: {', '.join(regex_result.validation_flags)}")
        
        # Test LLM extraction (if available)
        try:
            llm_result = await extractor.extract_with_llm(test['text'], test['title'])
            print(f"\nLLM:")
            print(f"  N = {llm_result.sample_size}")
            print(f"  Confidence = {llm_result.confidence:.2f}")
            print(f"  Context: {llm_result.context[:60] if llm_result.context else 'None'}...")
        except Exception as e:
            print(f"\nLLM: Not available ({e})")
        
        # Test combined extraction
        combined = await extractor.extract_combined(test['text'], test['title'])
        print(f"\nCOMBINED:")
        print(f"  Selected N = {combined['sample_size']}")
        print(f"  Method = {combined['auto_selected']}")
        print(f"  Confidence = {combined['confidence']:.2f}")
        print(f"  Extractions differ = {combined['extractions_differ']}")
        print(f"  Quality scores: Regex={combined['regex_quality_score']:.2f}, LLM={combined['llm_quality_score']:.2f}")
    
    # Test feedback logger
    print("\n" + "=" * 80)
    print("TESTING FEEDBACK LOGGER")
    print("-" * 40)
    
    logger = UserFeedbackLogger()
    
    # Log some feedback
    success = logger.log_feedback(
        pmid="12345",
        title="Test paper",
        reported_n=100,
        correct_n=321,
        method="regex",
        confidence=0.75,
        notes="User reported actual N from methods section"
    )
    print(f"Logged feedback: {success}")
    
    # Try duplicate (should be skipped)
    success = logger.log_feedback(
        pmid="12345",
        title="Test paper",
        reported_n=100,
        correct_n=321,
        method="regex",
        confidence=0.75,
        notes="Duplicate"
    )
    print(f"Duplicate feedback (should be False): {success}")
    
    print("\n✅ Enhanced extraction test complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_extraction())