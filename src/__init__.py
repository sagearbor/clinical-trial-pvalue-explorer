"""
Clinical Trial P-Value Explorer - Core Module

This package contains the core functionality for the clinical trial p-value explorer:
- API endpoints and LLM integration (api.py)
- Statistical test implementations and factory pattern (statistical_tests.py) 
- Statistical utility functions (statistical_utils.py)
- Research intelligence and web search pipeline (research_intelligence.py)
"""

__version__ = "2.1.0"
__author__ = "Clinical Trial P-Value Explorer Team"

# Make key classes and functions easily importable
from .statistical_tests import get_factory, StatisticalTestFactory
from .statistical_utils import calculate_p_value_from_N_d, calculate_power_from_N_d

# Research intelligence imports are optional (require additional dependencies)
try:
    from .research_intelligence import ResearchIntelligenceEngine, ResearchSummary
    _research_available = True
except ImportError:
    _research_available = False
    ResearchIntelligenceEngine = None
    ResearchSummary = None

__all__ = [
    'get_factory',
    'StatisticalTestFactory', 
    'calculate_p_value_from_N_d',
    'calculate_power_from_N_d'
]

if _research_available:
    __all__.extend(['ResearchIntelligenceEngine', 'ResearchSummary'])