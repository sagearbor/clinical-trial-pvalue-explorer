# Claude Code Assistant - Project Structure Guide

This file helps Claude Code understand the repository organization and development context.

## Repository Structure

```
clinical-trial-pvalue-explorer/
├── app.py                     # Main Streamlit application (entry point)
├── requirements.txt           # Python dependencies  
├── README.md                  # Main project documentation
├── DEVELOPMENT_ROADMAP.md     # High-level development plan
├── IMPLEMENTATION.md          # Current implementation status
│
├── src/                       # Core Python modules
│   ├── __init__.py           # Package initialization
│   ├── api.py                # FastAPI backend with LLM integration
│   ├── statistical_tests.py  # Statistical test factory pattern
│   ├── statistical_utils.py  # Utility functions for calculations
│   └── research_intelligence.py # Web search and research pipeline
│
├── tests/                     # Test suites
│   ├── test_*.py             # Current test files
│   └── legacy/               # Legacy test files
│
├── docs/                      # Documentation
│   ├── development/          # Development-specific docs
│   │   ├── PHASE_*_*.md     # Phase completion reports
│   │   ├── ADDING_NEW_STATISTICAL_TESTS.md
│   │   └── SYSTEM_CAPABILITY_MATRIX.md
│   └── archived/             # Historical documentation
│
├── examples/                  # Usage examples and demos
│   └── demos/                # Demo scripts for each statistical test
│
├── scripts/                   # Utility scripts
│   ├── generate-flowchart.py
│   ├── log-agent-activity.py
│   └── scan-codebase.py
│
└── logs/                      # Development logs
    └── agent-activity.yaml
```

## Development Context

### Current Status: Phase 2 Complete → Phase 3+ Enhancements

**Working Features:**
- Multi-LLM integration (Gemini, OpenAI, Anthropic)
- Statistical test factory pattern supporting:
  - Two-sample t-tests
  - Chi-square tests
  - One-way ANOVA
  - Correlation analysis (Pearson/Spearman)
- Streamlit frontend with dynamic parameter forms
- AI-powered study type detection and parameter suggestions

**Current Enhancement Goals:**
1. **Stage 1 LLM Web Search Pipeline** (research_intelligence.py) - IMPLEMENTED
2. **5-Scenario Generation** with uncertainty-based recommendations  
3. **Interactive Plotly.js Visualizations** for power curves
4. **Real Citation Integration** from PubMed/arXiv searches
5. **Repository Cleanup** - COMPLETED

### Key Files for Development:

**Main Application:**
- `app.py` - Streamlit frontend (update UI components here)
- `src/api.py` - FastAPI backend (add new endpoints here)

**Core Logic:**
- `src/statistical_tests.py` - Add new statistical tests
- `src/statistical_utils.py` - Mathematical utility functions
- `src/research_intelligence.py` - Web search and research synthesis

**Testing:**
- `tests/` - Add new test files here
- Run tests: `python -m pytest tests/`

**Documentation:**
- Update `README.md` for user-facing changes
- Update `IMPLEMENTATION.md` for development progress
- Add development notes to `docs/development/`

## Development Commands

**Start Application:**
```bash
# Terminal 1: Start FastAPI backend
uvicorn src.api:app --reload --port 8000

# Terminal 2: Start Streamlit frontend  
streamlit run app.py
```

**Run Tests:**
```bash
python -m pytest tests/ -v
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

## API Configuration

Set environment variables for LLM providers:
```bash
# Required for research intelligence
PUBMED_EMAIL=researcher@example.com
PUBMED_API_KEY=your_key_here

# LLM providers (at least one required)
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  
ANTHROPIC_API_KEY=your_key_here
```

## Current Issues to Address

1. **LLM Parameter Generation**: Chi-square, ANOVA, correlation tests need proper parameter generation
2. **Web Search Integration**: Connect research_intelligence.py to main pipeline
3. **Visualization Enhancement**: Replace matplotlib with Plotly.js for interactive charts
4. **Frontend Performance**: Address polyfill.js loading issues
5. **Effect Size Explanation**: Better Cohen's d interpretation and user guidance

## Next Development Priorities

1. Fix current statistical test parameter generation (chi-square missing contingency_table)
2. Integrate Stage 1 research intelligence pipeline  
3. Implement 5-scenario generation with uncertainty assessment
4. Add interactive visualizations with Plotly.js
5. Create real citation validation from web searches

---

*This file helps maintain development context across sessions. Update as the project evolves.*