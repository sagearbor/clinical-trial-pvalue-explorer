# Universal Study P-Value Explorer - Implementation Guide

## Current Status: Phase 6 Complete (v6.0)

**Last Updated**: December 2024  
**Current Version**: 6.0  
**Phase**: Phase 6 Production Hardening Complete

## Implementation Overview

The Universal Study P-Value Explorer has evolved from a simple t-test calculator to a comprehensive AI-powered statistical analysis platform supporting multiple study types with intelligent test detection.

### Core Architecture

**Factory Pattern Implementation**
- `StatisticalTestFactory` in `statistical_tests.py` provides extensible test registration
- 23+ test aliases support various naming conventions
- Easy addition of new statistical tests following established patterns

**Enhanced API Integration**
- Universal `/process_idea` endpoint with AI-powered study type detection
- `/available_tests` endpoint provides UI-ready test metadata
- Backwards compatibility maintained with legacy endpoints

**Dynamic Frontend**
- AI study type suggestions with confidence levels
- Dynamic parameter forms that adapt to selected test types
- Enhanced results display with test-specific visualizations

## Implemented Statistical Tests

### 1. Two-Sample T-Test (Original)
- **Aliases**: `two_sample_t_test`, `t_test`, `ttest`
- **Use Case**: Compare means between two groups
- **Effect Size**: Cohen's d
- **Implementation**: Enhanced with new factory pattern

### 2. Chi-Square Test of Independence
- **Aliases**: `chi_square`, `chi2`, `categorical`, `independence`
- **Use Case**: Test association between categorical variables
- **Effect Size**: Cramér's V
- **Implementation**: Supports 2x2 and larger contingency tables

### 3. One-Way ANOVA
- **Aliases**: `one_way_anova`, `anova`, `f_test`, `multiple_groups`
- **Use Case**: Compare means across multiple groups
- **Effect Size**: Eta-squared (η²)
- **Implementation**: Handles 3+ groups with post-hoc recommendations

### 4. Correlation Analysis
- **Aliases**: `correlation`, `pearson`, `spearman`, `relationship`
- **Use Case**: Measure relationship strength between variables
- **Effect Size**: R-squared (r²)
- **Implementation**: Both Pearson and Spearman methods

### 5. Mann-Whitney U Test (NEW)
- **Aliases**: `mann_whitney`, `wilcoxon_ranksum`
- **Use Case**: Non-parametric alternative to two-sample t-test
- **Effect Size**: Rank-biserial correlation
- **Implementation**: Full test with power analysis

### 6. Kruskal-Wallis Test (NEW)
- **Aliases**: `kruskal_wallis`
- **Use Case**: Non-parametric alternative to one-way ANOVA
- **Effect Size**: Epsilon-squared
- **Implementation**: Supports multiple groups with post-hoc suggestions

### 7. Wilcoxon Signed-Rank Test (NEW)
- **Aliases**: `wilcoxon_signed`, `wilcoxon_paired`
- **Use Case**: Non-parametric alternative to paired t-test
- **Effect Size**: Rank-biserial correlation
- **Implementation**: Includes Hodges-Lehmann estimator

### 8. Spearman Correlation (NEW)
- **Aliases**: `spearman`, `spearman_correlation`
- **Use Case**: Non-parametric correlation analysis
- **Effect Size**: Spearman's rho
- **Implementation**: With confidence intervals

## Key Components

### Backend (`api.py`)
- **FastAPI server** with comprehensive endpoint coverage
- **LLM Integration** supporting OpenAI, Gemini, Azure OpenAI
- **Study type mapping** with intelligent fallback logic
- **Error handling** and validation for all endpoints

### Statistical Engine (`statistical_tests.py`)
- **Abstract base classes** for consistent test implementation
- **Factory registration** system for automatic test discovery
- **Comprehensive validation** and error handling
- **Effect size calculations** for all test types

### Frontend (`app.py` + modular components)
- **Dual-mode interface** (AI-assisted and Manual Expert modes)
- **Modular components** in `frontend/components/`
  - `input_forms.py`: Dynamic parameter forms and presets
  - `visualizations.py`: Interactive Plotly visualizations
- **AI-powered suggestions** with confidence scores
- **Interactive visualizations** with Plotly.js
- **Parameter presets** for common study designs
- **Export functionality** for analysis results

### Utilities (`statistical_utils.py`)
- **Power analysis functions** for sample size planning
- **Effect size interpretations** with contextual explanations
- **Common statistical calculations** used across tests

### Visualization Engine (`visualizations.py`) (NEW)
- **Interactive Plotly charts** for all analyses
- **Power curves** with multiple effect sizes
- **P-value distributions** under null/alternative
- **Effect size sensitivity** analysis
- **Sample size optimization** visualizations
- **Confidence interval** plots
- **Scenario comparison** dashboards

### Non-Parametric Tests (`nonparametric.py`) (NEW)
- **Complete suite** of non-parametric alternatives
- **Factory pattern** for test selection
- **Power analysis** for all tests
- **Effect size calculations** specific to each test

### Bayesian Engine (`bayesian.py`) (NEW)
- **Bayesian t-test** with conjugate priors
- **Bayesian proportion test** using Beta-Binomial
- **Bayesian ANOVA** with BIC approximation
- **Bayesian correlation** analysis
- **Prior library** (uninformative, skeptical, optimistic)
- **ROPE analysis** for practical significance
- **Bayes Factors** for hypothesis testing
- **Bayesian power analysis** with simulations

## Test Coverage

**Total Test Functions**: 250+ across all components
**Pass Rate**: 100% (all tests passing)

### Test Categories
- **Unit Tests**: Individual function validation
- **Integration Tests**: API and frontend connectivity
- **Visualization Tests**: Plotly chart generation
- **Non-Parametric Tests**: Complete test suite for new methods
- **Bayesian Tests**: Comprehensive Bayesian engine testing
- **Backwards Compatibility**: Legacy functionality preservation
- **Edge Case Tests**: Error handling and boundary conditions

## Development Workflow

### Adding New Statistical Tests

1. **Create Test Class** following `StatisticalTest` abstract base
2. **Implement Required Methods**:
   - `calculate()`: Core statistical computation
   - `get_required_parameters()`: Parameter specification
   - `validate_parameters()`: Input validation
3. **Register with Factory** using descriptive aliases
4. **Create Test Suite** with comprehensive edge cases
5. **Update API Routing** if new study type detection needed
6. **Update Frontend Forms** for test-specific parameters

### Quality Assurance Process

1. **Implementation**: Developer creates feature with tests
2. **Validation**: QA agent validates implementation
3. **Integration**: Verify compatibility with existing features
4. **Documentation**: Update implementation guide
5. **Deployment**: Ready for production use

## LLM Provider Configuration

### OpenAI
```python
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
```

### Google Gemini
```python
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-pro
```

### Azure OpenAI
```python
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
```

## Deployment Instructions

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn api:app --reload --port 8000

# Start Streamlit frontend
streamlit run app.py --server.port 8501
```

### Production Deployment
- API server ready for containerization
- Environment variables configured
- Health checks implemented at `/health`
- Logging and monitoring available

## Completed Features (Phases 1-6)

### Phase 1-2: Core Statistical Platform
1. ✅ **Factory Pattern Architecture** (23+ test aliases)
2. ✅ **Multi-LLM Integration** (OpenAI, Gemini, Anthropic)
3. ✅ **Universal API Endpoints** (AI-powered study detection)
4. ✅ **Dynamic Frontend** (adaptive parameter forms)

### Phase 3: Advanced Statistical Methods
1. ✅ **Dual-Mode Interface** (AI-assisted and Manual Expert modes)
2. ✅ **Non-Parametric Tests** (Mann-Whitney, Kruskal-Wallis, Wilcoxon, Spearman)
3. ✅ **Bayesian Statistical Engine** (t-test, proportions, ANOVA, correlation)
4. ✅ **Interactive Visualizations** (Plotly.js integration)
5. ✅ **Modular Frontend Architecture** (separated components)

### Phase 4: Advanced Trial Designs
1. ✅ **Monte Carlo Simulations** (parallel processing framework)
2. ✅ **Group Sequential Designs** (O'Brien-Fleming, Pocock, alpha spending)
3. ✅ **Master Protocols** (basket, umbrella, platform trials)
4. ✅ **Sample Size Re-estimation** (adaptive designs)

### Phase 5: Domain Specialization
1. ✅ **Clinical Trial Specialization** (FDA guidelines, endpoints)
2. ✅ **Psychology Research** (behavioral studies, effect sizes)
3. ✅ **Education Research** (learning outcomes, intervention studies)
4. ✅ **Marketing Experiments** (A/B testing, conversion optimization)
5. ✅ **File Upload/Integration** (CSV, Excel, SAS, SPSS)
6. ✅ **Export Functionality** (PDF, CSV, JSON, PNG, SVG)

### Phase 6: Production Hardening
1. ✅ **Statistical Validation** (R/SAS comparison suite)
2. ✅ **API Documentation** (OpenAPI, Postman, SDK examples)
3. ✅ **Monitoring & Alerting** (Prometheus metrics, multi-channel alerts)
4. ✅ **WebSocket Support** (real-time parameter updates)
5. ✅ **Comprehensive Test Suite** (280+ tests across all components)

## Architecture Decisions

### Why Factory Pattern?
- **Extensibility**: Easy addition of new tests
- **Consistency**: Standardized interface across all tests
- **Discoverability**: Automatic test registration and availability
- **Maintainability**: Clear separation of concerns

### Why FastAPI?
- **Performance**: Async support for concurrent requests
- **Documentation**: Automatic OpenAPI spec generation
- **Validation**: Pydantic models for request/response validation
- **Standards**: Modern Python web framework

### Why Streamlit?
- **Rapid Development**: Quick UI prototyping and iteration
- **Scientific Focus**: Built for data science applications
- **Python Integration**: Seamless integration with backend
- **Visualization**: Built-in plotting and chart capabilities

## Contributing Guidelines

### Code Standards
- Follow PEP 8 Python style guidelines
- Comprehensive test coverage for new features
- Documentation strings for all public methods
- Type hints for function parameters and returns

### Testing Requirements
- Unit tests for all statistical calculations
- Integration tests for API endpoints
- Frontend tests for user interactions
- Backwards compatibility verification

### Documentation Updates
- Update this implementation guide
- Add examples for new statistical tests
- Update API documentation
- Create user guides for new features

---

*This implementation guide is maintained as the single source of truth for the project. Historical implementation details are archived in `docs/archived/`.*