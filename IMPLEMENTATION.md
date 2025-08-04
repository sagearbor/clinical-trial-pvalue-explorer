# Universal Study P-Value Explorer - Implementation Guide

## Current Status: Phase 2 Complete (v2.0)

**Last Updated**: August 3, 2025  
**Current Version**: 2.0  
**Phase**: Ready for Phase 3 Development

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

### Frontend (`app.py`)
- **Streamlit interface** with dynamic form generation
- **AI-powered suggestions** with user override capability
- **Test-specific visualizations** and result formatting
- **Export functionality** for analysis results

### Utilities (`statistical_utils.py`)
- **Power analysis functions** for sample size planning
- **Effect size interpretations** with contextual explanations
- **Common statistical calculations** used across tests

## Test Coverage

**Total Test Functions**: 195+ across all components
**Pass Rate**: 99.5% (194/195 tests passing)

### Test Categories
- **Unit Tests**: Individual function validation
- **Integration Tests**: API and frontend connectivity
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

## Phase 3 Roadmap

**Next Implementation Priorities**:
1. **Dual-Mode Interface** (AI vs Manual Expert modes)
2. **Advanced Statistical Tests** (non-parametric alternatives)
3. **Data Integration** (file upload and direct analysis)
4. **Domain Specialization** (clinical, psychology, education)

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