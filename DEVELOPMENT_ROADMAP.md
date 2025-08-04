# Universal Study P-Value Explorer - Development Roadmap

## Current Foundation
- **Working**: Two-sample t-test with Cohen's d, multi-LLM integration, Streamlit UI
- **Goal**: Expand to handle any study type with AI-driven study design detection

---

## Phase 1: Core Infrastructure (Weeks 1-2)

### 1.1 Enhanced LLM Integration (Foundation for everything else)
- [ ] **Update AI prompts** in `api.py` to return structured study analysis:
  ```json
  {
    "suggested_study_type": "two_sample_t_test",
    "rationale": "...",
    "parameters": { "total_n": 120, "cohens_d": 0.5 },
    "alternative_tests": ["welch_t_test", "mann_whitney"]
  }
  ```
- [ ] **Add study type detection logic** to identify:
  - Experimental vs observational
  - Number of groups/conditions
  - Data types (continuous, categorical)
  - Relationship being tested

### 1.2 Statistical Test Factory Pattern
- [ ] **Create `statistical_tests.py` module** with base classes:
  ```python
  class StatisticalTest:
      def calculate_p_value(self, **params) -> float
      def calculate_power(self, **params) -> float
      def get_required_params(self) -> list
  ```
- [ ] **Implement test factory** that routes study types to appropriate test classes
- [ ] **Refactor existing t-test** to use new pattern

---

## Phase 2: Essential Statistical Tests (Weeks 3-4)

### 2.1 Immediate Test Additions (builds on factory pattern)
- [ ] **Chi-square test** (categorical data - common AI suggestion)
- [ ] **One-way ANOVA** (multiple groups - extends current logic)
- [ ] **Correlation tests** (Pearson/Spearman - relationship studies)

### 2.2 API Endpoint Evolution
- [ ] **Extend `/process_idea`** to handle new study types
- [ ] **Add `/available_tests`** endpoint for UI dropdown
- [ ] **Update response models** to include test-specific parameters

### 2.3 Frontend Adaptation
- [ ] **Add study type display** showing AI's suggestion
- [ ] **Create dynamic parameter forms** that change based on detected test
- [ ] **Implement "Switch to Manual Mode"** toggle

---

## Phase 3: User Experience Enhancement (Weeks 5-6)

### 3.1 Dual-Mode Interface
- [ ] **AI Suggestion Mode** (default):
  - Show AI's study type recommendation
  - Pre-populate estimated parameters
  - Allow parameter tweaking
- [ ] **Manual Expert Mode**:
  - Study type dropdown
  - Manual parameter entry
  - Validation warnings

### 3.2 Results Enhancement
- [ ] **Effect size interpretation** with context
- [ ] **Statistical vs practical significance** indicators
- [ ] **Confidence intervals** display alongside p-values
- [ ] **Power visualization** with sample size recommendations

---

## Phase 4: Statistical Depth (Weeks 7-10)

### 4.1 Test Coverage Expansion
- [ ] **Non-parametric alternatives**:
  - Mann-Whitney U (for non-normal data)
  - Kruskal-Wallis (non-parametric ANOVA)
- [ ] **Advanced t-tests**:
  - Paired t-test (within-subjects)
  - Welch's t-test (unequal variances)
- [ ] **Regression analysis**:
  - Simple linear regression
  - Multiple regression

### 4.2 Power Analysis Suite
- [ ] **Pre-study power calculations**
- [ ] **Sample size optimization**
- [ ] **Power curves** showing N vs power
- [ ] **Minimum detectable effect sizes**

### 4.3 Multiple Comparison Handling
- [ ] **Bonferroni correction**
- [ ] **FDR (False Discovery Rate)**
- [ ] **Automatic correction suggestions**

---

## Phase 5: Advanced Features (Weeks 11-16)

### 5.1 Study Design Framework
- [ ] **Study design templates**:
  - RCT, crossover, factorial
  - Cohort, case-control, cross-sectional
- [ ] **Parameter validation** by study type
- [ ] **Design-specific effect sizes**

### 5.2 Domain Specialization
- [ ] **Domain-specific prompts**:
  - Clinical trials (current)
  - Psychology experiments
  - Educational research
  - Marketing studies
- [ ] **Field-specific parameter ranges**
- [ ] **Domain expertise in interpretations**

### 5.3 Data Integration
- [ ] **File upload support** (CSV, Excel)
- [ ] **Direct data analysis** option
- [ ] **Summary statistics calculation**

---

## Implementation Notes

### Critical Success Dependencies
1. **Phase 1 must complete before Phase 2** - Factory pattern enables all new tests
2. **Enhanced LLM prompts are foundation** - Everything builds on better AI detection
3. **Dual-mode UI design** - Maintains simplicity while adding power

### Key Technical Decisions
- **Keep existing t-test working** while adding new capabilities
- **Maintain single API endpoint** but enhance response structure
- **Preserve current UI flow** while adding intelligence
- **Use existing LLM providers** with enhanced prompts

### Validation Strategy
- **Test each new statistical function** against known results
- **Validate AI suggestions** with statistical experts
- **Maintain backwards compatibility** with current functionality

### Success Metrics
1. AI correctly identifies study type >80% of time
2. All statistical calculations match R/SPSS results
3. User can complete analysis in <5 clicks
4. Tool handles 10+ common study types

This roadmap transforms your tool incrementally, with each phase building on the previous one, ensuring no functionality gaps while expanding capabilities systematically.