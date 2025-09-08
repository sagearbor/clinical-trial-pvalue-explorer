
# src/api.py — decoupled research + optional scenarios
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

from src.statistical_utils import perform_statistical_calculations
try:
    from src.research_intelligence import get_research_engine, RESEARCH_INTELLIGENCE_AVAILABLE
except Exception:
    RESEARCH_INTELLIGENCE_AVAILABLE = False
    def get_research_engine():
        return None

load_dotenv(override=True)
app = FastAPI()

ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "AZURE_OPENAI").upper()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

class EnhancedIdeaInput(BaseModel):
    study_description: str
    llm_provider: Optional[str] = None
    include_research: bool = False
    max_papers: Optional[int] = 5
    pubmed_papers: Optional[int] = None
    arxiv_papers: Optional[int] = None
    clinicaltrials_papers: Optional[int] = None

class StatisticalAnalysisOutput(BaseModel):
    suggested_study_type: Optional[str] = None
    rationale: Optional[str] = None
    parameters: Optional[dict] = None
    alternative_tests: Optional[List[str]] = None
    data_type: Optional[str] = None
    study_design: Optional[str] = None
    confidence_level: Optional[float] = None
    calculated_p_value: Optional[float] = None
    calculated_power: Optional[float] = None
    statistical_test_used: Optional[str] = None
    calculation_error: Optional[str] = None
    references: Optional[List[str]] = None
    research_papers_data: Optional[List[Dict[str, Any]]] = None
    research_debug: Optional[Dict[str, Any]] = None
    processed_idea: Optional[str] = None
    llm_provider_used: Optional[str] = None
    error: Optional[str] = None
    llm_warning: Optional[str] = None  # Warning when LLM is unavailable
    fallback_mode: Optional[bool] = None  # True when using pattern matching
    default_used: Optional[bool] = None  # True when defaulting to t-test

class MultiScenarioAnalysisOutput(BaseModel):
    scenarios: Optional[Dict[str, dict]] = None
    recommended_scenario: Optional[str] = None
    effect_size_uncertainty: Optional[str] = None
    evidence_quality: Optional[str] = None
    llm_provider_used: Optional[str] = None
    error: Optional[str] = None

async def get_llm_enhanced_analysis_azure_openai(text: str) -> dict:
    """Use Azure OpenAI to analyze study description."""
    
    # Check if Azure OpenAI is configured
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        try:
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            
            prompt = f"""Analyze this clinical study description and determine the appropriate statistical test.
            Study: {text}
            
            Return a JSON object with these fields:
            - suggested_study_type: one of [two_sample_t_test, paired_t_test, mixed_effects, chi_square, logistic_regression, cox_regression, mann_whitney, kruskal_wallis, one_way_anova, pearson_correlation, spearman_correlation, linear_regression]
            - rationale: brief explanation why this test is appropriate
            - parameters: object with total_n (sample size) and effect_size_value
            - alternative_tests: array of 2-3 other viable test options from the list above
            - data_type: one of [continuous, categorical, binary, survival, ordinal]
            - study_design: one of [randomized_controlled_trial, observational, cross_sectional, longitudinal_repeated_measures, prospective_cohort, case_control]
            
            Example response:
            {{"suggested_study_type": "mixed_effects", "rationale": "Repeated measures over time with multiple covariates", "parameters": {{"total_n": 200, "effect_size_value": 0.5}}, "alternative_tests": ["repeated_measures_anova", "gee", "linear_regression"], "data_type": "continuous", "study_design": "longitudinal_repeated_measures"}}
            """
            
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a biostatistics expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse the response
            result_text = response.choices[0].message.content
            import json
            try:
                result = json.loads(result_text)
                result["llm_used"] = True
                result["provider"] = "Azure OpenAI"
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse Azure OpenAI response: {result_text}")
                # Fall through to pattern matching
                
        except Exception as e:
            print(f"Azure OpenAI error: {e}")
    
    # If Azure OpenAI fails or isn't configured, fall back to pattern matching
    return await get_llm_enhanced_analysis_fallback(text)

async def get_llm_enhanced_analysis_fallback(text: str) -> dict:
    """Fallback pattern matching when LLM is unavailable."""
    
    # This is the existing pattern matching logic
    return await get_llm_enhanced_analysis_gemini(text)

async def get_llm_enhanced_analysis_gemini(text: str) -> dict:
    """Analyze study description to detect appropriate statistical test."""
    
    # First try to use actual Gemini API if available
    try:
        import google.generativeai as genai
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_key_here":
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            
            prompt = f"""Analyze this clinical study description and suggest the appropriate statistical test.
            Study: {text}
            
            Return a JSON with:
            - suggested_study_type: the test name (e.g., 'mixed_effects', 'two_sample_t_test', 'cox_regression')
            - rationale: why this test is appropriate
            - parameters: estimated sample size and effect size
            - alternative_tests: other viable options
            """
            
            response = model.generate_content(prompt)
            # Parse response and return
            # This would need proper JSON parsing from the response
            
    except Exception as e:
        # Log the error but continue with fallback
        print(f"⚠️ Gemini API error: {e}")
    
    # FALLBACK: Pattern matching when LLM is unavailable
    text_lower = text.lower()
    
    # Add warning flag to indicate fallback mode
    warning_message = "⚠️ LLM unavailable - using pattern matching fallback. Results may be less accurate."
    
    # Quick pattern matching for common complex scenarios
    if any(term in text_lower for term in ['mixed effects', 'mmrm', 'repeated measures', 'mixed model', 'longitudinal']):
        return {
            "suggested_study_type": "mixed_effects",
            "rationale": "Study mentions mixed effects model or repeated measures analysis",
            "parameters": {"total_n": 200, "effect_size_value": 0.5, "alpha": 0.05},
            "alternative_tests": ["repeated_measures_anova", "gee"],
            "data_type": "continuous",
            "study_design": "longitudinal_repeated_measures",
            "llm_warning": warning_message,
            "fallback_mode": True
        }
    elif 'ancova' in text_lower:
        return {
            "suggested_study_type": "ancova",
            "rationale": "Study mentions ANCOVA for covariate adjustment",
            "parameters": {"total_n": 150, "effect_size_value": 0.5, "alpha": 0.05},
            "alternative_tests": ["linear_regression", "two_way_anova"],
            "data_type": "continuous",
            "study_design": "randomized_controlled_trial",
            "llm_warning": warning_message,
            "fallback_mode": True
        }
    elif any(term in text_lower for term in ['logistic regression', 'binary outcome', 'odds ratio']):
        return {
            "suggested_study_type": "logistic_regression", 
            "rationale": "Binary outcome suggests logistic regression",
            "parameters": {"total_n": 200, "effect_size_value": 1.5, "effect_size_type": "odds_ratio"},
            "alternative_tests": ["chi_square", "fisher_exact"],
            "data_type": "binary",
            "study_design": "observational",
        }
    elif any(term in text_lower for term in ['survival', 'cox', 'kaplan', 'time to event', 'hazard']):
        return {
            "suggested_study_type": "cox_regression",
            "rationale": "Survival or time-to-event analysis detected",
            "parameters": {"total_n": 250, "effect_size_value": 0.7, "effect_size_type": "hazard_ratio"},
            "alternative_tests": ["kaplan_meier", "log_rank"],
            "data_type": "survival",
            "study_design": "prospective_cohort",
        }
    elif 'anova' in text_lower or 'multiple groups' in text_lower:
        return {
            "suggested_study_type": "one_way_anova",
            "rationale": "Multiple group comparison suggests ANOVA",
            "parameters": {"total_n": 150, "effect_size_value": 0.4, "effect_size_type": "eta_squared"},
            "alternative_tests": ["kruskal_wallis", "two_way_anova"],
            "data_type": "continuous",
            "study_design": "randomized_controlled_trial",
        }
    elif any(term in text_lower for term in ['chi-square', 'chi square', 'categorical', 'contingency']):
        return {
            "suggested_study_type": "chi_square",
            "rationale": "Categorical data analysis detected",
            "parameters": {"total_n": 200},
            "alternative_tests": ["fisher_exact", "mcnemar"],
            "data_type": "categorical",
            "study_design": "cross_sectional",
        }
    elif any(term in text_lower for term in ['mann-whitney', 'wilcoxon', 'non-parametric', 'nonparametric']):
        return {
            "suggested_study_type": "mann_whitney",
            "rationale": "Non-parametric test requested",
            "parameters": {"total_n": 100, "effect_size_value": 0.5},
            "alternative_tests": ["wilcoxon_signed", "kruskal_wallis"],
            "data_type": "ordinal",
            "study_design": "randomized_controlled_trial",
        }
    elif any(term in text_lower for term in ['correlation', 'pearson', 'spearman', 'relationship']):
        return {
            "suggested_study_type": "pearson_correlation",
            "rationale": "Correlation analysis detected",
            "parameters": {"total_n": 100, "effect_size_value": 0.3, "effect_size_type": "correlation_r"},
            "alternative_tests": ["spearman_correlation", "linear_regression"],
            "data_type": "continuous",
            "study_design": "observational",
        }
    else:
        # Default to t-test for simple two-group comparisons
        return {
            "suggested_study_type": "two_sample_t_test",
            "rationale": "⚠️ DEFAULT FALLBACK: Could not detect specific test type. Defaulting to two-sample t-test.",
            "parameters": {"total_n": 100, "effect_size_value": 0.5, "effect_size_type": "cohens_d", "alpha": 0.05},
            "alternative_tests": ["welch_t_test", "mann_whitney"],
            "data_type": "continuous",
            "study_design": "randomized_controlled_trial",
            "llm_warning": "⚠️ WARNING: LLM analysis unavailable. Using basic pattern matching. Please verify the suggested test is appropriate for your study.",
            "fallback_mode": True,
            "default_used": True
        }

def validate_and_extract_enhanced_response(d: dict) -> dict:
    return {
        "suggested_study_type": d.get("suggested_study_type"),
        "rationale": d.get("rationale"),
        "parameters": d.get("parameters") or {},
        "alternative_tests": d.get("alternative_tests") or [],
        "data_type": d.get("data_type"),
        "study_design": d.get("study_design"),
        "confidence_level": d.get("confidence_level"),
        "initial_N": d.get("initial_N"),
        "initial_cohens_d": d.get("initial_cohens_d"),
        "estimation_justification": d.get("estimation_justification"),
        # NEVER extract references from LLM - they hallucinate!
        # References must only come from actual web searches via research_intelligence.py
    }


def map_study_type_to_test(suggested_study_type: Optional[str]) -> str:
    """Map LLM-provided study type strings to canonical test IDs used by the system."""
    if not suggested_study_type:
        return "two_sample_t_test"
    s = suggested_study_type.lower()
    aliases = {
        "t_test": "two_sample_t_test",
        "two_sample_ttest": "two_sample_t_test",
        "independent_samples_t_test": "two_sample_t_test",
        "chi2": "chi_square",
        "chi-square": "chi_square",
        "anova": "one_way_anova",
        "f_test": "one_way_anova",
        "pearson": "correlation",
        "spearman": "correlation",
        "correlation_test": "correlation",
    }
    # If not a known alias, fallback to the default two-sample t-test
    return aliases.get(s, "two_sample_t_test")

@app.get("/available_tests")
async def get_available_tests():
    """Get all available statistical tests."""
    return [
        {"value": "two_sample_t_test", "label": "Two-Sample T-Test", "category": "Parametric"},
        {"value": "paired_t_test", "label": "Paired T-Test", "category": "Parametric"},
        {"value": "welch_t_test", "label": "Welch's T-Test", "category": "Parametric"},
        {"value": "one_way_anova", "label": "One-Way ANOVA", "category": "Parametric"},
        {"value": "two_way_anova", "label": "Two-Way ANOVA", "category": "Parametric"},
        {"value": "repeated_measures_anova", "label": "Repeated Measures ANOVA", "category": "Parametric"},
        {"value": "chi_square", "label": "Chi-Square Test", "category": "Non-Parametric"},
        {"value": "fisher_exact", "label": "Fisher's Exact Test", "category": "Non-Parametric"},
        {"value": "mann_whitney", "label": "Mann-Whitney U Test", "category": "Non-Parametric"},
        {"value": "wilcoxon_signed", "label": "Wilcoxon Signed-Rank", "category": "Non-Parametric"},
        {"value": "kruskal_wallis", "label": "Kruskal-Wallis Test", "category": "Non-Parametric"},
        {"value": "friedman", "label": "Friedman Test", "category": "Non-Parametric"},
        {"value": "pearson_correlation", "label": "Pearson Correlation", "category": "Correlation"},
        {"value": "spearman_correlation", "label": "Spearman Correlation", "category": "Correlation"},
        {"value": "linear_regression", "label": "Linear Regression", "category": "Regression"},
        {"value": "logistic_regression", "label": "Logistic Regression", "category": "Regression"},
        {"value": "cox_regression", "label": "Cox Proportional Hazards", "category": "Survival"},
        {"value": "kaplan_meier", "label": "Kaplan-Meier Survival", "category": "Survival"},
        {"value": "mixed_effects", "label": "Mixed Effects Model (MMRM)", "category": "Advanced"},
        {"value": "gee", "label": "Generalized Estimating Equations", "category": "Advanced"},
        {"value": "bayesian_t_test", "label": "Bayesian T-Test", "category": "Bayesian"},
        {"value": "bayesian_anova", "label": "Bayesian ANOVA", "category": "Bayesian"},
        {"value": "sequential_design", "label": "Group Sequential Design", "category": "Adaptive"}
    ]

@app.post("/process_idea", response_model=StatisticalAnalysisOutput)
async def process_idea(item: EnhancedIdeaInput):
    if not item.study_description or not item.study_description.strip():
        raise HTTPException(status_code=400, detail="study_description must be provided and non-empty.")
    provider = (item.llm_provider or ACTIVE_LLM_PROVIDER).upper()
    
    # Use the appropriate LLM based on provider
    if provider == "AZURE_OPENAI":
        llm = await get_llm_enhanced_analysis_azure_openai(item.study_description)
    else:
        # For now, default to pattern matching for other providers
        llm = await get_llm_enhanced_analysis_fallback(item.study_description)
    v = validate_and_extract_enhanced_response(llm)

    calc = perform_statistical_calculations(v["suggested_study_type"], v["parameters"])

    refs = []
    refs_structured = []
    research_debug = {}
    
    # Only get references from actual web searches, never from LLM
    if item.include_research and RESEARCH_INTELLIGENCE_AVAILABLE:
        print(f"📚 Literature search enabled for: {item.study_description[:50]}...")
        try:
            engine = get_research_engine()
            if engine:
                summary = await engine.analyze_research_topic(
                    item.study_description,
                    max_papers=int(item.max_papers or 5),
                    pubmed_papers=item.pubmed_papers,
                    arxiv_papers=item.arxiv_papers,
                    clinicaltrials_papers=item.clinicaltrials_papers,
                )
                research_debug = {'pubmed': {}, 'clinicaltrials': {}, 'arxiv': {}}
                if summary and summary.papers_analyzed:
                    # Provide both simple markdown references and structured data for the frontend table
                    papers_payload = []
                    for p in summary.papers_analyzed:
                        # record source-level presence
                        src = (p.extras or {}).get('source') if p.extras and isinstance(p.extras, dict) else p.__dict__.get('source', None)
                        if not src:
                            # try journal-based heuristics
                            j = (p.journal or '').lower()
                            if 'clinicaltrials' in j or 'clinicaltrials.gov' in j:
                                src = 'clinicaltrials'
                            elif 'arxiv' in j:
                                src = 'arxiv'
                            else:
                                src = 'pubmed' if 'pubmed' in j else 'unknown'

                        research_debug.setdefault(src, {}).setdefault('count', 0)
                        research_debug[src]['count'] += 1

                        if "ClinicalTrials.gov" in (p.journal or ""):
                            s = f"**{p.title}**"
                            if p.sample_size: s += f" (N={p.sample_size})"
                            s += f" [ClinicalTrials.gov]({p.url})" if p.url else " [ClinicalTrials.gov]"
                        else:
                            auth = ", ".join(p.authors[:2]) + (" et al." if p.authors and len(p.authors) > 2 else "")
                            s = f"**{p.title}** {auth} ({p.year}). *{p.journal}*"
                            if p.url: s += f" [View]({p.url})"
                        refs.append(s)

                        # structured record for table view
                        paper_dict = {
                            "title": p.title,
                            "authors": p.authors,
                            "year": p.year,
                            "journal": p.journal,
                            "sample_size": p.sample_size,
                            "sample_size_method": p.sample_size_method,  # Pass through extraction method
                            "p_value": p.p_value,  # Pass through extracted p-value
                            "study_signal": p.study_signal,
                            "url": p.url,
                            "extras": p.extras,
                            "_inferred_source": src,
                        }
                        
                        # Debug logging
                        if p.sample_size_method or p.p_value:
                            print(f"   📊 Paper has fields: sample_size_method={p.sample_size_method}, p_value={p.p_value}")
                        
                        papers_payload.append(paper_dict)
                    # attach structured data to response so frontend can render table
                    refs_structured = papers_payload
                    print(f"✅ Found {len(refs)} references from web searches")
                else:
                    print("⚠️ No papers found in literature search")
                    research_debug = {'pubmed': {'count':0}, 'clinicaltrials': {'count':0}, 'arxiv': {'count':0}}
            else:
                print("⚠️ Research engine not available")
        except Exception as e:
            print(f"❌ Literature search error: {e}")
    else:
        print(f"📚 Literature search disabled (include_research={item.include_research})")

    return StatisticalAnalysisOutput(
        suggested_study_type=v["suggested_study_type"],
        rationale=v["rationale"],
        parameters=v["parameters"],
        alternative_tests=v["alternative_tests"],
        data_type=v["data_type"],
        study_design=v["study_design"],
        confidence_level=v["confidence_level"],
        calculated_p_value=calc.get("calculated_p_value"),
        calculated_power=calc.get("calculated_power"),
        statistical_test_used=calc.get("statistical_test_used"),
        calculation_error=calc.get("calculation_error"),
        references=refs,  # Only from actual web searches
        research_papers_data=refs_structured,  # Already initialized to []
        research_debug=research_debug,  # Already initialized to {}
        processed_idea=item.study_description,
        llm_provider_used=provider,
        llm_warning=llm.get("llm_warning"),  # Pass through warning
        fallback_mode=llm.get("fallback_mode"),  # Indicate fallback mode
        default_used=llm.get("default_used"),  # Indicate if defaulted to t-test
    )

@app.post("/analyze_scenarios", response_model=MultiScenarioAnalysisOutput)
async def analyze_scenarios(item: EnhancedIdeaInput):
    if not item.study_description or not item.study_description.strip():
        raise HTTPException(status_code=400, detail="study_description must be provided and non-empty.")
    base = {
        "exploratory": {"effect_size": 0.3, "power": 0.6, "desc": "Lower power, smaller effect"},
        "cautious": {"effect_size": 0.4, "power": 0.7, "desc": "Conservative approach"},
        "standard": {"effect_size": 0.5, "power": 0.8, "desc": "Typical research standard"},
        "optimistic": {"effect_size": 0.6, "power": 0.9, "desc": "Higher effect expectation"},
        "minimum viable": {"effect_size": 0.2, "power": 0.5, "desc": "Smallest detectable effect"},
    }
    scenarios: Dict[str, dict] = {}
    for i, (name, cfg) in enumerate(base.items(), start=1):
        n = int(100 / (cfg["effect_size"]**2))
        scenarios[f"scenario_{i}"] = {
            "name": name.title(),
            "description": cfg["desc"],
            "target_p_value": 0.05,
            "parameters": {
                "total_n": n,
                "effect_size_value": cfg["effect_size"],
                "effect_size_type": "cohens_d",
                "alpha": 0.05,
                "power": cfg["power"],
            },
        }
    return MultiScenarioAnalysisOutput(
        scenarios=scenarios,
        recommended_scenario="scenario_3",
        effect_size_uncertainty="medium",
        evidence_quality="unknown",
        llm_provider_used=(item.llm_provider or ACTIVE_LLM_PROVIDER).upper()
    )
