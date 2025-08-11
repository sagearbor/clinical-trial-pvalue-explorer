
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

ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "GEMINI").upper()

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

class MultiScenarioAnalysisOutput(BaseModel):
    scenarios: Optional[Dict[str, dict]] = None
    recommended_scenario: Optional[str] = None
    effect_size_uncertainty: Optional[str] = None
    evidence_quality: Optional[str] = None
    llm_provider_used: Optional[str] = None
    error: Optional[str] = None

async def get_llm_enhanced_analysis_gemini(text: str) -> dict:
    return {
        "suggested_study_type": "two_sample_t_test",
        "rationale": "Default heuristic rationale.",
        "parameters": {"total_n": 100, "effect_size_value": 0.5, "effect_size_type": "cohens_d", "alpha": 0.05, "power": 0.8},
        "alternative_tests": ["chi_square"],
        "data_type": "continuous",
        "study_design": "randomized_controlled_trial",
        "confidence_level": 0.95,
        "initial_N": 100,
        "initial_cohens_d": 0.5,
        "estimation_justification": "Heuristic baseline",
        "references": [],
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
        "references": d.get("references") or [],
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

@app.post("/process_idea", response_model=StatisticalAnalysisOutput)
async def process_idea(item: EnhancedIdeaInput):
    if not item.study_description or not item.study_description.strip():
        raise HTTPException(status_code=400, detail="study_description must be provided and non-empty.")
    provider = (item.llm_provider or ACTIVE_LLM_PROVIDER).upper()

    llm = await get_llm_enhanced_analysis_gemini(item.study_description)
    v = validate_and_extract_enhanced_response(llm)

    calc = perform_statistical_calculations(v["suggested_study_type"], v["parameters"])

    refs = []
    if item.include_research and RESEARCH_INTELLIGENCE_AVAILABLE:
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
                        papers_payload.append({
                            "title": p.title,
                            "authors": p.authors,
                            "year": p.year,
                            "journal": p.journal,
                            "sample_size": p.sample_size,
                            "study_signal": p.study_signal,
                            "url": p.url,
                            "extras": p.extras,
                            "_inferred_source": src,
                        })
                    # attach structured data to response so frontend can render table
                    refs_structured = papers_payload
                else:
                    refs_structured = []
                    research_debug = {'pubmed': {'count':0}, 'clinicaltrials': {'count':0}, 'arxiv': {'count':0}}
        except Exception:
            refs = []

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
        references=refs,
        research_papers_data=refs_structured if 'refs_structured' in locals() else [],
        research_debug=research_debug if 'research_debug' in locals() else {},
        processed_idea=item.study_description,
        llm_provider_used=provider,
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
