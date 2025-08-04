# api.py (New Refactored Structure with Debug Prints)
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests # Ensure this is imported if used by fetch_url_content
from bs4 import BeautifulSoup # Ensure this is imported if used by fetch_url_content
from dotenv import load_dotenv
import re # Added for the OpenAI fallback logic

# Import LLM SDKs
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

# Import statistical test factory
from statistical_tests import get_factory, StatisticalTestFactory

# --- CRITICAL: load_dotenv() must be called BEFORE accessing os.getenv for .env variables ---
load_dotenv(override=True)

app = FastAPI()

# --- LLM Interaction Logic ---
ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "GEMINI").upper()


# --- Pydantic Models (IdeaInput, EstimationOutput) remain the same ---
class IdeaInput(BaseModel):
    text_idea: str | None = None
    url_idea: str | None = None

# Enhanced input model for Task 2.4
class EnhancedIdeaInput(BaseModel):
    study_description: str
    llm_provider: str | None = None  # Optional LLM provider override

class EstimationOutput(BaseModel):
    initial_N: int | None = None
    initial_cohens_d: float | None = None
    estimation_justification: str | None = None
    references: list[str] | None = None
    processed_idea: str | None = None
    llm_provider_used: str | None = None  # Add this to know which LLM responded
    error: str | None = None

# Enhanced response model for Phase 1.1
class StudyAnalysisOutput(BaseModel):
    suggested_study_type: str | None = None
    rationale: str | None = None
    parameters: dict | None = None
    alternative_tests: list[str] | None = None
    data_type: str | None = None
    study_design: str | None = None
    confidence_level: float | None = None
    processed_idea: str | None = None
    llm_provider_used: str | None = None
    error: str | None = None
    # Backwards compatibility fields
    initial_N: int | None = None
    initial_cohens_d: float | None = None
    estimation_justification: str | None = None
    references: list[str] | None = None

# Phase 1.2 enhanced response model with statistical calculations
class StatisticalAnalysisOutput(BaseModel):
    suggested_study_type: str | None = None
    rationale: str | None = None
    parameters: dict | None = None
    alternative_tests: list[str] | None = None
    data_type: str | None = None
    study_design: str | None = None
    confidence_level: float | None = None
    # Statistical calculations
    calculated_p_value: float | None = None
    calculated_power: float | None = None
    statistical_test_used: str | None = None
    calculation_error: str | None = None
    # Backwards compatibility fields  
    initial_N: int | None = None
    initial_cohens_d: float | None = None
    estimation_justification: str | None = None
    references: list[str] | None = None
    processed_idea: str | None = None
    llm_provider_used: str | None = None
    error: str | None = None

# --- URL Fetching (fetch_url_content) remains the same ---
def fetch_url_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'body', 'li', 'span'])
        text_content = "\n".join([para.get_text(separator=" ", strip=True) for para in paragraphs])
        return text_content[:5000]
    except Exception as e:
        return f"Error fetching/parsing URL: {str(e)}"


# --- System Prompt for Basic JSON Output (backwards compatibility) ---
SYSTEM_PROMPT_FOR_JSON = """You are a biostatistical assistant. A researcher is planning a clinical trial.
Assume this will be a two-arm study (e.g., treatment vs. control) with equal participants per arm (N_total / 2 per arm), and the primary outcome is continuous. The goal is to detect a difference between the groups.
Based on the research idea provided by the user, provide plausible estimates for:
1.  Total number of participants (N_total).
2.  A typical Cohen's d effect size.
3.  A brief justification for your N_total and Cohen's d estimates.
4.  A short list of references (URLs or article titles) that informed your estimates.

Consider that researchers often aim for 80% power at a significance level (alpha) of 0.05 (two-sided).
Your response MUST be a valid JSON object with four keys: "N_total" (an integer), "cohens_d" (a float), "justification" (a string), and "references" (an array of strings).
Example:
{
  "N_total": 128,
  "cohens_d": 0.5,
  "justification": "Medium effect size is common for this type of behavioral intervention, and N is estimated for 80% power.",
  "references": ["https://example.com/trial-guidelines"]
}
Provide only the JSON object. Do not include any other text, greetings, or explanations outside of the JSON structure.
"""

# --- Enhanced System Prompt for Study Analysis (Phase 1.1) ---
ENHANCED_SYSTEM_PROMPT = """You are an expert biostatistician and research methodologist. Analyze the provided research idea and recommend the most appropriate statistical approach.

Based on the research idea, you must provide a detailed study analysis with the following information:

1. **Study Type Detection**: Identify the most appropriate statistical test/analysis
2. **Study Design**: Determine if this is experimental vs observational
3. **Data Characteristics**: Identify the type of data (continuous, categorical, count, etc.)
4. **Sample Size & Effect Size**: Provide realistic estimates
5. **Alternative Approaches**: Suggest alternative statistical tests that might be appropriate

Your response MUST be a valid JSON object with the following schema:
{
  "suggested_study_type": "string (e.g., 'two_sample_t_test', 'one_way_anova', 'chi_square_test', 'survival_analysis', 'regression_analysis')",
  "rationale": "string (detailed explanation of why this test is recommended)",
  "parameters": {
    "total_n": "integer (recommended total sample size)",
    "cohens_d": "float (for continuous outcomes) or null",
    "effect_size_type": "string (e.g., 'cohens_d', 'odds_ratio', 'hazard_ratio')",
    "effect_size_value": "float (the actual effect size estimate)",
    "alpha": "float (significance level, typically 0.05)",
    "power": "float (desired statistical power, typically 0.8)"
  },
  "alternative_tests": ["array of strings listing alternative statistical approaches"],
  "data_type": "string (e.g., 'continuous', 'categorical', 'count', 'time_to_event', 'binary')",
  "study_design": "string (e.g., 'randomized_controlled_trial', 'cohort_study', 'case_control_study', 'cross_sectional')",
  "confidence_level": "float (confidence in the recommendation, 0.0-1.0)",
  "justification": "string (brief justification for sample size and effect size estimates)",
  "references": ["array of strings with relevant references"]
}

Example for a two-group comparison:
{
  "suggested_study_type": "two_sample_t_test",
  "rationale": "The research involves comparing a continuous outcome between two distinct groups (treatment vs control). A two-sample t-test is appropriate assuming normal distribution and equal variances.",
  "parameters": {
    "total_n": 128,
    "cohens_d": 0.5,
    "effect_size_type": "cohens_d",
    "effect_size_value": 0.5,
    "alpha": 0.05,
    "power": 0.8
  },
  "alternative_tests": ["welch_t_test", "mann_whitney_u", "permutation_test"],
  "data_type": "continuous",
  "study_design": "randomized_controlled_trial",
  "confidence_level": 0.85,
  "justification": "Medium effect size (d=0.5) is realistic for behavioral interventions. Sample size calculated for 80% power at alpha=0.05.",
  "references": ["Cohen J. Statistical Power Analysis for the Behavioral Sciences", "https://clinicaltrials.gov/guidance"]
}

Provide only the JSON object. No additional text, explanations, or formatting outside the JSON structure.
"""

# --- Provider-Specific Functions ---

async def get_llm_estimate_gemini(text_input: str) -> dict:
    print("--- Attempting to use GEMINI provider ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here": # Added check for placeholder
        raise ValueError("GEMINI_API_KEY not found or is a placeholder in environment variables.")
    genai.configure(api_key=api_key)
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") 
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "N_total": {"type": "INTEGER"},
                    "cohens_d": {"type": "NUMBER"},
                    "justification": {"type": "STRING"}
                },
                "required": ["N_total", "cohens_d", "justification"]
            }
        ),
        system_instruction=SYSTEM_PROMPT_FOR_JSON 
    )
    prompt_parts = [f"Research Idea: '{text_input[:3000]}'"]
    response = await model.generate_content_async(prompt_parts) 
    return json.loads(response.text) 

async def get_llm_estimate_openai(text_input: str, is_azure: bool = False) -> dict:
    if is_azure:
        print("--- Attempting to use AZURE_OPENAI provider ---")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Defaulting to a known stable version
        if not all([api_key, endpoint, deployment_name]) or api_key == "REDACTED": # Added check for placeholder
            raise ValueError("Azure OpenAI environment variables (API_KEY, ENDPOINT, DEPLOYMENT_NAME) not fully set or API_KEY is placeholder.")
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        model_to_call = deployment_name
    else:
        print("--- Attempting to use OPENAI (direct) provider ---")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here": # Added check for placeholder
            raise ValueError("OPENAI_API_KEY not found or is a placeholder in environment variables.")
        client = OpenAI(api_key=api_key)
        model_to_call = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_JSON},
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    
    try:
        print(f"--- Calling OpenAI/Azure model: {model_to_call} ---")
        completion = client.chat.completions.create(
            model=model_to_call,
            messages=messages,
            response_format={"type": "json_object"}, 
            temperature=0.2,
            max_tokens=400 
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e: 
        print(f"OpenAI API error with JSON mode (model: {model_to_call}): {e}")
        print("--- Attempting OpenAI/Azure call without explicit JSON mode (fallback) ---")
        try:
            completion = client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.2,
                max_tokens=400
            )
            response_content = completion.choices[0].message.content
            match = re.search(r"\{.*\}", response_content, re.DOTALL) # Using re import
            if match:
                return json.loads(match.group(0))
            else:
                print(f"--- Fallback failed: No JSON object found in OpenAI response: {response_content[:500]}... ---")
                raise ValueError(f"No JSON object found in OpenAI response: {response_content}")
        except Exception as fallback_e:
            print(f"OpenAI API fallback error (model: {model_to_call}): {fallback_e}")
            raise fallback_e


async def get_llm_estimate_anthropic(text_input: str) -> dict:
    print("--- Attempting to use ANTHROPIC provider ---")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here": # Added check for placeholder
        raise ValueError("ANTHROPIC_API_KEY not found or is a placeholder in environment variables.")
    
    client = Anthropic(api_key=api_key)
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    messages = [
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    print(f"--- Calling Anthropic model: {model_name} ---")
    response_obj = client.messages.create( # Corrected variable name
        model=model_name,
        system=SYSTEM_PROMPT_FOR_JSON, 
        max_tokens=1024, 
        messages=messages,
        temperature=0.2
    )
    # Ensure content is a list and not empty, and first item is a TextBlock
    if response_obj.content and isinstance(response_obj.content, list) and hasattr(response_obj.content[0], 'text'):
        response_text = response_obj.content[0].text
        return json.loads(response_text)
    else:
        print(f"--- Anthropic unexpected response structure: {response_obj.content} ---")
        raise ValueError("Anthropic response content not in expected format.")


# --- Enhanced LLM Functions for Phase 1.1 ---

async def get_llm_enhanced_analysis_gemini(text_input: str) -> dict:
    print("--- Attempting enhanced analysis with GEMINI provider ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY not found or is a placeholder in environment variables.")
    genai.configure(api_key=api_key)
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest")
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "suggested_study_type": {"type": "STRING"},
                    "rationale": {"type": "STRING"},
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "total_n": {"type": "INTEGER"},
                            "cohens_d": {"type": "NUMBER"},
                            "effect_size_type": {"type": "STRING"},
                            "effect_size_value": {"type": "NUMBER"},
                            "alpha": {"type": "NUMBER"},
                            "power": {"type": "NUMBER"}
                        }
                    },
                    "alternative_tests": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "data_type": {"type": "STRING"},
                    "study_design": {"type": "STRING"},
                    "confidence_level": {"type": "NUMBER"},
                    "justification": {"type": "STRING"},
                    "references": {"type": "ARRAY", "items": {"type": "STRING"}}
                },
                "required": ["suggested_study_type", "rationale", "parameters", "data_type", "study_design"]
            }
        ),
        system_instruction=ENHANCED_SYSTEM_PROMPT
    )
    prompt_parts = [f"Research Idea: '{text_input[:3000]}'"]
    response = await model.generate_content_async(prompt_parts)
    return json.loads(response.text)

async def get_llm_enhanced_analysis_openai(text_input: str, is_azure: bool = False) -> dict:
    if is_azure:
        print("--- Attempting enhanced analysis with AZURE_OPENAI provider ---")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not all([api_key, endpoint, deployment_name]) or api_key == "REDACTED":
            raise ValueError("Azure OpenAI environment variables not fully set or API_KEY is placeholder.")
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        model_to_call = deployment_name
    else:
        print("--- Attempting enhanced analysis with OPENAI (direct) provider ---")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not found or is a placeholder in environment variables.")
        client = OpenAI(api_key=api_key)
        model_to_call = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    messages = [
        {"role": "system", "content": ENHANCED_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    
    try:
        print(f"--- Calling enhanced OpenAI/Azure model: {model_to_call} ---")
        completion = client.chat.completions.create(
            model=model_to_call,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800  # Increased for more detailed response
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e:
        print(f"OpenAI API error with enhanced analysis (model: {model_to_call}): {e}")
        print("--- Attempting OpenAI/Azure enhanced call without explicit JSON mode (fallback) ---")
        try:
            completion = client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.2,
                max_tokens=800
            )
            response_content = completion.choices[0].message.content
            match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                print(f"--- Enhanced fallback failed: No JSON object found in OpenAI response: {response_content[:500]}... ---")
                raise ValueError(f"No JSON object found in OpenAI response: {response_content}")
        except Exception as fallback_e:
            print(f"OpenAI API enhanced fallback error (model: {model_to_call}): {fallback_e}")
            raise fallback_e

async def get_llm_enhanced_analysis_anthropic(text_input: str) -> dict:
    print("--- Attempting enhanced analysis with ANTHROPIC provider ---")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        raise ValueError("ANTHROPIC_API_KEY not found or is a placeholder in environment variables.")
    
    client = Anthropic(api_key=api_key)
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    messages = [
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    print(f"--- Calling enhanced Anthropic model: {model_name} ---")
    response_obj = client.messages.create(
        model=model_name,
        system=ENHANCED_SYSTEM_PROMPT,
        max_tokens=1500,  # Increased for more detailed response
        messages=messages,
        temperature=0.2
    )
    # Ensure content is a list and not empty, and first item is a TextBlock
    if response_obj.content and isinstance(response_obj.content, list) and hasattr(response_obj.content[0], 'text'):
        response_text = response_obj.content[0].text
        return json.loads(response_text)
    else:
        print(f"--- Enhanced Anthropic unexpected response structure: {response_obj.content} ---")
        raise ValueError("Anthropic response content not in expected format.")

def validate_and_extract_enhanced_response(llm_response_data: dict) -> dict:
    """Validate and extract data from enhanced LLM response"""
    try:
        # Validate required fields
        required_fields = ['suggested_study_type', 'rationale', 'data_type', 'study_design']
        for field in required_fields:
            if field not in llm_response_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Extract parameters safely
        parameters = llm_response_data.get('parameters', {})
        
        # For backwards compatibility, extract N and Cohen's d if available
        initial_N = parameters.get('total_n')
        initial_cohens_d = parameters.get('cohens_d') or parameters.get('effect_size_value')
        
        return {
            'suggested_study_type': llm_response_data.get('suggested_study_type'),
            'rationale': llm_response_data.get('rationale'),
            'parameters': parameters,
            'alternative_tests': llm_response_data.get('alternative_tests', []),
            'data_type': llm_response_data.get('data_type'),
            'study_design': llm_response_data.get('study_design'),
            'confidence_level': llm_response_data.get('confidence_level', 0.8),
            'initial_N': initial_N,
            'initial_cohens_d': initial_cohens_d,
            'estimation_justification': llm_response_data.get('justification'),
            'references': llm_response_data.get('references', [])
        }
    except Exception as e:
        print(f"--- Error validating enhanced response: {e} ---")
        raise ValueError(f"Invalid enhanced response format: {e}")


def map_study_type_to_test(suggested_study_type: str) -> str:
    """
    Map LLM-suggested study types to factory test identifiers.
    
    Args:
        suggested_study_type: Study type from LLM response
        
    Returns:
        Factory test identifier
    """
    # Mapping from LLM suggestions to factory test types
    type_mapping = {
        # Two-sample t-test mappings
        "two_sample_t_test": "two_sample_t_test",
        "t_test": "two_sample_t_test", 
        "two_sample_ttest": "two_sample_t_test",
        "independent_samples_t_test": "two_sample_t_test",
        "unpaired_t_test": "two_sample_t_test",
        "student_t_test": "two_sample_t_test",
        "welch_t_test": "two_sample_t_test",  # For now, map to regular t-test
        
        # Chi-square test mappings (Phase 2.1)
        "chi_square_test": "chi_square",
        "chi_square": "chi_square",
        "chi-square": "chi_square",  # Added hyphenated version
        "chi2": "chi_square",
        "categorical": "chi_square",  # Added single word version
        "categorical_test": "chi_square",
        "independence_test": "chi_square",
        "goodness_of_fit": "chi_square",
        "contingency_table_test": "chi_square",
        "association_test": "chi_square",
        "categorical_analysis": "chi_square",
        "chi_squared": "chi_square",
        "chi2_test": "chi_square",
        
        # One-way ANOVA test mappings (Phase 2.2)
        "one_way_anova": "one_way_anova",
        "anova": "one_way_anova",
        "f_test": "one_way_anova",
        "multiple_groups": "one_way_anova",
        "multiple groups": "one_way_anova",  # Added space version
        "analysis_of_variance": "one_way_anova",
        "oneway_anova": "one_way_anova",
        "compare_multiple_groups": "one_way_anova",
        "three_group_comparison": "one_way_anova",
        "multi_group_test": "one_way_anova",
        
        # Correlation test mappings (Phase 2.3)
        "correlation": "correlation",
        "pearson": "correlation",
        "spearman": "correlation",
        "relationship": "correlation",
        "correlation_test": "correlation",
        "pearson_correlation": "correlation",
        "spearman_correlation": "correlation",
        "correlation_analysis": "correlation",
        "linear_correlation": "correlation",
        "rank_correlation": "correlation",
        "bivariate_correlation": "correlation",
        "continuous_correlation": "correlation",
        
        # Future tests (fallback to appropriate tests when implemented)
        "regression_analysis": "two_sample_t_test",  # Fallback for now
    }
    
    study_type_lower = suggested_study_type.lower() if suggested_study_type else ""
    return type_mapping.get(study_type_lower, "two_sample_t_test")  # Default fallback


def perform_statistical_calculations(suggested_study_type: str, parameters: dict) -> dict:
    """
    Perform statistical calculations using the factory pattern.
    
    Args:
        suggested_study_type: Study type from LLM response
        parameters: Parameters from LLM response
        
    Returns:
        Dictionary with calculation results
    """
    try:
        # Get factory and map study type to test
        factory = get_factory()
        test_type = map_study_type_to_test(suggested_study_type)
        
        print(f"--- Mapping study type '{suggested_study_type}' to test '{test_type}' ---")
        
        # Get the statistical test
        stat_test = factory.get_test(test_type)
        
        # Extract common parameters
        alpha = parameters.get('alpha', 0.05)
        
        # Handle different test types with their specific parameters
        if test_type == "chi_square":
            # Chi-square test parameters
            contingency_table = parameters.get('contingency_table')
            expected_frequencies = parameters.get('expected_frequencies')
            effect_size = parameters.get('effect_size') or parameters.get('effect_size_value')
            total_n = parameters.get('total_n')
            
            # Check for required chi-square parameters
            if contingency_table is None:
                return {
                    'calculated_p_value': None,
                    'calculated_power': None,
                    'statistical_test_used': test_type,
                    'calculation_error': "Missing required parameter 'contingency_table' for chi-square test. Expected a 2D array of observed frequencies."
                }
            
            # Prepare chi-square parameters
            chi_params = {'contingency_table': contingency_table, 'alpha': alpha}
            if expected_frequencies is not None:
                chi_params['expected_frequencies'] = expected_frequencies
            if effect_size is not None:
                chi_params['effect_size'] = effect_size
            if total_n is not None:
                chi_params['total_n'] = total_n
            
            # Calculate p-value and power for chi-square
            p_value, p_error = stat_test.calculate_p_value(**chi_params)
            power, power_error = stat_test.calculate_power(**chi_params)
            
        elif test_type == "one_way_anova":
            # One-way ANOVA test parameters
            groups = parameters.get('groups')
            effect_size = parameters.get('effect_size') or parameters.get('effect_size_value')
            total_n = parameters.get('total_n')
            
            # Check for groups data or individual group parameters
            group_params = {k: v for k, v in parameters.items() if k.startswith("group") and k[5:].isdigit()}
            
            if groups is None and not group_params:
                return {
                    'calculated_p_value': None,
                    'calculated_power': None,
                    'statistical_test_used': test_type,
                    'calculation_error': "Missing required parameter 'groups' for ANOVA test. Expected a list of groups or individual group1, group2, etc. parameters."
                }
            
            # Prepare ANOVA parameters
            anova_params = {'alpha': alpha}
            if groups is not None:
                anova_params['groups'] = groups
            else:
                # Add individual group parameters
                for key, value in group_params.items():
                    anova_params[key] = value
            
            if effect_size is not None:
                anova_params['effect_size'] = effect_size
            if total_n is not None:
                anova_params['total_n'] = total_n
            
            # Calculate p-value and power for ANOVA
            p_value, p_error = stat_test.calculate_p_value(**anova_params)
            power, power_error = stat_test.calculate_power(**anova_params)
            
        elif test_type == "correlation":
            # Correlation test parameters
            x_values = parameters.get('x_values')
            y_values = parameters.get('y_values')
            correlation_type = parameters.get('correlation_type', 'pearson')
            effect_size = parameters.get('effect_size') or parameters.get('effect_size_value')
            n = parameters.get('n') or parameters.get('total_n')
            
            if x_values is None or y_values is None:
                return {
                    'calculated_p_value': None,
                    'calculated_power': None,
                    'statistical_test_used': test_type,
                    'calculation_error': "Missing required parameters 'x_values' and 'y_values' for correlation test. Expected arrays of paired numeric values."
                }
            
            # Prepare correlation parameters
            corr_params = {
                'x_values': x_values,
                'y_values': y_values,
                'correlation_type': correlation_type,
                'alpha': alpha
            }
            
            if effect_size is not None:
                corr_params['effect_size'] = effect_size
            if n is not None:
                corr_params['n'] = n
            
            # Calculate p-value and power for correlation
            p_value, p_error = stat_test.calculate_p_value(**corr_params)
            power, power_error = stat_test.calculate_power(**corr_params)
            
        else:
            # Two-sample t-test and other tests (existing logic)
            N_total = parameters.get('total_n')
            cohens_d = parameters.get('cohens_d') or parameters.get('effect_size_value')
            
            if N_total is None or cohens_d is None:
                return {
                    'calculated_p_value': None,
                    'calculated_power': None,
                    'statistical_test_used': test_type,
                    'calculation_error': "Missing required parameters (total_n or cohens_d) for statistical calculations"
                }
            
            # Calculate p-value and power for t-test
            p_value, p_error = stat_test.calculate_p_value(N_total=N_total, cohens_d=cohens_d)
            power, power_error = stat_test.calculate_power(N_total=N_total, cohens_d=cohens_d, alpha=alpha)
        
        # Combine any errors
        error_msg = None
        if p_error and power_error:
            error_msg = f"P-value error: {p_error}; Power error: {power_error}"
        elif p_error:
            error_msg = f"P-value error: {p_error}"
        elif power_error:
            error_msg = f"Power error: {power_error}"
        
        return {
            'calculated_p_value': p_value,
            'calculated_power': power,
            'statistical_test_used': test_type,
            'calculation_error': error_msg
        }
        
    except Exception as e:
        print(f"--- Error in statistical calculations: {e} ---")
        return {
            'calculated_p_value': None,
            'calculated_power': None,
            'statistical_test_used': test_type if 'test_type' in locals() else suggested_study_type,
            'calculation_error': f"Statistical calculation error: {str(e)}"
        }


# --- Enhanced Universal Endpoint (Task 2.4 Completion) ---
@app.post("/process_idea", response_model=StatisticalAnalysisOutput)
async def process_idea_enhanced(item: EnhancedIdeaInput):
    """
    Enhanced universal endpoint that processes any study description and returns
    complete statistical analysis with appropriate test routing.
    
    This is the main endpoint for Task 2.4 that handles all study types:
    - Two-sample t-test (treatment vs control)
    - Chi-square test (categorical associations) 
    - One-way ANOVA (multiple group comparisons)
    - Correlation test (relationships between variables)
    """
    if not item.study_description or not item.study_description.strip():
        raise HTTPException(status_code=400, detail="study_description must be provided and non-empty.")
    
    # Use provided LLM provider or default
    provider_used = item.llm_provider.upper() if item.llm_provider else ACTIVE_LLM_PROVIDER
    print(f"--- Processing enhanced idea with provider: {provider_used} ---")
    print(f"--- Study description: {item.study_description[:100]}... ---")
    
    try:
        llm_response_data = {}
        
        # Get enhanced LLM analysis
        if provider_used == "GEMINI":
            llm_response_data = await get_llm_enhanced_analysis_gemini(item.study_description)
        elif provider_used == "OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(item.study_description, is_azure=False)
        elif provider_used == "AZURE_OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(item.study_description, is_azure=True)
        elif provider_used == "ANTHROPIC":
            llm_response_data = await get_llm_enhanced_analysis_anthropic(item.study_description)
        else:
            return StatisticalAnalysisOutput(
                error=f"Unsupported LLM provider: {provider_used}", 
                llm_provider_used=provider_used,
                processed_idea=item.study_description
            )

        # Validate and extract the enhanced response
        validated_data = validate_and_extract_enhanced_response(llm_response_data)
        
        # Perform statistical calculations using factory pattern
        suggested_study_type = validated_data['suggested_study_type']
        parameters = validated_data['parameters']
        
        print(f"--- Performing statistical calculations for study type: {suggested_study_type} ---")
        calculation_results = perform_statistical_calculations(suggested_study_type, parameters)
        
        return StatisticalAnalysisOutput(
            suggested_study_type=validated_data['suggested_study_type'],
            rationale=validated_data['rationale'],
            parameters=validated_data['parameters'],
            alternative_tests=validated_data['alternative_tests'],
            data_type=validated_data['data_type'],
            study_design=validated_data['study_design'],
            confidence_level=validated_data['confidence_level'],
            # Statistical calculations from factory pattern
            calculated_p_value=calculation_results['calculated_p_value'],
            calculated_power=calculation_results['calculated_power'],
            statistical_test_used=calculation_results['statistical_test_used'],
            calculation_error=calculation_results['calculation_error'],
            # Backwards compatibility fields
            initial_N=validated_data['initial_N'],
            initial_cohens_d=validated_data['initial_cohens_d'],
            estimation_justification=validated_data['estimation_justification'],
            references=validated_data['references'],
            processed_idea=item.study_description,
            llm_provider_used=provider_used
        )

    except ValueError as ve:
        print(f"--- ValueError during enhanced processing ({provider_used}): {ve} ---")
        return StatisticalAnalysisOutput(
            error=str(ve), 
            processed_idea=item.study_description, 
            llm_provider_used=provider_used
        )
    except json.JSONDecodeError as je:
        print(f"--- JSONDecodeError from enhanced processing ({provider_used}): {je} ---")
        return StatisticalAnalysisOutput(
            error=f"Failed to parse JSON response from LLM: {str(je)}", 
            processed_idea=item.study_description, 
            llm_provider_used=provider_used
        )
    except Exception as e:
        print(f"--- An unexpected error occurred with enhanced processing {provider_used}: {e} ---")
        import traceback
        traceback.print_exc()
        return StatisticalAnalysisOutput(
            error=f"An unexpected error occurred with {provider_used}: {str(e)}", 
            processed_idea=item.study_description, 
            llm_provider_used=provider_used
        )


# --- Legacy Main Endpoint (Backwards Compatibility) ---
@app.post("/process_idea_legacy", response_model=EstimationOutput)
async def process_idea_legacy(item: IdeaInput):
    """Legacy endpoint for backwards compatibility with existing clients"""
    if not item.text_idea and not item.url_idea:
        raise HTTPException(status_code=400, detail="Either text_idea or url_idea must be provided.")

    combined_idea = item.text_idea if item.text_idea else ""
    if item.url_idea:
        url_content = fetch_url_content(item.url_idea)
        combined_idea += f"\n\nContent from URL ({item.url_idea}):\n{url_content}"
    
    if not combined_idea.strip():
        return EstimationOutput(error="Could not extract any content to process.")

    # This uses the ACTIVE_LLM_PROVIDER defined at the module level
    provider_used_for_this_request = ACTIVE_LLM_PROVIDER 
    print(f"--- Processing legacy request with provider: {provider_used_for_this_request} ---")

    try:
        llm_response_data = {}
        
        if provider_used_for_this_request == "GEMINI":
            llm_response_data = await get_llm_estimate_gemini(combined_idea)
        elif provider_used_for_this_request == "OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=False)
        elif provider_used_for_this_request == "AZURE_OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=True)
        elif provider_used_for_this_request == "ANTHROPIC":
            llm_response_data = await get_llm_estimate_anthropic(combined_idea)
        else:
            # This case should ideally not be hit if ACTIVE_LLM_PROVIDER is validated at startup,
            # but good as a safeguard.
            print(f"--- ERROR: Unsupported LLM provider configured: {provider_used_for_this_request} ---")
            return EstimationOutput(error=f"Unsupported LLM provider: {provider_used_for_this_request}", llm_provider_used=provider_used_for_this_request)

        initial_N = llm_response_data.get("N_total")
        initial_cohens_d = llm_response_data.get("cohens_d")
        justification = llm_response_data.get("justification")
        references = llm_response_data.get("references")

        if isinstance(initial_N, int) and \
           isinstance(initial_cohens_d, (float, int)) and \
           isinstance(justification, str):
            return EstimationOutput(
                initial_N=initial_N,
                initial_cohens_d=float(initial_cohens_d),
                estimation_justification=justification,
                references=references if isinstance(references, list) else None,
                processed_idea=combined_idea,
                llm_provider_used=provider_used_for_this_request
            )
        else:
            print(f"--- ERROR: LLM ({provider_used_for_this_request}) returned data in unexpected format: {llm_response_data} ---")
            return EstimationOutput(error="LLM returned data in unexpected format or missing fields.", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)

    except ValueError as ve: 
        print(f"--- ValueError during LLM call ({provider_used_for_this_request}): {ve} ---")
        return EstimationOutput(error=str(ve), processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except json.JSONDecodeError as je:
        print(f"--- JSONDecodeError from LLM ({provider_used_for_this_request}): {je} ---")
        return EstimationOutput(error=f"Failed to parse JSON response from LLM: {str(je)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except Exception as e:
        print(f"--- An unexpected error occurred with {provider_used_for_this_request}: {e} ---")
        import traceback
        traceback.print_exc()
        return EstimationOutput(error=f"An unexpected error occurred with {provider_used_for_this_request}: {str(e)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)


# --- Enhanced Analysis Endpoint (Phase 1.1) ---
@app.post("/analyze_study", response_model=StudyAnalysisOutput)
async def analyze_study(item: IdeaInput):
    """Enhanced endpoint that provides comprehensive study analysis and statistical test recommendations"""
    if not item.text_idea and not item.url_idea:
        raise HTTPException(status_code=400, detail="Either text_idea or url_idea must be provided.")

    combined_idea = item.text_idea if item.text_idea else ""
    if item.url_idea:
        url_content = fetch_url_content(item.url_idea)
        combined_idea += f"\n\nContent from URL ({item.url_idea}):\n{url_content}"
    
    if not combined_idea.strip():
        return StudyAnalysisOutput(error="Could not extract any content to process.")

    provider_used_for_this_request = ACTIVE_LLM_PROVIDER
    print(f"--- Processing enhanced analysis request with provider: {provider_used_for_this_request} ---")

    try:
        llm_response_data = {}
        
        if provider_used_for_this_request == "GEMINI":
            llm_response_data = await get_llm_enhanced_analysis_gemini(combined_idea)
        elif provider_used_for_this_request == "OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(combined_idea, is_azure=False)
        elif provider_used_for_this_request == "AZURE_OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(combined_idea, is_azure=True)
        elif provider_used_for_this_request == "ANTHROPIC":
            llm_response_data = await get_llm_enhanced_analysis_anthropic(combined_idea)
        else:
            print(f"--- ERROR: Unsupported LLM provider configured: {provider_used_for_this_request} ---")
            return StudyAnalysisOutput(error=f"Unsupported LLM provider: {provider_used_for_this_request}", llm_provider_used=provider_used_for_this_request)

        # Validate and extract the enhanced response
        validated_data = validate_and_extract_enhanced_response(llm_response_data)
        
        return StudyAnalysisOutput(
            suggested_study_type=validated_data['suggested_study_type'],
            rationale=validated_data['rationale'],
            parameters=validated_data['parameters'],
            alternative_tests=validated_data['alternative_tests'],
            data_type=validated_data['data_type'],
            study_design=validated_data['study_design'],
            confidence_level=validated_data['confidence_level'],
            initial_N=validated_data['initial_N'],
            initial_cohens_d=validated_data['initial_cohens_d'],
            estimation_justification=validated_data['estimation_justification'],
            references=validated_data['references'],
            processed_idea=combined_idea,
            llm_provider_used=provider_used_for_this_request
        )

    except ValueError as ve:
        print(f"--- ValueError during enhanced LLM call ({provider_used_for_this_request}): {ve} ---")
        return StudyAnalysisOutput(error=str(ve), processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except json.JSONDecodeError as je:
        print(f"--- JSONDecodeError from enhanced LLM ({provider_used_for_this_request}): {je} ---")
        return StudyAnalysisOutput(error=f"Failed to parse JSON response from LLM: {str(je)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except Exception as e:
        print(f"--- An unexpected error occurred with enhanced analysis {provider_used_for_this_request}: {e} ---")
        import traceback
        traceback.print_exc()
        return StudyAnalysisOutput(error=f"An unexpected error occurred with {provider_used_for_this_request}: {str(e)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)


# --- Enhanced Analysis with Statistical Calculations Endpoint (Phase 1.2) ---
@app.post("/analyze_study_complete", response_model=StatisticalAnalysisOutput)
async def analyze_study_complete(item: IdeaInput):
    """
    Enhanced endpoint that provides comprehensive study analysis with statistical calculations.
    
    This endpoint combines Phase 1.1 LLM analysis with Phase 1.2 factory pattern
    statistical calculations, providing both study recommendations and computed p-values/power.
    """
    if not item.text_idea and not item.url_idea:
        raise HTTPException(status_code=400, detail="Either text_idea or url_idea must be provided.")

    combined_idea = item.text_idea if item.text_idea else ""
    if item.url_idea:
        url_content = fetch_url_content(item.url_idea)
        combined_idea += f"\n\nContent from URL ({item.url_idea}):\n{url_content}"
    
    if not combined_idea.strip():
        return StatisticalAnalysisOutput(error="Could not extract any content to process.")

    provider_used_for_this_request = ACTIVE_LLM_PROVIDER
    print(f"--- Processing complete analysis request with provider: {provider_used_for_this_request} ---")

    try:
        llm_response_data = {}
        
        # Get LLM analysis (same as analyze_study endpoint)
        if provider_used_for_this_request == "GEMINI":
            llm_response_data = await get_llm_enhanced_analysis_gemini(combined_idea)
        elif provider_used_for_this_request == "OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(combined_idea, is_azure=False)
        elif provider_used_for_this_request == "AZURE_OPENAI":
            llm_response_data = await get_llm_enhanced_analysis_openai(combined_idea, is_azure=True)
        elif provider_used_for_this_request == "ANTHROPIC":
            llm_response_data = await get_llm_enhanced_analysis_anthropic(combined_idea)
        else:
            print(f"--- ERROR: Unsupported LLM provider configured: {provider_used_for_this_request} ---")
            return StatisticalAnalysisOutput(error=f"Unsupported LLM provider: {provider_used_for_this_request}", llm_provider_used=provider_used_for_this_request)

        # Validate and extract the enhanced response
        validated_data = validate_and_extract_enhanced_response(llm_response_data)
        
        # Perform statistical calculations using factory pattern
        suggested_study_type = validated_data['suggested_study_type']
        parameters = validated_data['parameters']
        
        print(f"--- Performing statistical calculations for study type: {suggested_study_type} ---")
        calculation_results = perform_statistical_calculations(suggested_study_type, parameters)
        
        return StatisticalAnalysisOutput(
            suggested_study_type=validated_data['suggested_study_type'],
            rationale=validated_data['rationale'],
            parameters=validated_data['parameters'],
            alternative_tests=validated_data['alternative_tests'],
            data_type=validated_data['data_type'],
            study_design=validated_data['study_design'],
            confidence_level=validated_data['confidence_level'],
            # Statistical calculations from factory pattern
            calculated_p_value=calculation_results['calculated_p_value'],
            calculated_power=calculation_results['calculated_power'],
            statistical_test_used=calculation_results['statistical_test_used'],
            calculation_error=calculation_results['calculation_error'],
            # Backwards compatibility fields
            initial_N=validated_data['initial_N'],
            initial_cohens_d=validated_data['initial_cohens_d'],
            estimation_justification=validated_data['estimation_justification'],
            references=validated_data['references'],
            processed_idea=combined_idea,
            llm_provider_used=provider_used_for_this_request
        )

    except ValueError as ve:
        print(f"--- ValueError during complete analysis ({provider_used_for_this_request}): {ve} ---")
        return StatisticalAnalysisOutput(error=str(ve), processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except json.JSONDecodeError as je:
        print(f"--- JSONDecodeError from complete analysis ({provider_used_for_this_request}): {je} ---")
        return StatisticalAnalysisOutput(error=f"Failed to parse JSON response from LLM: {str(je)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except Exception as e:
        print(f"--- An unexpected error occurred with complete analysis {provider_used_for_this_request}: {e} ---")
        import traceback
        traceback.print_exc()
        return StatisticalAnalysisOutput(error=f"An unexpected error occurred with {provider_used_for_this_request}: {str(e)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)


# --- Enhanced Factory Information Endpoint (Task 2.4) ---
@app.get("/available_tests")
async def get_available_tests():
    """
    Get comprehensive list of available statistical tests with descriptions and parameters.
    This endpoint provides UI-ready information for dropdown menus and form generation.
    """
    try:
        factory = get_factory()
        available_tests = factory.get_available_tests()
        
        # Enhanced test information for UI consumption
        test_details = {
            "two_sample_t_test": {
                "name": "Two-Sample t-Test",
                "description": "Compare means between two independent groups (e.g., treatment vs control)",
                "data_type": "continuous",
                "study_design": ["randomized_controlled_trial", "cohort_study"],
                "required_parameters": ["N_total", "cohens_d"],
                "optional_parameters": ["alpha", "power"],
                "effect_size_type": "cohens_d",
                "aliases": ["t_test", "independent_samples_t_test", "unpaired_t_test"]
            },
            "chi_square": {
                "name": "Chi-Square Test",
                "description": "Test associations between categorical variables (independence or goodness-of-fit)",
                "data_type": "categorical",
                "study_design": ["cross_sectional", "case_control_study", "cohort_study"],
                "required_parameters": ["contingency_table"],
                "optional_parameters": ["expected_frequencies", "effect_size", "alpha"],
                "effect_size_type": "cramers_v",
                "aliases": ["chi2", "categorical_test", "independence_test", "association_test"]
            },
            "one_way_anova": {
                "name": "One-Way ANOVA",
                "description": "Compare means across multiple groups (3 or more groups)",
                "data_type": "continuous",
                "study_design": ["randomized_controlled_trial", "cross_sectional"],
                "required_parameters": ["groups"],
                "optional_parameters": ["effect_size", "alpha", "total_n"],
                "effect_size_type": "eta_squared",
                "aliases": ["anova", "f_test", "multiple_groups", "compare_multiple_groups"]
            },
            "correlation": {
                "name": "Correlation Analysis",
                "description": "Analyze relationships between two continuous variables (Pearson or Spearman)",
                "data_type": "continuous",
                "study_design": ["cross_sectional", "cohort_study"],
                "required_parameters": ["x_values", "y_values"],
                "optional_parameters": ["correlation_type", "effect_size", "alpha", "n"],
                "effect_size_type": "correlation_coefficient",
                "aliases": ["pearson", "spearman", "relationship", "correlation_test"]
            }
        }
        
        # Build response with enhanced information
        enhanced_tests = []
        for test_name in available_tests:
            if test_name in test_details:
                enhanced_tests.append({
                    "test_id": test_name,
                    **test_details[test_name]
                })
        
        return {
            "available_tests": available_tests,
            "enhanced_test_info": enhanced_tests,
            "factory_status": "operational",
            "total_tests": len(available_tests),
            "supported_study_types": [
                "two_group_comparison",
                "categorical_association", 
                "multiple_group_comparison",
                "relationship_analysis"
            ],
            "supported_data_types": ["continuous", "categorical", "count"],
            "api_version": "2.4"
        }
    except Exception as e:
        return {
            "available_tests": [],
            "enhanced_test_info": [],
            "factory_status": "error",
            "error": str(e),
            "api_version": "2.4"
        }


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "provider": ACTIVE_LLM_PROVIDER}

if __name__ == "__main__":
    # This block is for direct execution (python api.py), not when run by uvicorn
    # Uvicorn handles the app loading.
    print("--- Starting API directly (not via uvicorn, for testing only if needed) ---")
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
