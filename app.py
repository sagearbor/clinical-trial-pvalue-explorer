# app.py - Enhanced Universal Study P-Value Explorer
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
sys.path.insert(0, 'src')
from statistical_utils import calculate_p_value_from_N_d, calculate_power_from_N_d

# Research intelligence is now loaded in background by the API backend

# Configuration for the FastAPI backend URLs
# Assumes FastAPI is running on localhost:8000
BASE_URL = "http://localhost:8000"
BACKEND_URL = f"{BASE_URL}/process_idea"
AVAILABLE_TESTS_URL = f"{BASE_URL}/available_tests"
CALCULATE_STATISTICS_URL = f"{BASE_URL}/calculate_statistics"  # New endpoint for direct calculations
MULTI_SCENARIO_URL = f"{BASE_URL}/analyze_scenarios"  # Multi-scenario analysis endpoint


st.set_page_config(layout="wide", page_title="Universal Study P-Value Explorer", page_icon="📊")
st.title("📊 Universal Study P-Value Explorer (AI-Assisted)")
st.markdown("*Supports T-tests, Chi-square, ANOVA, and Correlation Analysis*")

# Initialize session state variables if they don't exist
default_N = 100
default_d = 0.5

# Original session state variables (backwards compatibility)
if 'initial_N' not in st.session_state:
    st.session_state.initial_N = default_N
if 'initial_cohens_d' not in st.session_state:
    st.session_state.initial_cohens_d = default_d
if 'estimation_justification' not in st.session_state:
    st.session_state.estimation_justification = ""
if 'processed_idea_text' not in st.session_state:
    st.session_state.processed_idea_text = ""
if 'current_p_value' not in st.session_state:
    st.session_state.current_p_value = None
if 'p_value_message' not in st.session_state:
    st.session_state.p_value_message = ""
if 'llm_provider_used' not in st.session_state:
    st.session_state.llm_provider_used = ""
if 'references' not in st.session_state:
    st.session_state.references = []
if 'current_N' not in st.session_state:
    st.session_state.current_N = st.session_state.initial_N
if 'current_d' not in st.session_state:
    st.session_state.current_d = st.session_state.initial_cohens_d

# Enhanced session state variables for Phase 2
if 'study_analysis' not in st.session_state:
    st.session_state.study_analysis = {}
if 'suggested_test_type' not in st.session_state:
    st.session_state.suggested_test_type = "two_sample_t_test"
if 'selected_test_type' not in st.session_state:
    st.session_state.selected_test_type = "two_sample_t_test"
if 'available_tests' not in st.session_state:
    st.session_state.available_tests = []
if 'test_parameters' not in st.session_state:
    st.session_state.test_parameters = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'show_ai_suggestion' not in st.session_state:
    st.session_state.show_ai_suggestion = True
if 'calculated_statistics' not in st.session_state:
    st.session_state.calculated_statistics = {}
if 'current_contingency_table' not in st.session_state:
    st.session_state.current_contingency_table = [[25, 15], [20, 30]]
if 'current_anova_groups' not in st.session_state:
    st.session_state.current_anova_groups = []
if 'current_x_values' not in st.session_state:
    st.session_state.current_x_values = []
if 'current_y_values' not in st.session_state:
    st.session_state.current_y_values = []
if 'current_correlation_type' not in st.session_state:
    st.session_state.current_correlation_type = "pearson"

# Multi-scenario session state variables
if 'multi_scenarios' not in st.session_state:
    st.session_state.multi_scenarios = {}
if 'evidence_quality' not in st.session_state:
    st.session_state.evidence_quality = None
if 'effect_size_uncertainty' not in st.session_state:
    st.session_state.effect_size_uncertainty = None
if 'recommended_scenario' not in st.session_state:
    st.session_state.recommended_scenario = None
if 'selected_scenario' not in st.session_state:
    st.session_state.selected_scenario = None
if 'scenario_analysis_complete' not in st.session_state:
    st.session_state.scenario_analysis_complete = False
if 'selected_llm_provider' not in st.session_state:
    st.session_state.selected_llm_provider = None


# Helper functions for enhanced features
def fetch_available_tests():
    """Fetch available statistical tests from API"""
    try:
        response = requests.get(AVAILABLE_TESTS_URL, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data.get('enhanced_test_info', []), data.get('available_tests', [])
        else:
            st.error(f"Failed to fetch available tests. Status: {response.status_code}")
            return [], []
    except Exception as e:
        st.error(f"Error fetching available tests: {e}")
        return [], []

def get_test_display_name(test_id):
    """Get human-readable display name for test"""
    test_names = {
        "two_sample_t_test": "Two-Sample t-Test",
        "chi_square": "Chi-Square Test", 
        "one_way_anova": "One-Way ANOVA",
        "correlation": "Correlation Analysis",
        "ancova": "ANCOVA (Analysis of Covariance)",
        "fishers_exact": "Fisher's Exact Test",
        "logistic_regression": "Logistic Regression",
        "repeated_measures_anova": "Repeated Measures ANOVA",
        "survival_analysis": "🚫 Survival Analysis (Not Yet Supported)"
    }
    return test_names.get(test_id, test_id.replace('_', ' ').title())

def get_effect_size_interpretation(test_type, effect_size):
    """Get interpretation for effect sizes"""
    if test_type == "two_sample_t_test" and effect_size is not None:
        if abs(effect_size) < 0.2:
            return "Very Small Effect"
        elif abs(effect_size) < 0.5:
            return "Small Effect"
        elif abs(effect_size) < 0.8:
            return "Medium Effect"
        else:
            return "Large Effect"
    elif test_type == "correlation" and effect_size is not None:
        abs_r = abs(effect_size)
        if abs_r < 0.1:
            return "Very Weak Correlation"
        elif abs_r < 0.3:
            return "Weak Correlation"
        elif abs_r < 0.5:
            return "Moderate Correlation"
        elif abs_r < 0.7:
            return "Strong Correlation"
        else:
            return "Very Strong Correlation"
    elif test_type == "chi_square" and effect_size is not None:
        if effect_size < 0.1:
            return "Very Small Association"
        elif effect_size < 0.3:
            return "Small Association"
        elif effect_size < 0.5:
            return "Medium Association"
        else:
            return "Large Association"
    elif test_type == "one_way_anova" and effect_size is not None:
        if effect_size < 0.01:
            return "Very Small Effect"
        elif effect_size < 0.06:
            return "Small Effect"
        elif effect_size < 0.14:
            return "Medium Effect"
        else:
            return "Large Effect"
    return "Unknown"

def calculate_test_specific_statistics(test_type, parameters):
    """Calculate statistics for specific test types using backend API"""
    try:
        # For now, use the main process_idea endpoint with a mock study description
        # until the dedicated statistics endpoint is implemented
        mock_description = f"Calculate {test_type} statistics with parameters: {json.dumps(parameters)}"
        
        payload = {
            "study_description": mock_description,
            "test_override": test_type,
            "parameters_override": parameters
        }
        
        response = requests.post(BACKEND_URL, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            # Extract statistical results
            return {
                "p_value": data.get('calculated_p_value'),
                "power": data.get('calculated_power'),
                "effect_size": data.get('parameters', {}).get('effect_size_value'),
                "test_statistic": data.get('test_statistic'),
                "sample_size": data.get('parameters', {}).get('total_n'),
                "error": data.get('calculation_error')
            }
        else:
            return {"error": f"Backend calculation failed: {response.status_code}"}
    except Exception as e:
        # Fallback to local calculation for t-test
        if test_type == "two_sample_t_test":
            try:
                N_total = parameters.get('N_total')
                cohens_d = parameters.get('cohens_d')
                alpha = parameters.get('alpha', 0.05)
                
                if N_total and cohens_d:
                    p_val, p_msg = calculate_p_value_from_N_d(N_total, cohens_d)
                    power_val, power_msg = calculate_power_from_N_d(N_total, cohens_d, alpha)
                    
                    return {
                        "p_value": p_val,
                        "power": power_val,
                        "effect_size": cohens_d,
                        "sample_size": N_total,
                        "error": p_msg or power_msg
                    }
            except:
                pass
                
        return {"error": f"Error calculating statistics: {str(e)}"}

def format_test_results(test_type, results):
    """Format results display based on test type"""
    if not results or results.get('error'):
        return results
    
    formatted = {
        'p_value': results.get('p_value'),
        'power': results.get('power'),
        'effect_size': results.get('effect_size'),
        'test_statistic': results.get('test_statistic'),
        'degrees_of_freedom': results.get('degrees_of_freedom'),
        'sample_size': results.get('sample_size'),
        'interpretation': results.get('interpretation', {})
    }
    
    # Test-specific formatting
    if test_type == "chi_square":
        formatted['cramers_v'] = results.get('cramers_v')
        formatted['contingency_table'] = results.get('contingency_table')
        formatted['expected_frequencies'] = results.get('expected_frequencies')
    elif test_type == "one_way_anova":
        formatted['f_statistic'] = results.get('f_statistic')
        formatted['eta_squared'] = results.get('eta_squared')
        formatted['group_means'] = results.get('group_means')
    elif test_type == "correlation":
        formatted['correlation_coefficient'] = results.get('correlation_coefficient')
        formatted['correlation_type'] = results.get('correlation_type')
        
    return formatted

def call_multi_scenario_analysis(study_description: str, llm_provider: str = None):
    """Call the multi-scenario analysis endpoint"""
    max_papers = st.session_state.get('max_papers', 5)
    # Get individual source counts from session state
    pubmed_count = getattr(st.session_state, 'pubmed_papers', 3)
    arxiv_count = getattr(st.session_state, 'arxiv_papers', 2) 
    clinicaltrials_count = getattr(st.session_state, 'clinicaltrials_papers', 3)
    
    payload = {
        "study_description": study_description.strip(),
        "llm_provider": llm_provider,
        "max_papers": max_papers,
        "pubmed_papers": pubmed_count,
        "arxiv_papers": arxiv_count,
        "clinicaltrials_papers": clinicaltrials_count
    }
    
    try:
        with st.spinner("🤖 AI is generating 5 statistical design scenarios based on effect size uncertainty..."):
            response = requests.post(MULTI_SCENARIO_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("error"):
                st.error(f"Error from multi-scenario analysis: {data['error']}")
                return None
            else:
                return data
        else:
            st.error(f"Failed to get multi-scenario analysis. Status: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the backend API. Is it running at " + BASE_URL + "?")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Multi-scenario analysis timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error during multi-scenario analysis: {e}")
        return None

def create_scenario_comparison_chart(scenarios: dict, recommended_scenario: str = None):
    """Create interactive Plotly chart comparing all 5 scenarios"""
    if not scenarios:
        return None
    
    # Extract data for visualization
    scenario_names = []
    sample_sizes = []
    p_values = []
    powers = []
    effect_sizes = []
    colors = []
    
    # Define colors for each scenario
    color_map = {
        'exploratory': '#FF6B6B',      # Red - most rigorous
        'cautious': '#FF8E53',         # Orange
        'standard': '#4ECDC4',         # Teal - balanced 
        'optimistic': '#45B7D1',      # Blue
        'minimum_viable': '#96CEB4'    # Green - most efficient
    }
    
    for name, scenario in scenarios.items():
        if isinstance(scenario, dict) and 'parameters' in scenario:
            scenario_names.append(name.replace('_', ' ').title())
            
            params = scenario['parameters']
            sample_sizes.append(params.get('total_n', 0))
            p_values.append(scenario.get('target_p_value', params.get('alpha', 0.05)))
            powers.append(scenario.get('target_power', params.get('power', 0.8)))
            effect_sizes.append(params.get('effect_size_value', 0.5))
            
            # Highlight recommended scenario
            if name == recommended_scenario:
                colors.append('#FFD700')  # Gold for recommended
            else:
                colors.append(color_map.get(name, '#999999'))
    
    # Create subplot with multiple charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sample Size Comparison', 'Statistical Rigor (α level)', 
                       'Target Power', 'Effect Size Assumptions'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sample sizes - smooth curve with points
    fig.add_trace(
        go.Scatter(x=scenario_names, y=sample_sizes, 
                   mode='lines+markers+text',
                   line=dict(color='#2E86C1', width=3, shape='spline'),
                   marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                   text=[f'N={n}' for n in sample_sizes],
                   textposition='top center',
                   name='Sample Size'),
        row=1, col=1
    )
    
    # P-value targets - smooth curve with points (lower is more rigorous)
    fig.add_trace(
        go.Scatter(x=scenario_names, y=p_values,
                   mode='lines+markers+text',
                   line=dict(color='#E74C3C', width=3, shape='spline'),
                   marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                   text=[f'α={p}' for p in p_values],
                   textposition='top center',
                   name='Target α'),
        row=1, col=2
    )
    
    # Power targets - smooth curve with points
    fig.add_trace(
        go.Scatter(x=scenario_names, y=[p*100 for p in powers],
                   mode='lines+markers+text',
                   line=dict(color='#27AE60', width=3, shape='spline'),
                   marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                   text=[f'{p*100:.0f}%' for p in powers],
                   textposition='top center',
                   name='Power (%)'),
        row=2, col=1
    )
    
    # Effect sizes - smooth curve with points
    fig.add_trace(
        go.Scatter(x=scenario_names, y=effect_sizes,
                   mode='lines+markers+text',
                   line=dict(color='#8E44AD', width=3, shape='spline'),
                   marker=dict(size=12, color=colors, line=dict(width=2, color='white')),
                   text=[f'd={e:.2f}' for e in effect_sizes],
                   textposition='top center',
                   name='Effect Size'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="📊 Multi-Scenario Comparison Dashboard",
        title_x=0.5,
        showlegend=False,
        font=dict(size=12)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Participants", row=1, col=1)
    fig.update_yaxes(title_text="Alpha Level", row=1, col=2, type="log")
    fig.update_yaxes(title_text="Power (%)", row=2, col=1)
    fig.update_yaxes(title_text="Effect Size", row=2, col=2)
    
    return fig

def create_power_curves_comparison(scenarios: dict, recommended_scenario: str = None):
    """Create power curves showing how power changes with sample size for each scenario"""
    if not scenarios:
        return None
        
    fig = go.Figure()
    
    # Generate sample size range for power curves
    n_range = np.arange(20, 500, 10)
    
    color_map = {
        'exploratory': '#FF6B6B',
        'cautious': '#FF8E53', 
        'standard': '#4ECDC4',
        'optimistic': '#45B7D1',
        'minimum_viable': '#96CEB4'
    }
    
    for name, scenario in scenarios.items():
        # Debug: Check what type of data we're getting
        print(f"Debug: Scenario {name} type: {type(scenario)}")
        if isinstance(scenario, dict):
            print(f"Debug: Scenario {name} keys: {list(scenario.keys())}")
        else:
            print(f"Debug: Scenario {name} value: {scenario}")
            continue  # Skip non-dict scenarios
        
        if isinstance(scenario, dict) and 'parameters' in scenario:
            params = scenario['parameters']
            effect_size = params.get('effect_size_value', 0.5)
            alpha = scenario.get('target_p_value', params.get('alpha', 0.05))
            
            # Calculate power curve for this scenario
            power_values = []
            for n in n_range:
                power, _ = calculate_power_from_N_d(n, effect_size, alpha)
                power_values.append(power if power else 0)
            
            # Line style for recommended scenario
            line_width = 4 if name == recommended_scenario else 2
            line_dash = 'solid' if name == recommended_scenario else 'dot'
            
            fig.add_trace(go.Scatter(
                x=n_range, 
                y=power_values,
                mode='lines+markers',
                name=f'{name.replace("_", " ").title()}',
                line=dict(
                    color=color_map.get(name, '#999999'),
                    width=line_width,
                    dash=line_dash,
                    shape='spline'
                ),
                marker=dict(size=4, opacity=0.6),
                hovertemplate=f'<b>{name.replace("_", " ").title()}</b><br>' +
                             'Sample Size: %{x}<br>' +
                             'Power: %{y:.3f}<br>' +
                             f'α = {alpha}, Effect = {effect_size}<extra></extra>'
            ))
    
    # Add horizontal lines for common power thresholds
    fig.add_hline(y=0.8, line_dash="dash", line_color="gray", 
                  annotation_text="80% Power", annotation_position="right")
    fig.add_hline(y=0.9, line_dash="dash", line_color="lightgray",
                  annotation_text="90% Power", annotation_position="right")
    
    fig.update_layout(
        title="🔥 Power Curves Comparison Across Scenarios",
        title_x=0.5,
        xaxis_title="Sample Size (N)",
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1]),
        height=500,
        hovermode='x unified',
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="right", 
            x=0.98
        )
    )
    
    return fig

with st.sidebar:
    # Clean, compact gear icon with tooltip
    st.markdown('## ⚙️ <span title="Advanced configuration settings">ℹ️</span>', unsafe_allow_html=True)
    
    # Research Intelligence Settings
    with st.expander("🔬 Research Intelligence"):
        st.markdown("**Configure Research Sources**")
        
        # PubMed row
        st.markdown("**📚 PubMed** (Medical Literature)")
        pubmed_papers = st.slider(
            "Number of PubMed papers",
            min_value=0,
            max_value=20,
            value=3,
            help="Medical literature from peer-reviewed journals",
            key="pubmed_slider"
        )
        
        # arXiv row  
        st.markdown("**🔬 arXiv** (Academic Preprints)")
        arxiv_papers = st.slider(
            "Number of arXiv papers", 
            min_value=0,
            max_value=20,
            value=2,
            help="Academic preprints and cutting-edge research",
            key="arxiv_slider"
        )
        
        # ClinicalTrials.gov row
        st.markdown("**🏥 ClinicalTrials.gov** (Clinical Trials)")
        clinicaltrials_papers = st.slider(
            "Number of clinical trials",
            min_value=0,
            max_value=20, 
            value=3,
            help="Registered clinical trials with sample sizes and outcomes",
            key="clinicaltrials_slider"
        )
        
        # Calculate total and store in session state
        total_papers = pubmed_papers + arxiv_papers + clinicaltrials_papers
        st.session_state.max_papers = total_papers
        st.session_state.pubmed_papers = pubmed_papers
        st.session_state.arxiv_papers = arxiv_papers
        st.session_state.clinicaltrials_papers = clinicaltrials_papers
        
        st.success(f"**Total: {total_papers} sources** selected")
    
    # LLM Provider Selection
    with st.expander("🤖 AI Provider"):
        llm_provider = st.selectbox(
            "Select AI Provider:",
            ["Default", "GEMINI", "OPENAI", "AZURE_OPENAI", "ANTHROPIC"],
            help="Choose specific AI provider or use default system setting",
            key="sidebar_llm_provider"
        )
        if llm_provider == "Default":
            llm_provider = None
        # Store in session state so it can be accessed later
        st.session_state.selected_llm_provider = llm_provider
    
    # Collapsed About section
    with st.expander("ℹ️ About", expanded=False):
        st.info(
            "**Clinical Trial P-Value Explorer**\n\n"
            "Supports 8 statistical tests covering 95%+ of clinical trials:\n"
            "• **T-tests & ANCOVA**: Group comparisons\n"
            "• **Chi-square & Fisher's Exact**: Categorical data\n"
            "• **ANOVA & Repeated Measures**: Multiple groups/timepoints\n"
            "• **Logistic Regression**: Binary outcomes\n"
            "• **Correlation**: Relationships\n\n"
            "🔬 **Real Research Intelligence**: Fetches actual citations from PubMed, arXiv & ClinicalTrials.gov"
            "\n\n**Disclaimer:** AI estimates are for exploratory purposes only. "
            "Consult a biostatistician for actual clinical trial design."
        )
    
    # Collapsed Statistical Test Information
    with st.expander("📊 Statistical Tests", expanded=False):
        if st.session_state.available_tests:
            for test in st.session_state.available_tests:
                with st.expander(f"📊 {test.get('name', test['test_id'])}", expanded=False):
                    st.write(f"**Description:** {test.get('description', 'No description')}")
        else:
            if st.button("🔄 Load Test Info", help="Fetch latest test information from API"):
                enhanced_tests, basic_tests = fetch_available_tests()
                st.session_state.available_tests = enhanced_tests


# Load available tests if not already loaded
if not st.session_state.available_tests:
    enhanced_tests, basic_tests = fetch_available_tests()
    st.session_state.available_tests = enhanced_tests

st.header("1. Describe Your Research Idea")
st.markdown("🤖 **AI will analyze your study and suggest the most appropriate statistical test**")

text_idea = st.text_area(
    "Enter your research idea, study description, or hypothesis:", 
    height=150, 
    key="text_idea_input_v4",
    help="Describe what you want to study, compare, or analyze. Be specific about your variables and groups."
)

# Get LLM provider from sidebar selection
llm_provider = st.session_state.get('selected_llm_provider', None)

# Single combined button with literature search option
col1, col2 = st.columns([0.8, 0.2])
with col1:
    analyze_button = st.button("🔍 Analyze Study & Get AI Recommendations", key="analyze_study_button_v5", type="primary")
with col2:
    lit_search_enabled = st.checkbox("📚 Literature Search", value=True, help="Include real research citations from PubMed, arXiv & ClinicalTrials.gov")

# Store literature search state for later reference
if 'lit_search_was_enabled' not in st.session_state:
    st.session_state.lit_search_was_enabled = False

if analyze_button:
    # Store the lit search preference for display purposes
    st.session_state.lit_search_was_enabled = lit_search_enabled
    
    if not text_idea.strip():
        st.error("Please provide a research idea or study description.")
    else:
        if lit_search_enabled:
            # Use multi-scenario analysis with literature search (full featured)
            scenario_data = call_multi_scenario_analysis(text_idea.strip(), llm_provider)
            
            if scenario_data:
                st.session_state.multi_scenario_data = scenario_data
                st.session_state.evidence_quality = scenario_data.get('evidence_quality')
                st.session_state.effect_size_uncertainty = scenario_data.get('effect_size_uncertainty')
                st.session_state.recommended_scenario = scenario_data.get('recommended_scenario')
                st.session_state.scenario_analysis_complete = True
                
                # Extract references from scenario data
                st.session_state.references = scenario_data.get('references', [])
                st.session_state.references_source = scenario_data.get('references_source', 'multi_scenario_search')
                st.session_state.references_warning = scenario_data.get('references_warning')
                
                # Extract scenarios for multi-scenario display
                scenarios = scenario_data.get('scenarios', [])
                if scenarios:
                    # Debug: Check structure of scenarios from API
                    st.write(f"Debug: API returned {len(scenarios)} scenarios")
                    for i, scenario in enumerate(scenarios):
                        st.write(f"Debug: Scenario {i+1} type: {type(scenario)}")
                        if isinstance(scenario, dict):
                            st.write(f"Debug: Scenario {i+1} keys: {list(scenario.keys())}")
                        else:
                            st.write(f"Debug: Scenario {i+1} value: {scenario}")
                    
                    # Handle different API response formats
                    if all(isinstance(s, dict) for s in scenarios):
                        # Expected format: scenarios are dictionaries
                        st.session_state.multi_scenarios = {f"scenario_{i+1}": scenario for i, scenario in enumerate(scenarios)}
                    elif all(isinstance(s, str) for s in scenarios):
                        # Fallback: scenarios are strings, create basic structure
                        st.info("ℹ️ Creating scenario structure from API response...")
                        
                        # Create basic scenario structure from names
                        scenario_templates = {
                            'exploratory': {'name': 'Exploratory', 'effect_size': 0.3, 'power': 0.6, 'description': 'Lower power, smaller effect'},
                            'cautious': {'name': 'Cautious', 'effect_size': 0.4, 'power': 0.7, 'description': 'Conservative approach'},
                            'standard': {'name': 'Standard', 'effect_size': 0.5, 'power': 0.8, 'description': 'Typical research standard'},
                            'optimistic': {'name': 'Optimistic', 'effect_size': 0.6, 'power': 0.9, 'description': 'Higher effect expectation'},
                            'minimum_viable': {'name': 'Minimum Viable', 'effect_size': 0.2, 'power': 0.5, 'description': 'Smallest detectable effect'}
                        }
                        
                        multi_scenarios = {}
                        for i, scenario_name in enumerate(scenarios):
                            template = scenario_templates.get(scenario_name, {
                                'name': scenario_name.title(), 
                                'effect_size': 0.5, 
                                'power': 0.8, 
                                'description': f'{scenario_name.title()} scenario'
                            })
                            
                            multi_scenarios[f"scenario_{i+1}"] = {
                                'name': template['name'],
                                'description': template['description'],
                                'target_p_value': 0.05,
                                'parameters': {
                                    'total_n': int(100 / (template['effect_size'] ** 2)),  # Rough estimate
                                    'effect_size_value': template['effect_size'],
                                    'effect_size_type': 'cohens_d',
                                    'alpha': 0.05,
                                    'power': template['power']
                                }
                            }
                        
                        st.session_state.multi_scenarios = multi_scenarios
                        st.success(f"✅ Created {len(multi_scenarios)} scenarios from API response")
                    else:
                        st.warning("⚠️ API returned scenarios in unexpected format - not storing for multi-scenario display")
                        st.session_state.multi_scenarios = {}
                
                # Also populate the basic analysis fields for consistency
                st.session_state.study_analysis = {
                    'suggested_study_type': scenario_data.get('suggested_study_type'),
                    'rationale': scenario_data.get('rationale'),
                    'data_type': scenario_data.get('data_type'),
                    'study_design': scenario_data.get('study_design'),
                    'alternative_tests': scenario_data.get('alternative_tests', [])
                }
                st.session_state.llm_provider_used = scenario_data.get('llm_provider_used', 'Unknown')
                
                st.success(f"✅ Complete analysis with literature search finished! (Provider: {st.session_state.llm_provider_used})")
            else:
                st.error("❌ Failed to get complete analysis. Please try again.")
        else:
            # Use basic analysis without literature search (faster)
            payload = {
                "study_description": text_idea.strip(),
                "llm_provider": llm_provider
            }
            st.session_state.llm_provider_used = "" # Reset before new call
            try:
                with st.spinner("🤖 AI is analyzing your study (no literature search)..."):
                    response = requests.post(BACKEND_URL, json=payload, timeout=120)
                
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.llm_provider_used = data.get("llm_provider_used", "Unknown")
                        st.session_state.study_analysis = data

                        if data.get("error"):
                            st.error(f"Error from AI analysis (Provider: {st.session_state.llm_provider_used}): {data['error']}")
                            st.session_state.processed_idea_text = data.get("processed_idea", st.session_state.processed_idea_text)
                            st.session_state.estimation_justification = ""
                            st.session_state.references = []
                        else:
                            # Store enhanced analysis results
                            st.session_state.suggested_test_type = data.get("suggested_study_type", "two_sample_t_test")
                            st.session_state.selected_test_type = st.session_state.suggested_test_type  # Default to AI suggestion
                            
                            # Store analysis results for display
                            st.session_state.analysis_results = {
                                'p_value': data.get('calculated_p_value'),
                                'power': data.get('calculated_power'),
                                'test_used': data.get('statistical_test_used'),
                                'calculation_error': data.get('calculation_error')
                            }
                            
                            # Store enhanced statistical results
                            st.session_state.calculated_statistics = format_test_results(
                                st.session_state.suggested_test_type, 
                                data
                            )
                            
                            # Backwards compatibility: extract basic parameters
                            if data.get("initial_N") is not None and data.get("initial_cohens_d") is not None:
                                st.session_state.initial_N = data["initial_N"]
                                st.session_state.initial_cohens_d = data["initial_cohens_d"]
                                st.session_state.current_N = data["initial_N"]
                                st.session_state.current_d = data["initial_cohens_d"]
                            
                            st.session_state.estimation_justification = data.get("estimation_justification", data.get("rationale", "No justification provided by AI."))
                            st.session_state.processed_idea_text = data.get("processed_idea", "Idea processed successfully.")
                            st.session_state.references = data.get("references", [])
                            st.session_state.test_parameters = data.get("parameters", {})
                            
                            st.success(f"✅ AI analysis complete! Suggested test: **{get_test_display_name(st.session_state.suggested_test_type)}** (Provider: {st.session_state.llm_provider_used})")
                    else:
                        st.error(f"Failed to get AI analysis from backend. Status code: {response.status_code}")
                        try:
                            error_detail = response.json().get("detail", response.text)
                            st.error(f"Details: {error_detail}")
                        except: 
                            st.error(f"Details: {response.text}")
                        st.session_state.processed_idea_text = "Failed to process idea."
                        st.session_state.estimation_justification = ""
                        st.session_state.references = []
                        st.session_state.study_analysis = {}
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to the backend API. Is it running at " + BACKEND_URL + "?")
            except requests.exceptions.Timeout:
                st.error("⏱️ The request to the backend API timed out. The AI analysis might be taking too long or the server is busy.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")
                st.session_state.estimation_justification = ""
                st.session_state.references = []
                st.session_state.study_analysis = {}

# === Separate 5-Scenario Analysis Button ===
if st.session_state.study_analysis:  # Only show after basic analysis is done
    st.markdown("---")
    st.subheader("🎲 Advanced Uncertainty Analysis")
    
    if st.button("🔍 Generate 5 Scenarios Analysis", key="generate_scenarios_button", type="secondary"):
        if not text_idea.strip():
            st.error("Please provide a research idea or study description.")
        else:
            with st.spinner("🔄 Running advanced uncertainty analysis with 5 scenarios..."):
                try:
                    # Use basic analysis without literature search for 5 scenarios
                    payload = {
                        "study_description": text_idea.strip(),
                        "llm_provider": llm_provider,
                        # Don't include literature search parameters - just uncertainty analysis
                    }
                    
                    response = requests.post(MULTI_SCENARIO_URL, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        scenario_data = response.json()
                        
                        if scenario_data and not scenario_data.get("error"):
                            # Extract scenarios for display (independent of references)
                            scenarios = scenario_data.get('scenarios', [])
                            if scenarios:
                                # Debug: Check structure of scenarios from 5-scenario API
                                st.write(f"Debug: 5-Scenario API returned {len(scenarios)} scenarios")
                                for i, scenario in enumerate(scenarios):
                                    st.write(f"Debug: Scenario {i+1} type: {type(scenario)}")
                                    if isinstance(scenario, dict):
                                        st.write(f"Debug: Scenario {i+1} keys: {list(scenario.keys())}")
                                    else:
                                        st.write(f"Debug: Scenario {i+1} value: {scenario}")
                                
                                # Handle different API response formats
                                if all(isinstance(s, dict) for s in scenarios):
                                    # Expected format: scenarios are dictionaries
                                    st.session_state.multi_scenarios = {f"scenario_{i+1}": scenario for i, scenario in enumerate(scenarios)}
                                elif all(isinstance(s, str) for s in scenarios):
                                    # Fallback: scenarios are strings, create basic structure
                                    st.info("ℹ️ Creating scenario structure from 5-scenario API response...")
                                    
                                    # Create basic scenario structure from names (same as above)
                                    scenario_templates = {
                                        'exploratory': {'name': 'Exploratory', 'effect_size': 0.3, 'power': 0.6, 'description': 'Lower power, smaller effect'},
                                        'cautious': {'name': 'Cautious', 'effect_size': 0.4, 'power': 0.7, 'description': 'Conservative approach'},
                                        'standard': {'name': 'Standard', 'effect_size': 0.5, 'power': 0.8, 'description': 'Typical research standard'},
                                        'optimistic': {'name': 'Optimistic', 'effect_size': 0.6, 'power': 0.9, 'description': 'Higher effect expectation'},
                                        'minimum_viable': {'name': 'Minimum Viable', 'effect_size': 0.2, 'power': 0.5, 'description': 'Smallest detectable effect'}
                                    }
                                    
                                    multi_scenarios = {}
                                    for i, scenario_name in enumerate(scenarios):
                                        template = scenario_templates.get(scenario_name, {
                                            'name': scenario_name.title(), 
                                            'effect_size': 0.5, 
                                            'power': 0.8, 
                                            'description': f'{scenario_name.title()} scenario'
                                        })
                                        
                                        multi_scenarios[f"scenario_{i+1}"] = {
                                            'name': template['name'],
                                            'description': template['description'],
                                            'target_p_value': 0.05,
                                            'parameters': {
                                                'total_n': int(100 / (template['effect_size'] ** 2)),  # Rough estimate
                                                'effect_size_value': template['effect_size'],
                                                'effect_size_type': 'cohens_d',
                                                'alpha': 0.05,
                                                'power': template['power']
                                            }
                                        }
                                    
                                    st.session_state.multi_scenarios = multi_scenarios
                                else:
                                    st.warning("⚠️ 5-Scenario API returned scenarios in unexpected format")
                                    st.session_state.multi_scenarios = {}
                                st.session_state.scenario_analysis_complete = True
                                st.session_state.evidence_quality = scenario_data.get('evidence_quality')
                                st.session_state.effect_size_uncertainty = scenario_data.get('effect_size_uncertainty')
                                st.session_state.recommended_scenario = scenario_data.get('recommended_scenario', 'scenario_3')
                                
                                st.success("✅ **5 Scenarios Analysis completed!** Check the dashboard below.")
                            else:
                                st.warning("⚠️ No scenarios were generated in the analysis.")
                        else:
                            st.error(f"❌ Error from scenario analysis: {scenario_data.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Failed to generate scenario analysis.")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Could not connect to the backend API. Is it running at " + BACKEND_URL + "?")
                except requests.exceptions.Timeout:
                    st.error("⏱️ The request to the backend API timed out. The scenario analysis might be taking too long.")
                except Exception as e:
                    st.error(f"❌ An unexpected error occurred: {e}")

# === NEW: Multi-Scenario Analysis Button ===
st.markdown("---")
# Generate Scenarios section is now integrated into the main Analyze button above

# === Multi-Scenario Display ===
# Debug: Check if we have malformed scenario data
if st.session_state.scenario_analysis_complete and st.session_state.multi_scenarios:
    # Check if data is malformed
    malformed_scenarios = [k for k, v in st.session_state.multi_scenarios.items() if not isinstance(v, dict)]
    if malformed_scenarios:
        st.error(f"🐛 **Debug**: Found malformed scenario data: {malformed_scenarios}")
        st.write("Multi-scenarios data structure:")
        st.write(st.session_state.multi_scenarios)
        st.info("💡 Try clicking '🔍 Generate 5 Scenarios Analysis' button to regenerate proper scenario data.")

# Only show if scenarios are properly structured
if (st.session_state.scenario_analysis_complete and 
    st.session_state.multi_scenarios and 
    all(isinstance(v, dict) for v in st.session_state.multi_scenarios.values())):
    st.header("🎨 Multi-Scenario Statistical Design Dashboard")
    
    # Scenario selection dropdown
    scenario_options = list(st.session_state.multi_scenarios.keys())
    scenario_display_names = [name.replace('_', ' ').title() for name in scenario_options]
    
    # Find current selection index
    current_index = 0
    if st.session_state.selected_scenario in scenario_options:
        current_index = scenario_options.index(st.session_state.selected_scenario)
    
    selected_index = st.selectbox(
        "🎯 Choose Scenario to Analyze:",
        range(len(scenario_display_names)),
        index=current_index,
        format_func=lambda i: f"{scenario_display_names[i]} {'⭐ (Recommended)' if scenario_options[i] == st.session_state.recommended_scenario else ''}",
        help="Select different scenarios to compare statistical approaches",
        key="scenario_selector"
    )
    
    st.session_state.selected_scenario = scenario_options[selected_index]
    selected_scenario_data = st.session_state.multi_scenarios[st.session_state.selected_scenario]
    
    # Debug: Check data structure
    st.write(f"Debug: selected_scenario_data type: {type(selected_scenario_data)}")
    if isinstance(selected_scenario_data, dict):
        st.write(f"Debug: scenario keys: {list(selected_scenario_data.keys())}")
    else:
        st.error(f"Error: Expected dictionary but got {type(selected_scenario_data)}: {selected_scenario_data}")
        st.stop()
    
    # Display selected scenario details
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        params = selected_scenario_data.get('parameters', {})
        st.metric(
            "Sample Size", 
            params.get('total_n', 'N/A'),
            help="Total number of participants needed"
        )
        st.metric(
            "Target Power",
            f"{(params.get('power', 0.8) * 100):.0f}%",
            help="Probability of detecting true effect"
        )
    
    with col2:
        st.metric(
            "Significance Level (α)", 
            selected_scenario_data.get('target_p_value', params.get('alpha', 0.05)),
            help="Probability of Type I error"
        )
        st.metric(
            "Effect Size",
            f"{params.get('effect_size_value', 0.5):.3f}",
            help="Expected effect size (Cohen's d or equivalent)"
        )
    
    with col3:
        st.info(f"**{selected_scenario_data.get('name', 'Scenario')}**\n\n{selected_scenario_data.get('description', 'No description available')}")
    
    # Interactive Visualizations
    st.subheader("📊 Interactive Scenario Comparisons")
    
    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["📈 Scenario Comparison", "🔥 Power Curves"])
    
    with tab1:
        comparison_chart = create_scenario_comparison_chart(
            st.session_state.multi_scenarios, 
            st.session_state.recommended_scenario
        )
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)
        else:
            st.warning("Unable to create comparison chart")
    
    with tab2:
        try:
            # Debug: Check multi_scenarios structure
            st.write(f"Debug: multi_scenarios type: {type(st.session_state.multi_scenarios)}")
            if st.session_state.multi_scenarios:
                for key, value in st.session_state.multi_scenarios.items():
                    st.write(f"Debug: {key} -> type: {type(value)}")
                    if isinstance(value, dict):
                        st.write(f"Debug: {key} keys: {list(value.keys())}")
                    else:
                        st.write(f"Debug: {key} value: {str(value)[:100]}...")
            
            power_curves_chart = create_power_curves_comparison(
                st.session_state.multi_scenarios,
                st.session_state.recommended_scenario  
            )
            if power_curves_chart:
                st.plotly_chart(power_curves_chart, use_container_width=True)
                st.markdown("**💡 How to read this chart:** Each line shows how statistical power increases with sample size for different scenarios. The **bold line** is the recommended scenario. Higher/thicker lines need fewer participants to achieve the same power.")
            else:
                st.warning("Unable to create power curves")
        except Exception as e:
            st.error(f"Error creating power curves chart: {e}")
            st.write("Multi-scenario data structure:")
            st.write(st.session_state.multi_scenarios)
    
    # Update current parameters based on selected scenario
    if selected_scenario_data and 'parameters' in selected_scenario_data:
        scenario_params = selected_scenario_data['parameters']
        st.session_state.current_N = scenario_params.get('total_n', st.session_state.current_N)
        st.session_state.current_d = scenario_params.get('effect_size_value', st.session_state.current_d)
    

# === Enhanced Study Analysis Display ===
if st.session_state.study_analysis:
    st.header("2. 🤖 AI Study Analysis Results")
    
    analysis = st.session_state.study_analysis
    
    # Main suggestion display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎯 Recommended Statistical Test")
        suggested_test = analysis.get('suggested_study_type', 'Unknown')
        
        # Special handling for unsupported survival analysis
        if suggested_test == "survival_analysis":
            st.error(f"**{get_test_display_name(suggested_test)}**")
            st.warning("⚠️ **Survival Analysis is not yet supported.** This requires specialized methods like Cox Proportional-Hazards models and Log-rank tests. Consider consulting with a biostatistician for time-to-event analyses.")
        else:
            st.success(f"**{get_test_display_name(suggested_test)}**")
        
        # Rationale
        rationale = analysis.get('rationale', 'No rationale provided')
        st.write(f"**Rationale:** {rationale}")
    
    with col2:
        confidence = analysis.get('confidence_level', 0.0)
        if confidence > 0:
            st.metric(
                label="AI Confidence",
                value=f"{confidence*100:.0f}%"
            )
            st.progress(confidence)
    
    with col3:
        data_type = analysis.get('data_type', 'Unknown')
        study_design = analysis.get('study_design', 'Unknown')
        st.write(f"**Data Type:** {data_type.title()}")
        st.write(f"**Study Design:** {study_design.replace('_', ' ').title()}")
    
    # Alternative tests
    alternatives = analysis.get('alternative_tests', [])
    if alternatives:
        with st.expander("🔄 Alternative Test Options", expanded=False):
            st.write("**Other suitable statistical tests for your study:**")
            for alt_test in alternatives:
                st.write(f"• {get_test_display_name(alt_test)}")
    
    # Analysis parameters
    parameters = analysis.get('parameters', {})
    if parameters:
        with st.expander("📊 Study Parameters from AI", expanded=False):
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                if 'total_n' in parameters:
                    st.metric("Recommended Sample Size", parameters['total_n'])
                if 'alpha' in parameters:
                    st.metric("Significance Level (α)", parameters['alpha'])
            
            with param_col2:
                if 'power' in parameters:
                    st.metric("Target Power", f"{parameters['power']*100:.0f}%")
                
                effect_size_val = parameters.get('effect_size_value') or parameters.get('cohens_d')
                if effect_size_val:
                    effect_type = parameters.get('effect_size_type', 'effect_size')
                    st.metric(f"Effect Size ({effect_type})", f"{effect_size_val:.3f}")
    
    # === Research References Section ===
    # Only show if literature search was enabled and references exist
    if st.session_state.get('lit_search_was_enabled', False):
        references = st.session_state.get('references', [])
        
        if references:
            with st.expander("📚 Research References", expanded=False):
                references_source = getattr(st.session_state, 'references_source', 'basic_search')
                references_warning = getattr(st.session_state, 'references_warning', None)
                
                # Debug info
                st.write(f"Debug: Found {len(references)} references, source: {references_source}")
                
                if references_warning:
                    st.error(f"🚨 **{references_warning}**")
                
                if 'pubmed(' in references_source and 'arxiv(' in references_source:
                    st.success(f"✅ **Real Research Citations** - Sources searched: {references_source}")
                elif references_source == 'pubmed_arxiv_clinicaltrials':
                    st.success("✅ **Real Research Citations** (from PubMed, arXiv & ClinicalTrials.gov)")
                elif references_source == 'pubmed_arxiv':
                    st.success("✅ **Real Research Citations** (from PubMed & arXiv)")
                else:
                    st.warning("⚠️ **AI-Generated References** (research search failed)")
                
                # View toggle buttons
                col1, col2 = st.columns(2)
                with col1:
                    view_list = st.button("📄 List View", key="list_view_btn")
                with col2:
                    view_table = st.button("📊 Table View", key="table_view_btn") 
                
                # Default to list view, switch to table if requested
                if view_table:
                    st.session_state.reference_view = 'table'
                elif view_list:
                    st.session_state.reference_view = 'list'
                
                current_view = getattr(st.session_state, 'reference_view', 'list')
                
                if current_view == 'table':
                    # Create table view with research data
                    if hasattr(st.session_state, 'multi_scenario_data') and 'research_papers_data' in st.session_state.multi_scenario_data:
                        papers_data = st.session_state.multi_scenario_data.get('research_papers_data', [])
                        
                        if papers_data:
                            # Create DataFrame for table view
                            table_data = []
                            for paper in papers_data:
                                # Calculate quality score based on available criteria
                                quality_score = 0
                                quality_factors = []
                                
                                # Check for study type indicators
                                title_lower = (paper.get('title', '') or '').lower()
                                journal = (paper.get('journal', '') or '').lower()
                                
                                if 'randomized' in title_lower or 'rct' in title_lower:
                                    quality_score += 3
                                    quality_factors.append("RCT")
                                if 'meta-analysis' in title_lower:
                                    quality_score += 4
                                    quality_factors.append("Meta-analysis")
                                if 'systematic review' in title_lower:
                                    quality_score += 3
                                    quality_factors.append("Systematic Review")
                                if 'clinical trial' in title_lower:
                                    quality_score += 2
                                    quality_factors.append("Clinical Trial")
                                if paper.get('sample_size') and paper.get('sample_size') > 100:
                                    quality_score += 1
                                    quality_factors.append("Large N")
                                
                                # Determine study type
                                if 'clinicaltrials.gov' in journal:
                                    study_type = "Clinical Trial"
                                elif 'arxiv' in journal:
                                    study_type = "Preprint"
                                else:
                                    study_type = "Published Research"
                                
                                # Format authors
                                authors = paper.get('authors', []) or []
                                if isinstance(authors, list) and authors:
                                    author_str = ', '.join(authors[:2])
                                    if len(authors) > 2:
                                        author_str += ' et al.'
                                else:
                                    author_str = 'N/A'
                                
                                row = {
                                    'Title': paper.get('title', 'N/A')[:60] + '...' if len(paper.get('title', '')) > 60 else paper.get('title', 'N/A'),
                                    'Year': paper.get('year', 'N/A'),
                                    'Authors': author_str,
                                    'Journal': paper.get('journal', 'N/A')[:25] + '...' if len(paper.get('journal', '')) > 25 else paper.get('journal', 'N/A'),
                                    'Sample Size': f"N={paper.get('sample_size')}" if paper.get('sample_size') else 'N/A',
                                    'Study Signal': paper.get('study_signal', 'Unknown'),
                                    'Type': study_type,
                                    'Quality Score': f"{quality_score}/10",
                                    'Quality Factors': ', '.join(quality_factors) if quality_factors else 'Standard',
                                    'Link': paper.get('url', 'N/A')
                                }
                                table_data.append(row)
                            
                            # Create and display DataFrame
                            df = pd.DataFrame(table_data)
                            
                            # Configure column display
                            column_config = {
                                "Link": st.column_config.LinkColumn(
                                    "Link",
                                    help="Click to view the original source",
                                    validate="^https?://.*",
                                    max_chars=100,
                                    display_text="View"
                                ),
                                "Study Signal": st.column_config.SelectboxColumn(
                                    "Study Signal",
                                    help="Whether the study found positive, negative, or mixed results. Important for identifying underpowered studies with null results.",
                                    width="small",
                                    options=["Positive", "Negative", "Mixed", "Unclear", "Unknown"]
                                ),
                                "Quality Score": st.column_config.ProgressColumn(
                                    "Quality Score",
                                    help="Quality assessment based on study design, sample size, and methodology",
                                    min_value=0,
                                    max_value=10,
                                    format="%d/10"
                                ),
                                "Quality Factors": st.column_config.TextColumn(
                                    "Quality Factors",
                                    help="Key quality indicators found in this study",
                                    width="medium"
                                )
                            }
                            
                            st.dataframe(
                                df, 
                                use_container_width=True, 
                                hide_index=True,
                                column_config=column_config
                            )
                            
                            st.caption(f"📊 **Quality Score Legend**: RCT (+3), Meta-analysis (+4), Systematic Review (+3), Clinical Trial (+2), Large Sample (+1)")
                            st.caption(f"🎯 **Study Signal**: Shows whether studies found positive, negative, or mixed results - helps identify underpowered null studies")
                        else:
                            st.info("No structured research data available for table view")
                    else:
                        st.info("Research data not loaded yet. Run analysis to populate table view.")
                else:
                    # List view (default)
                    for i, ref in enumerate(references, 1):
                        st.markdown(f"**{i}.** {ref}")
    
    st.markdown("---")

# === Test Type Selection & Override ===
# Always show this section but in an accordion if analysis not run yet
if st.session_state.study_analysis:
    st.header("3. 🛠️ Select & Configure Statistical Test")
    show_tests_directly = True
else:
    # Show in collapsed accordion when analysis not run yet
    with st.expander("⚙️ Advanced: Manual Statistical Test Configuration (analysis not run yet)", expanded=False):
        st.info("💡 **Tip**: Run 'Analyze Study' above first for AI-recommended test selection, or manually configure a test here if you prefer.")
        show_tests_directly = True

# Load available tests without showing error messages when analysis not run
if show_tests_directly:
    # Fetch tests quietly
    if not st.session_state.available_tests:
        try:
            enhanced_test_info, available_tests = fetch_available_tests()
            if available_tests:
                st.session_state.available_tests = available_tests
            else:
                # Fallback to default without showing error
                st.session_state.available_tests = [{
                    'test_id': 'two_sample_t_test',
                    'name': 'Two-Sample t-Test',
                    'description': 'Default test for comparing means between two groups'
                }]
        except:
            # Fallback silently
            st.session_state.available_tests = [{
                'test_id': 'two_sample_t_test', 
                'name': 'Two-Sample t-Test',
                'description': 'Default test for comparing means between two groups'
            }]

    if st.session_state.available_tests:
        test_options = [(test['test_id'], test['name']) for test in st.session_state.available_tests]
        test_ids = [test_id for test_id, _ in test_options]
        test_names = [name for _, name in test_options]
    
    # Test selection with AI suggestion highlighted
    current_index = 0
    if st.session_state.suggested_test_type in test_ids:
        current_index = test_ids.index(st.session_state.suggested_test_type)
    
    selected_index = st.selectbox(
        "Choose statistical test:",
        range(len(test_names)),
        index=current_index,
        format_func=lambda i: f"{test_names[i]} {'🤖 (AI Recommended)' if test_ids[i] == st.session_state.suggested_test_type else ''}",
        help="Select the statistical test you want to use. The AI recommendation is marked with 🤖"
    )
    
    st.session_state.selected_test_type = test_ids[selected_index]
    
    if st.session_state.selected_test_type != st.session_state.suggested_test_type:
        st.info(f"🔄 You've overridden the AI suggestion. Using: **{test_names[selected_index]}**")

if st.session_state.processed_idea_text:
    with st.expander("📜 View Processed Idea Sent to AI", expanded=False):
        st.caption("This is the text that was sent to the AI model for analysis:")
        display_text = st.session_state.processed_idea_text
        if len(display_text) > 1500:
            display_text = display_text[:1500] + "..."
        st.markdown(f"```\n{display_text}\n```")

# Display LLM provider if available
if st.session_state.llm_provider_used and (st.session_state.estimation_justification or st.session_state.study_analysis):
    st.caption(f"🤖 Analysis powered by: **{st.session_state.llm_provider_used}**")

if st.session_state.estimation_justification:
    with st.expander("📝 AI's Analysis & Justification", expanded=True):
        st.info(st.session_state.estimation_justification)

# Legacy references section removed - now using Research References from multi-scenario analysis


# === Dynamic Parameter Forms ===
st.subheader(f"🎨 Configure {get_test_display_name(st.session_state.selected_test_type)} Parameters")

# Get test-specific parameters from AI if available
ai_params = st.session_state.test_parameters
current_test = st.session_state.selected_test_type

if current_test == "two_sample_t_test":
    st.markdown("📊 **Two-Sample t-Test Configuration**")
    st.markdown("*Compare means between two independent groups (e.g., treatment vs control)*")
    
    col1, col2 = st.columns(2)
    with col1:
        N_total_input = st.number_input(
            "Total Number of Participants (N)",
            min_value=3,
            value=ai_params.get('total_n', st.session_state.current_N),
            step=1,
            key="N_total_ttest",
            help="Total participants, split equally into two groups"
        )
    
    with col2:
        cohens_d_input = st.number_input(
            "Expected Cohen's d (Effect Size)",
            min_value=-5.0,
            max_value=5.0,
            value=ai_params.get('cohens_d', ai_params.get('effect_size_value', st.session_state.current_d)),
            step=0.01,
            format="%.3f",
            key="cohens_d_ttest",
            help="Standardized mean difference. Common values: 0.2 (small), 0.5 (medium), 0.8 (large)"
        )
    
    st.session_state.current_N = N_total_input
    st.session_state.current_d = cohens_d_input
    
    # Effect size interpretation
    effect_interp = get_effect_size_interpretation("two_sample_t_test", cohens_d_input)
    st.write(f"**Effect Size Interpretation:** {effect_interp}")

elif current_test == "chi_square":
    st.markdown("📊 **Chi-Square Test Configuration**")
    st.markdown("*Test associations between categorical variables*")
    
    # Chi-square specific parameters
    st.subheader("Contingency Table Data")
    
    # Simple 2x2 table input for common case
    table_type = st.radio(
        "Table Type:",
        ["2x2 Table (Common)", "Custom Table"],
        help="Choose table size for your categorical analysis"
    )
    
    if table_type == "2x2 Table (Common)":
        st.markdown("**Enter observed frequencies for a 2x2 contingency table:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Group 1**")
            a = st.number_input("Category A", min_value=0, value=25, key="chi_a")
            b = st.number_input("Category B", min_value=0, value=15, key="chi_b")
        
        with col2:
            st.write("**Group 2**")
            c = st.number_input("Category A", min_value=0, value=20, key="chi_c")
            d = st.number_input("Category B", min_value=0, value=30, key="chi_d")
        
        contingency_table = [[a, b], [c, d]]
        
        # Display table
        df_display = pd.DataFrame(contingency_table, 
                                columns=["Category A", "Category B"],
                                index=["Group 1", "Group 2"])
        st.write("**Your Contingency Table:**")
        st.dataframe(df_display)
        
        total_n = a + b + c + d
        st.write(f"**Total Sample Size:** {total_n}")
    
    else:  # Custom table
        st.markdown("**Enter your custom contingency table (comma-separated values):**")
        table_input = st.text_area(
            "Contingency Table",
            value="25,15\n20,30",
            help="Enter rows separated by newlines, values separated by commas"
        )
        
        try:
            rows = table_input.strip().split('\n')
            contingency_table = [[int(val.strip()) for val in row.split(',')] for row in rows]
            
            # Display table
            df_display = pd.DataFrame(contingency_table)
            st.write("**Your Contingency Table:**")
            st.dataframe(df_display)
            
            total_n = sum(sum(row) for row in contingency_table)
            st.write(f"**Total Sample Size:** {total_n}")
        except:
            st.error("Invalid table format. Use comma-separated values with newlines for rows.")
            contingency_table = [[25, 15], [20, 30]]  # Default
    
    # Store chi-square parameters
    st.session_state.current_contingency_table = contingency_table

elif current_test == "one_way_anova":
    st.markdown("📊 **One-Way ANOVA Configuration**")
    st.markdown("*Compare means across multiple groups (3 or more)*")
    
    # Number of groups
    num_groups = st.slider(
        "Number of Groups",
        min_value=3,
        max_value=6,
        value=3,
        help="Select how many groups you want to compare"
    )
    
    groups_data = []
    
    st.subheader("Group Parameters")
    cols = st.columns(min(num_groups, 3))
    
    for i in range(num_groups):
        col_idx = i % 3
        with cols[col_idx]:
            st.write(f"**Group {i+1}**")
            mean = st.number_input(f"Mean", key=f"anova_mean_{i}", value=10.0 + i*2)
            std = st.number_input(f"Std Dev", key=f"anova_std_{i}", value=2.5, min_value=0.1)
            n = st.number_input(f"Sample Size", key=f"anova_n_{i}", value=20, min_value=2)
            
            # Generate sample data for the group
            np.random.seed(42 + i)  # Reproducible
            group_data = np.random.normal(mean, std, n)
            groups_data.append(group_data.tolist())
    
    total_anova_n = sum(len(group) for group in groups_data)
    st.write(f"**Total Sample Size:** {total_anova_n}")
    
    # Store ANOVA parameters
    st.session_state.current_anova_groups = groups_data

elif current_test == "correlation":
    st.markdown("📊 **Correlation Analysis Configuration**")
    st.markdown("*Analyze relationships between two continuous variables*")
    
    # Correlation type
    corr_type = st.selectbox(
        "Correlation Type",
        ["pearson", "spearman"],
        help="Pearson for linear relationships, Spearman for monotonic relationships"
    )
    
    # Sample data generation
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input(
            "Number of Data Points",
            min_value=5,
            max_value=1000,
            value=50,
            help="Number of paired observations"
        )
        
        true_correlation = st.slider(
            "True Correlation (for simulation)",
            min_value=-1.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Expected correlation coefficient"
        )
    
    with col2:
        mean_x = st.number_input("X Variable Mean", value=50.0)
        mean_y = st.number_input("Y Variable Mean", value=100.0)
        std_x = st.number_input("X Variable Std Dev", value=10.0, min_value=0.1)
        std_y = st.number_input("Y Variable Std Dev", value=20.0, min_value=0.1)
    
    # Generate correlated data
    np.random.seed(42)  # Reproducible
    x = np.random.normal(mean_x, std_x, n_samples)
    noise = np.random.normal(0, std_y * np.sqrt(1 - true_correlation**2), n_samples)
    y = mean_y + (std_y/std_x) * true_correlation * (x - mean_x) + noise
    
    # Store correlation parameters
    st.session_state.current_x_values = x.tolist()
    st.session_state.current_y_values = y.tolist()
    st.session_state.current_correlation_type = corr_type
    
    # Show sample of data
    if st.checkbox("Show Data Preview"):
        df_preview = pd.DataFrame({
            'X Variable': x[:10],
            'Y Variable': y[:10]
        })
        st.dataframe(df_preview)
    
    # Effect size interpretation
    effect_interp = get_effect_size_interpretation("correlation", true_correlation)
    st.write(f"**Correlation Strength:** {effect_interp}")

elif current_test == "ancova":
    st.markdown("📊 **ANCOVA (Analysis of Covariance) Configuration**")
    st.markdown("*Compare groups while adjusting for baseline covariates - very common in clinical trials*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        N_total_input = st.number_input(
            "Total Number of Participants (N)",
            min_value=5,
            value=ai_params.get('total_n', st.session_state.current_N),
            step=1,
            key="N_total_ancova",
            help="Total participants across all groups"
        )
    
    with col2:
        cohens_d_input = st.number_input(
            "Expected Cohen's d (Effect Size)",
            min_value=-5.0,
            max_value=5.0,
            value=ai_params.get('cohens_d', ai_params.get('effect_size_value', st.session_state.current_d)),
            step=0.01,
            format="%.3f",
            key="cohens_d_ancova",
            help="Effect size after adjusting for covariate"
        )
    
    with col3:
        covariate_corr = st.slider(
            "Covariate-Outcome Correlation",
            min_value=0.0,
            max_value=0.9,
            value=0.5,
            step=0.1,
            key="covariate_corr_ancova",
            help="How strongly the covariate predicts the outcome (higher = more power gain)"
        )
    
    st.session_state.current_N = N_total_input
    st.session_state.current_d = cohens_d_input
    st.session_state.covariate_correlation = covariate_corr
    
    st.info(f"💡 **ANCOVA Benefit:** With r={covariate_corr:.1f} covariate correlation, you gain ~{(1/(1-covariate_corr**2)-1)*100:.0f}% more power vs. standard t-test!")

elif current_test == "fishers_exact":
    st.markdown("📊 **Fisher's Exact Test Configuration**")
    st.markdown("*Exact test for 2×2 categorical data - ideal for small samples*")
    
    st.subheader("2×2 Contingency Table")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Group 1 (e.g., Treatment)**")
        group1_success = st.number_input("Success Count", min_value=0, value=15, step=1, key="g1_success")
        group1_failure = st.number_input("Failure Count", min_value=0, value=5, step=1, key="g1_failure")
        group1_total = group1_success + group1_failure
        st.write(f"**Group 1 Total: {group1_total}**")
        
    with col2:
        st.write("**Group 2 (e.g., Control)**")
        group2_success = st.number_input("Success Count", min_value=0, value=8, step=1, key="g2_success")
        group2_failure = st.number_input("Failure Count", min_value=0, value=12, step=1, key="g2_failure")
        group2_total = group2_success + group2_failure
        st.write(f"**Group 2 Total: {group2_total}**")
    
    # Create contingency table
    contingency_table = [[group1_success, group1_failure], [group2_success, group2_failure]]
    total_n = group1_total + group2_total
    
    # Display table
    st.subheader("Contingency Table")
    table_df = pd.DataFrame(
        contingency_table,
        index=["Group 1", "Group 2"],
        columns=["Success", "Failure"]
    )
    st.dataframe(table_df)
    
    st.session_state.contingency_table = contingency_table
    st.session_state.current_N = total_n
    
    if total_n < 30:
        st.success("✅ **Good choice!** Fisher's exact test is ideal for small samples (N < 30)")
    else:
        st.info("💡 For larger samples (N ≥ 30), Chi-square test might be more appropriate")

elif current_test == "logistic_regression":
    st.markdown("📊 **Logistic Regression Configuration**")
    st.markdown("*Model binary outcomes (response/non-response) with odds ratios*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        N_total_input = st.number_input(
            "Total Number of Participants (N)",
            min_value=10,
            value=ai_params.get('total_n', 100),
            step=1,
            key="N_total_logistic",
            help="Total participants across all groups"
        )
    
    with col2:
        baseline_rate = st.slider(
            "Control Group Response Rate",
            min_value=0.01,
            max_value=0.99,
            value=0.3,
            step=0.01,
            key="baseline_rate_logistic",
            help="Expected proportion of responses in control group"
        )
    
    with col3:
        odds_ratio = st.number_input(
            "Expected Odds Ratio",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="odds_ratio_logistic",
            help="Treatment effect as odds ratio (1.0 = no effect)"
        )
    
    # Calculate treatment group rate
    baseline_odds = baseline_rate / (1 - baseline_rate)
    treatment_odds = baseline_odds * odds_ratio
    treatment_rate = treatment_odds / (1 + treatment_odds)
    
    st.session_state.current_N = N_total_input
    st.session_state.baseline_rate = baseline_rate
    st.session_state.odds_ratio = odds_ratio
    
    # Show interpretation
    st.subheader("Expected Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Control Response Rate", f"{baseline_rate:.1%}")
    with col2:
        st.metric("Treatment Response Rate", f"{treatment_rate:.1%}")
    with col3:
        st.metric("Absolute Difference", f"+{(treatment_rate-baseline_rate):.1%}")

elif current_test == "repeated_measures_anova":
    st.markdown("📊 **Repeated Measures ANOVA Configuration**")
    st.markdown("*Analyze longitudinal data with multiple time points per subject*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        N_subjects = st.number_input(
            "Number of Subjects",
            min_value=5,
            value=ai_params.get('total_n', 30) // 3,  # Estimate subjects from total N
            step=1,
            key="N_subjects_rm",
            help="Number of individual participants"
        )
        
        n_timepoints = st.number_input(
            "Number of Time Points",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            key="n_timepoints_rm",
            help="Number of repeated measurements per subject"
        )
    
    with col2:
        cohens_f = st.number_input(
            "Expected Cohen's f (Effect Size)",
            min_value=0.0,
            max_value=2.0,
            value=0.25,
            step=0.05,
            format="%.3f",
            key="cohens_f_rm",
            help="Effect size for time/group differences (0.1=small, 0.25=medium, 0.4=large)"
        )
        
        correlation_between = st.slider(
            "Correlation Between Measures",
            min_value=0.0,
            max_value=0.9,
            value=0.6,
            step=0.1,
            key="correlation_rm",
            help="How correlated are repeated measures (higher = more power)"
        )
    
    total_observations = N_subjects * n_timepoints
    
    st.session_state.N_subjects = N_subjects
    st.session_state.n_timepoints = n_timepoints
    st.session_state.cohens_f = cohens_f
    st.session_state.correlation_between_measures = correlation_between
    st.session_state.current_N = total_observations  # For compatibility
    
    # Show study design summary
    st.subheader("Study Design Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Observations", total_observations)
    with col2:
        st.metric("Subjects", N_subjects)
    with col3:
        st.metric("Measurements per Subject", n_timepoints)
    
    # Effect size interpretation
    if cohens_f < 0.15:
        effect_size_interp = "Small Effect"
    elif cohens_f < 0.35:
        effect_size_interp = "Medium Effect"  
    else:
        effect_size_interp = "Large Effect"
    
    st.write(f"**Effect Size Interpretation:** {effect_size_interp}")

elif current_test == "survival_analysis":
    st.error("🚫 **Survival Analysis is not yet supported**")
    st.markdown("""
    This study requires time-to-event analysis methods like:
    - **Cox Proportional-Hazards Models**
    - **Log-rank Tests**  
    - **Kaplan-Meier Curves**
    
    These advanced methods will be added in a future update. For now, consider:
    - Consulting with a biostatistician
    - Using specialized survival analysis software (R, SAS, SPSS)
    - Converting to binary outcomes if appropriate (event/no-event at fixed timepoint)
    """)

else:
    st.warning(f"Parameter configuration for {current_test} is not yet implemented.")
    # Fallback to basic t-test parameters
    col1, col2 = st.columns(2)
    with col1:
        N_total_input = st.number_input(
            "Total Number of Participants (N)",
            min_value=3,
            value=st.session_state.current_N,
            step=1
        )
    with col2:
        cohens_d_input = st.number_input(
            "Expected Effect Size",
            min_value=-5.0,
            max_value=5.0,
            value=st.session_state.current_d,
            step=0.01,
            format="%.3f"
        )
    
    st.session_state.current_N = N_total_input
    st.session_state.current_d = cohens_d_input


# Only show calculations if analysis has been run
if st.session_state.study_analysis:
    p_val, msg = calculate_p_value_from_N_d(st.session_state.current_N, st.session_state.current_d)
    st.session_state.current_p_value = p_val
    st.session_state.p_value_message = msg

    st.subheader("Calculated P-Value")
    if st.session_state.p_value_message:
        st.warning(st.session_state.p_value_message)

    if st.session_state.current_p_value is not None:
        st.metric(label="Calculated P-Value", value=f"{st.session_state.current_p_value:.4f}")
        if st.session_state.current_p_value < 0.05:
            st.success("This p-value is typically considered statistically significant (p < 0.05).")
        else:
            st.info("This p-value is not typically considered statistically significant (p >= 0.05).")
    else:
        st.info("P-value will be calculated once valid parameters are set.")

    # --- New Power / Probability Visualization ---
    power_val, power_msg = calculate_power_from_N_d(st.session_state.current_N, st.session_state.current_d)

    st.subheader("Probability of Detecting the Effect")
    if power_msg:
        st.warning(power_msg)
    elif power_val is not None:
        st.metric(label="Estimated Power", value=f"{power_val*100:.1f}%")
        st.progress(min(max(power_val, 0.0), 1.0))
    else:
        st.info("Power will be calculated once valid parameters are set.")

    st.markdown("---")
    st.caption("Remember: This tool is for educational and exploratory purposes. Always consult with a qualified statistician for actual clinical trial design and sample size calculations.")

