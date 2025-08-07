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


# Helper functions for enhanced features
def fetch_available_tests():
    """Fetch available statistical tests from API"""
    try:
        response = requests.get(AVAILABLE_TESTS_URL, timeout=10)
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
    payload = {
        "study_description": study_description.strip(),
        "llm_provider": llm_provider,
        "max_papers": max_papers
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
        max_papers = st.slider(
            "Max Papers to Fetch",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of research papers to fetch from PubMed, arXiv & ClinicalTrials.gov"
        )
        st.session_state.max_papers = max_papers
        
        st.caption("Sources: PubMed, arXiv, ClinicalTrials.gov")
    
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

# Optional LLM provider selection
with st.expander("⚙️ Advanced Settings", expanded=False):
    llm_provider = st.selectbox(
        "Select AI Provider (optional):",
        ["Default", "GEMINI", "OPENAI", "AZURE_OPENAI", "ANTHROPIC"],
        help="Choose specific AI provider or use default system setting"
    )
    if llm_provider == "Default":
        llm_provider = None

if st.button("🔍 Analyze Study & Get AI Recommendations", key="analyze_study_button_v4", type="primary"):
    if not text_idea.strip():
        st.error("Please provide a research idea or study description.")
    else:
        # Prepare enhanced payload for new API endpoint
        payload = {
            "study_description": text_idea.strip(),
            "llm_provider": llm_provider
        }
        st.session_state.llm_provider_used = "" # Reset before new call
        try:
            with st.spinner("🤖 AI is analyzing your study and suggesting optimal statistical approaches..."):
                response = requests.post(BACKEND_URL, json=payload, timeout=120) # Increased timeout for enhanced analysis
            
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

# === NEW: Multi-Scenario Analysis Button ===
st.markdown("---")
st.markdown('### 🔥 5 Stat Designs <span title="Generate 5 statistical design scenarios based on evidence uncertainty and power requirements">ℹ️</span>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔥 Generate Scenarios", key="multi_scenario_button", type="secondary"):
        if not text_idea.strip():
            st.error("Please provide a research idea or study description.")
        else:
            # Call multi-scenario analysis
            scenario_data = call_multi_scenario_analysis(text_idea.strip(), llm_provider)
            
            if scenario_data:
                # Store scenario data in session state
                st.session_state.multi_scenarios = scenario_data.get('scenarios', {})
                st.session_state.evidence_quality = scenario_data.get('evidence_quality')
                st.session_state.effect_size_uncertainty = scenario_data.get('effect_size_uncertainty')
                st.session_state.recommended_scenario = scenario_data.get('recommended_scenario')
                st.session_state.selected_scenario = scenario_data.get('recommended_scenario')  # Default to recommended
                st.session_state.scenario_analysis_complete = True
                st.session_state.suggested_test_type = scenario_data.get('suggested_study_type', 'two_sample_t_test')
                st.session_state.selected_test_type = st.session_state.suggested_test_type
                
                # Store complete response data for references
                st.session_state.multi_scenario_data = scenario_data
                
                st.success(f"✅ Generated 5 scenarios! Recommended: **{scenario_data.get('recommended_scenario', 'Unknown').replace('_', ' ').title()}**")

with col2:
    if st.session_state.scenario_analysis_complete:
        st.info(f"📊 **{len(st.session_state.multi_scenarios)} scenarios generated**\n\n🎯 **Evidence Quality:** {st.session_state.evidence_quality or 'Unknown'}\n\n⚡ **Recommended:** {st.session_state.recommended_scenario or 'Unknown'}")

# === Multi-Scenario Display ===
if st.session_state.scenario_analysis_complete and st.session_state.multi_scenarios:
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
        power_curves_chart = create_power_curves_comparison(
            st.session_state.multi_scenarios,
            st.session_state.recommended_scenario  
        )
        if power_curves_chart:
            st.plotly_chart(power_curves_chart, use_container_width=True)
            st.markdown("**💡 How to read this chart:** Each line shows how statistical power increases with sample size for different scenarios. The **bold line** is the recommended scenario. Higher/thicker lines need fewer participants to achieve the same power.")
        else:
            st.warning("Unable to create power curves")
    
    # Update current parameters based on selected scenario
    if selected_scenario_data and 'parameters' in selected_scenario_data:
        scenario_params = selected_scenario_data['parameters']
        st.session_state.current_N = scenario_params.get('total_n', st.session_state.current_N)
        st.session_state.current_d = scenario_params.get('effect_size_value', st.session_state.current_d)
    
    # Display Research References
    st.subheader("📚 Research References")
    if hasattr(st.session_state, 'multi_scenario_data') and st.session_state.multi_scenario_data:
        scenario_data = st.session_state.multi_scenario_data
        references = scenario_data.get('references', [])
        references_source = scenario_data.get('references_source', 'unknown')
        references_warning = scenario_data.get('references_warning')
        
        if references_warning:
            st.error(f"🚨 **{references_warning}**")
        
        if references:
            if references_source == 'pubmed_arxiv_clinicaltrials':
                st.success("✅ **Real Research Citations** (from PubMed, arXiv & ClinicalTrials.gov)")
            elif references_source == 'pubmed_arxiv':
                st.success("✅ **Real Research Citations** (from PubMed & arXiv)")
            else:
                st.warning("⚠️ **AI-Generated References** (research search failed)")
                
            for i, ref in enumerate(references, 1):
                st.markdown(f"**{i}.** {ref}")
        else:
            st.info("No references available for this analysis.")
    else:
        st.info("References will appear after running multi-scenario analysis.")
    
    st.markdown("---")

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
    
    st.markdown("---")

# === Test Type Selection & Override ===
st.header("3. 🛠️ Select & Configure Statistical Test")

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
else:
    st.warning("⚠️ Unable to load available tests. Using default t-test.")
    st.session_state.selected_test_type = "two_sample_t_test"

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


p_val, msg = calculate_p_value_from_N_d(st.session_state.current_N, st.session_state.current_d)
st.session_state.current_p_value = p_val
st.session_state.p_value_message = msg

st.header("3. Calculated P-Value")
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

st.header("4. Probability of Detecting the Effect")
if power_msg:
    st.warning(power_msg)
elif power_val is not None:
    st.metric(label="Estimated Power", value=f"{power_val*100:.1f}%")
    st.progress(min(max(power_val, 0.0), 1.0))
else:
    st.info("Power will be calculated once valid parameters are set.")

st.markdown("---")
st.caption("Remember: This tool is for educational and exploratory purposes. Always consult with a qualified statistician for actual clinical trial design and sample size calculations.")

