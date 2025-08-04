# app.py - Enhanced Universal Study P-Value Explorer
import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistical_utils import calculate_p_value_from_N_d, calculate_power_from_N_d

# Configuration for the FastAPI backend URLs
# Assumes FastAPI is running on localhost:8000
BASE_URL = "http://localhost:8000"
BACKEND_URL = f"{BASE_URL}/process_idea"
AVAILABLE_TESTS_URL = f"{BASE_URL}/available_tests"
CALCULATE_STATISTICS_URL = f"{BASE_URL}/calculate_statistics"  # New endpoint for direct calculations


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
        "correlation": "Correlation Analysis"
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

with st.sidebar:
    st.header("About")
    st.info(
        "**Universal Study P-Value Explorer**\n\n"
        "This enhanced tool supports multiple statistical analyses:\n"
        "• **T-tests**: Compare two groups\n"
        "• **Chi-square**: Test categorical associations\n"
        "• **ANOVA**: Compare multiple groups\n"
        "• **Correlation**: Analyze relationships\n\n"
        "Get AI-powered study type suggestions and statistical calculations."
        "\n\n**Disclaimer:** AI estimates are for exploratory purposes only. "
        "Consult a statistician for actual study design."
    )
    st.header("Statistical Test Information")
    
    # Fetch and display available tests
    if st.button("🔄 Refresh Available Tests", help="Fetch latest test information from API"):
        enhanced_tests, basic_tests = fetch_available_tests()
        st.session_state.available_tests = enhanced_tests
    
    if st.session_state.available_tests:
        st.subheader("Available Statistical Tests")
        for test in st.session_state.available_tests:
            with st.expander(f"📊 {test.get('name', test['test_id'])}"):
                st.write(f"**Description:** {test.get('description', 'No description')}")
                st.write(f"**Data Type:** {test.get('data_type', 'Unknown')}")
                st.write(f"**Effect Size:** {test.get('effect_size_type', 'Unknown')}")
                if test.get('required_parameters'):
                    st.write(f"**Required Parameters:** {', '.join(test['required_parameters'])}")
    else:
        st.info("Click 'Refresh Available Tests' to see supported statistical tests.")
    
    st.header("Calculation Notes")
    st.markdown(
        "**Test-Specific Calculations:**\n\n"
        "• **T-test**: Uses Cohen's d and sample size\n"
        "• **Chi-square**: Uses contingency tables and Cramér's V\n"
        "• **ANOVA**: Uses group means and eta-squared\n"
        "• **Correlation**: Uses Pearson/Spearman coefficients\n\n"
        "All calculations include appropriate degrees of freedom and statistical assumptions."
    )


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

# === Enhanced Study Analysis Display ===
if st.session_state.study_analysis:
    st.header("2. 🤖 AI Study Analysis Results")
    
    analysis = st.session_state.study_analysis
    
    # Main suggestion display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎯 Recommended Statistical Test")
        suggested_test = analysis.get('suggested_study_type', 'Unknown')
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

if st.session_state.references:
    with st.expander("📚 References & Resources", expanded=False):
        for ref in st.session_state.references:
            st.markdown(f"• {ref}")


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

