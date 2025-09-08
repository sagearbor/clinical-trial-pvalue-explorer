"""
Enhanced Streamlit app integrating all new components.
This version includes dual-mode interface, Plotly visualizations, 
Bayesian analysis, and non-parametric tests.
"""

import os
import sys
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'frontend'))
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

# Import new components
from frontend.components.input_forms import (
    DualModeInterface,
    ParameterPresetsForm,
    ResultsDisplayForm
)
from frontend.components.visualizations import (
    InteractivePlots,
    ResultsVisualizer
)

# Backend imports
try:
    from backend.statistical.bayesian import BayesianEngine
    from backend.statistical.nonparametric import NonParametricTestFactory
    from backend.statistical.simulations import MonteCarloSimulator, SimulationParameters
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

# Configuration
BASE_URL = os.getenv("CTPE_BACKEND_URL", "http://localhost:8000")
st.set_page_config(
    layout="wide",
    page_title="Clinical Trial P-Value Explorer v3.0",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'ui_mode': 'ai_assisted',
        'study_analysis': {},
        'selected_test_type': 'two_sample_t_test',
        'test_parameters': {},
        'analysis_results': {},
        'display_options': {
            'show_visualizations': True,
            'show_confidence_intervals': True,
            'show_effect_sizes': True,
            'show_interpretations': True,
            'show_research': False,
            'show_scenarios': False
        },
        'bayesian_analysis': None,
        'simulation_results': None,
        'advanced_mode': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Main App
def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.title("🧪 Clinical Trial P-Value Explorer v3.0")
    st.markdown("""
    **Enhanced Features**: Dual-mode interface | Bayesian analysis | Non-parametric tests | 
    Interactive visualizations | Monte Carlo simulations
    """)
    
    # Initialize components
    dual_mode = DualModeInterface()
    presets = ParameterPresetsForm()
    results_display = ResultsDisplayForm()
    plots = InteractivePlots()
    results_viz = ResultsVisualizer()
    
    # Sidebar for advanced features
    with st.sidebar:
        st.markdown("### ⚙️ Advanced Settings")
        
        # Advanced mode toggle
        st.session_state.advanced_mode = st.checkbox(
            "Enable Advanced Features",
            value=st.session_state.advanced_mode,
            help="Enable Bayesian analysis, simulations, and adaptive designs"
        )
        
        if st.session_state.advanced_mode and ADVANCED_FEATURES:
            st.markdown("#### 🎯 Analysis Options")
            
            # Bayesian analysis
            use_bayesian = st.checkbox("Include Bayesian Analysis", value=False)
            if use_bayesian:
                prior_type = st.selectbox(
                    "Prior Type",
                    ["uninformative", "skeptical", "optimistic"],
                    help="Select prior distribution for Bayesian analysis"
                )
                st.session_state['bayesian_prior'] = prior_type
            
            # Monte Carlo simulations
            use_simulations = st.checkbox("Run Monte Carlo Simulations", value=False)
            if use_simulations:
                n_simulations = st.number_input(
                    "Number of Simulations",
                    min_value=100,
                    max_value=100000,
                    value=10000,
                    step=1000
                )
                st.session_state['n_simulations'] = n_simulations
            
            # Group sequential design
            use_sequential = st.checkbox("Group Sequential Design", value=False)
            if use_sequential:
                n_stages = st.slider("Number of Stages", 2, 5, 3)
                spending_func = st.selectbox(
                    "Alpha Spending Function",
                    ["O'Brien-Fleming", "Pocock", "Lan-DeMets"]
                )
                st.session_state['sequential_design'] = {
                    'n_stages': n_stages,
                    'spending': spending_func
                }
        
        # Display options
        st.markdown("### 📊 Display Options")
        for key in st.session_state.display_options:
            label = key.replace('_', ' ').title()
            st.session_state.display_options[key] = st.checkbox(
                label,
                value=st.session_state.display_options[key]
            )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## 📝 Study Configuration")
        
        # Mode selector
        mode = dual_mode.render_mode_selector()
        st.session_state.ui_mode = mode
        
        # Parameter presets
        st.markdown("### 📋 Presets")
        preset = presets.render_preset_selector()
        if preset and st.button("Apply Preset"):
            presets.apply_preset(preset)
            st.rerun()
        
        # Input forms based on mode
        if mode == 'ai_assisted':
            st.markdown("### 🤖 AI-Assisted Input")
            inputs = dual_mode.render_ai_assisted_form()
            
            if st.button("Analyze Study", type="primary", use_container_width=True):
                with st.spinner("Analyzing your study design..."):
                    # Call API for AI analysis
                    response = requests.post(
                        f"{BASE_URL}/process_idea",
                        json={
                            "study_description": inputs['study_description'],
                            "llm_provider": inputs.get('llm_provider', 'Default'),
                            "include_research": inputs.get('include_research', False)
                        }
                    )
                    
                    if response.status_code == 200:
                        st.session_state.analysis_results = response.json()
                        st.success("Analysis complete!")
                    else:
                        st.error("Analysis failed. Please try again.")
        
        else:  # Manual mode
            st.markdown("### 🔧 Manual Configuration")
            
            # Get available tests
            available_tests = fetch_available_tests()
            if available_tests:
                inputs = dual_mode.render_manual_form(available_tests)
                
                if st.button("Calculate", type="primary", use_container_width=True):
                    # Perform manual calculation
                    st.session_state.analysis_results = perform_manual_calculation(
                        inputs['test_type'],
                        inputs['parameters']
                    )
    
    with col2:
        st.markdown("## 📊 Results & Visualizations")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Results tabs
            tabs = st.tabs([
                "📈 Main Results",
                "📊 Visualizations",
                "🔬 Advanced Analysis",
                "📚 References"
            ])
            
            with tabs[0]:
                # Main results display
                results_viz.render_results_dashboard(
                    results,
                    st.session_state.display_options
                )
            
            with tabs[1]:
                # Interactive visualizations
                if st.session_state.display_options['show_visualizations']:
                    render_interactive_visualizations(results, plots)
            
            with tabs[2]:
                # Advanced analysis
                if st.session_state.advanced_mode and ADVANCED_FEATURES:
                    render_advanced_analysis(results)
            
            with tabs[3]:
                # References and literature
                if 'references' in results or 'research_papers_data' in results:
                    render_references(results)
        
        else:
            # Empty state
            st.info("""
            👈 Configure your study in the left panel to begin analysis.
            
            **Quick Start:**
            1. Choose between AI-assisted or manual mode
            2. Enter your study details or select parameters
            3. Click Analyze/Calculate to see results
            
            **New Features in v3.0:**
            - 🎯 Dual-mode interface for flexibility
            - 📊 Interactive Plotly visualizations
            - 🧮 Bayesian statistical analysis
            - 📈 Non-parametric test alternatives
            - 🎲 Monte Carlo simulations
            - 📉 Group sequential designs
            """)

def fetch_available_tests():
    """Fetch available statistical tests from API."""
    try:
        response = requests.get(f"{BASE_URL}/available_tests", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('enhanced_test_info', [])
        return []
    except:
        # Fallback to hardcoded list
        return [
            {'id': 'two_sample_t_test', 'name': 'Two-Sample t-Test', 
             'description': 'Compare means between two groups'},
            {'id': 'chi_square', 'name': 'Chi-Square Test',
             'description': 'Test association between categorical variables'},
            {'id': 'one_way_anova', 'name': 'One-Way ANOVA',
             'description': 'Compare means across multiple groups'},
            {'id': 'correlation', 'name': 'Correlation Analysis',
             'description': 'Measure relationship between variables'},
            {'id': 'mann_whitney', 'name': 'Mann-Whitney U Test',
             'description': 'Non-parametric alternative to t-test'},
            {'id': 'kruskal_wallis', 'name': 'Kruskal-Wallis Test',
             'description': 'Non-parametric alternative to ANOVA'}
        ]

def perform_manual_calculation(test_type, parameters):
    """Perform manual statistical calculation."""
    # This would call the appropriate backend function
    # For now, return mock results
    return {
        'statistical_test_used': test_type,
        'parameters': parameters,
        'calculated_p_value': np.random.uniform(0.001, 0.1),
        'calculated_power': np.random.uniform(0.7, 0.95),
        'effect_size': parameters.get('effect_size', 0.5)
    }

def render_interactive_visualizations(results, plots):
    """Render interactive Plotly visualizations."""
    st.markdown("### Interactive Visualizations")
    
    # Get parameters
    params = results.get('parameters', {})
    test_type = results.get('statistical_test_used', 'two_sample_t_test')
    
    # Visualization tabs
    viz_tabs = st.tabs([
        "Power Curve",
        "P-value Distribution", 
        "Effect Sensitivity",
        "Sample Size"
    ])
    
    with viz_tabs[0]:
        # Power curve
        effect_size = params.get('effect_size', params.get('cohens_d', 0.5))
        plots.render_power_curve(
            effect_sizes=[effect_size * 0.5, effect_size, effect_size * 1.5],
            alpha=params.get('alpha', 0.05),
            test_type=test_type
        )
    
    with viz_tabs[1]:
        # P-value distribution
        n = params.get('n_total', params.get('total_n', 100))
        effect = params.get('effect_size', params.get('cohens_d', 0.5))
        plots.render_p_value_distribution(
            n=n,
            effect_size=effect,
            alpha=params.get('alpha', 0.05),
            test_type=test_type
        )
    
    with viz_tabs[2]:
        # Effect sensitivity
        plots.render_effect_sensitivity(
            base_effect=params.get('effect_size', 0.5),
            n=params.get('n_total', 100),
            alpha=params.get('alpha', 0.05),
            test_type=test_type
        )
    
    with viz_tabs[3]:
        # Sample size optimization
        plots.render_sample_size_optimization(
            effect_size=params.get('effect_size', 0.5),
            desired_power=0.8,
            alpha=params.get('alpha', 0.05),
            test_type=test_type
        )

def render_advanced_analysis(results):
    """Render advanced analysis including Bayesian and simulations."""
    st.markdown("### 🔬 Advanced Statistical Analysis")
    
    analysis_tabs = st.tabs(["Bayesian", "Simulations", "Sequential"])
    
    with analysis_tabs[0]:
        # Bayesian analysis
        if ADVANCED_FEATURES:
            st.markdown("#### Bayesian Analysis")
            engine = BayesianEngine()
            
            # Perform Bayesian t-test if applicable
            if results.get('statistical_test_used') == 'two_sample_t_test':
                params = results.get('parameters', {})
                
                bayes_result = engine.bayesian_t_test(
                    n1=params.get('n_total', 100) // 2,
                    n2=params.get('n_total', 100) // 2,
                    mean1=0,
                    mean2=params.get('cohens_d', 0.5),
                    std1=1,
                    std2=1,
                    prior_type=st.session_state.get('bayesian_prior', 'uninformative')
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bayes Factor", f"{bayes_result['bayes_factor']:.2f}")
                    st.metric("Posterior Mean", f"{bayes_result['posterior_mean']:.3f}")
                
                with col2:
                    st.metric("P(Effect > 0)", f"{bayes_result['prob_positive_effect']:.2%}")
                    ci = bayes_result['credible_interval']
                    st.metric("95% Credible Interval", 
                             f"[{ci['lower']:.3f}, {ci['upper']:.3f}]")
                
                st.info(bayes_result['interpretation'])
    
    with analysis_tabs[1]:
        # Monte Carlo simulations
        if ADVANCED_FEATURES:
            st.markdown("#### Monte Carlo Simulations")
            
            if st.button("Run Simulations"):
                simulator = MonteCarloSimulator()
                sim_params = SimulationParameters(
                    n_simulations=st.session_state.get('n_simulations', 10000),
                    n_per_group=results.get('parameters', {}).get('n_total', 100) // 2,
                    effect_size=results.get('parameters', {}).get('effect_size', 0.5),
                    test_type=results.get('statistical_test_used', 'two_sample_t_test')
                )
                
                with st.spinner("Running simulations..."):
                    sim_results = simulator.simulate_trial(sim_params)
                    st.session_state.simulation_results = sim_results
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Simulated Power", f"{sim_results.power:.2%}")
                with col2:
                    st.metric("Mean P-value", f"{sim_results.mean_p_value:.4f}")
                with col3:
                    st.metric("Type I Error", f"{sim_results.type_i_error:.4f}"
                             if not np.isnan(sim_results.type_i_error) else "N/A")
    
    with analysis_tabs[2]:
        # Group sequential design
        st.markdown("#### Group Sequential Design")
        if st.session_state.get('sequential_design'):
            st.write("Sequential design configuration:", 
                    st.session_state['sequential_design'])
        else:
            st.info("Enable group sequential design in the sidebar to see analysis")

def render_references(results):
    """Render references and literature section."""
    st.markdown("### 📚 References & Literature")
    
    if 'research_papers_data' in results:
        papers = results['research_papers_data']
        st.markdown(f"Found {len(papers)} relevant papers")
        
        for paper in papers[:10]:  # Show first 10
            with st.expander(paper.get('title', 'Unknown Title')):
                st.write(f"**Authors**: {', '.join(paper.get('authors', []))}")
                st.write(f"**Journal**: {paper.get('journal', 'N/A')}")
                st.write(f"**Year**: {paper.get('year', 'N/A')}")
                if paper.get('abstract'):
                    st.write(f"**Abstract**: {paper['abstract'][:500]}...")
                if paper.get('url'):
                    st.write(f"**Link**: {paper['url']}")

if __name__ == "__main__":
    main()