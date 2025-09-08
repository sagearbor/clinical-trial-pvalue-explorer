"""Modular input forms for Streamlit interface."""

import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
import json


class DualModeInterface:
    """Implements dual-mode interface for AI-assisted and manual modes."""
    
    def __init__(self):
        """Initialize the dual-mode interface."""
        if 'ui_mode' not in st.session_state:
            st.session_state.ui_mode = 'ai_assisted'
        if 'manual_test_type' not in st.session_state:
            st.session_state.manual_test_type = 'two_sample_t_test'
    
    def render_mode_selector(self) -> str:
        """
        Render the mode selection interface.
        
        Returns:
            Selected mode ('ai_assisted' or 'manual')
        """
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### Analysis Mode")
        
        with col2:
            mode = st.radio(
                "Select Mode",
                options=['ai_assisted', 'manual'],
                format_func=lambda x: "🤖 AI-Assisted" if x == 'ai_assisted' else "🔧 Manual Expert",
                horizontal=True,
                key='ui_mode_selector',
                label_visibility='collapsed'
            )
            st.session_state.ui_mode = mode
        
        # Mode description
        if mode == 'ai_assisted':
            st.info(
                "**AI-Assisted Mode**: Describe your study in natural language. "
                "The AI will suggest the appropriate statistical test and parameters."
            )
        else:
            st.info(
                "**Manual Expert Mode**: Select your statistical test directly "
                "and input parameters manually."
            )
        
        return mode
    
    def render_ai_assisted_form(self) -> Dict[str, Any]:
        """
        Render the AI-assisted input form.
        
        Returns:
            Dictionary of form inputs
        """
        inputs = {}
        
        # Study description
        inputs['study_description'] = st.text_area(
            "Describe your clinical trial or study",
            placeholder=(
                "Example: We want to test if a new diabetes drug reduces HbA1c levels "
                "compared to placebo. We expect a moderate effect size based on preliminary data."
            ),
            height=120,
            key='ai_study_description'
        )
        
        # Advanced options in expander
        with st.expander("Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                inputs['confidence_threshold'] = st.slider(
                    "Minimum AI Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Minimum confidence level for AI suggestions"
                )
                
                inputs['include_alternatives'] = st.checkbox(
                    "Show alternative tests",
                    value=True,
                    help="Display alternative statistical tests that might be appropriate"
                )
            
            with col2:
                inputs['llm_provider'] = st.selectbox(
                    "LLM Provider",
                    options=['Default', 'GEMINI', 'OPENAI', 'ANTHROPIC', 'AZURE_OPENAI'],
                    help="Select the AI provider for analysis"
                )
                
                inputs['include_research'] = st.checkbox(
                    "Include literature search",
                    value=False,
                    help="Search PubMed and other sources for related studies"
                )
        
        # Research options if enabled
        if inputs['include_research']:
            st.markdown("#### Literature Search Settings")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                inputs['pubmed_papers'] = st.number_input(
                    "PubMed papers",
                    min_value=0,
                    max_value=20,
                    value=5
                )
            
            with col2:
                inputs['clinicaltrials_papers'] = st.number_input(
                    "ClinicalTrials.gov",
                    min_value=0,
                    max_value=20,
                    value=3
                )
            
            with col3:
                inputs['arxiv_papers'] = st.number_input(
                    "arXiv papers",
                    min_value=0,
                    max_value=20,
                    value=2
                )
        
        return inputs
    
    def render_manual_form(self, available_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Render the manual expert input form.
        
        Args:
            available_tests: List of available statistical tests
            
        Returns:
            Dictionary of form inputs
        """
        inputs = {}
        
        # Test selection
        test_names = [test['name'] for test in available_tests]
        test_descriptions = [test.get('description', '') for test in available_tests]
        
        selected_index = st.selectbox(
            "Select Statistical Test",
            range(len(test_names)),
            format_func=lambda i: f"{test_names[i]} - {test_descriptions[i][:50]}...",
            key='manual_test_selector'
        )
        
        inputs['test_type'] = available_tests[selected_index]['id']
        selected_test = available_tests[selected_index]
        
        # Display test information
        with st.expander("Test Information", expanded=True):
            st.markdown(f"**Description**: {selected_test.get('description', 'N/A')}")
            st.markdown(f"**Use Case**: {selected_test.get('use_case', 'N/A')}")
            st.markdown(f"**Assumptions**: {', '.join(selected_test.get('assumptions', []))}")
        
        # Dynamic parameter inputs based on test type
        st.markdown("### Test Parameters")
        inputs['parameters'] = self._render_test_parameters(inputs['test_type'])
        
        return inputs
    
    def _render_test_parameters(self, test_type: str) -> Dict[str, Any]:
        """
        Render parameter inputs specific to the selected test.
        
        Args:
            test_type: Type of statistical test
            
        Returns:
            Dictionary of test parameters
        """
        params = {}
        
        if test_type == 'two_sample_t_test':
            col1, col2 = st.columns(2)
            with col1:
                params['n_total'] = st.number_input(
                    "Total Sample Size",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10,
                    help="Total number of participants across both groups"
                )
                params['effect_size'] = st.number_input(
                    "Effect Size (Cohen's d)",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.5,
                    step=0.1,
                    help="Expected effect size (0.2=small, 0.5=medium, 0.8=large)"
                )
            with col2:
                params['alpha'] = st.selectbox(
                    "Significance Level (α)",
                    options=[0.01, 0.05, 0.10],
                    index=1,
                    help="Type I error rate"
                )
                params['power'] = st.slider(
                    "Desired Power",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.8,
                    step=0.05,
                    help="Probability of detecting a true effect (1 - β)"
                )
        
        elif test_type == 'chi_square':
            col1, col2 = st.columns(2)
            with col1:
                params['n_total'] = st.number_input(
                    "Total Sample Size",
                    min_value=20,
                    max_value=10000,
                    value=200,
                    step=10
                )
                params['n_groups'] = st.number_input(
                    "Number of Groups",
                    min_value=2,
                    max_value=10,
                    value=2
                )
            with col2:
                params['n_categories'] = st.number_input(
                    "Number of Categories",
                    min_value=2,
                    max_value=10,
                    value=2
                )
                params['effect_size'] = st.number_input(
                    "Effect Size (Cramér's V)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    help="0.1=small, 0.3=medium, 0.5=large"
                )
        
        elif test_type == 'one_way_anova':
            col1, col2 = st.columns(2)
            with col1:
                params['n_total'] = st.number_input(
                    "Total Sample Size",
                    min_value=30,
                    max_value=10000,
                    value=150,
                    step=10
                )
                params['n_groups'] = st.number_input(
                    "Number of Groups",
                    min_value=3,
                    max_value=10,
                    value=3
                )
            with col2:
                params['effect_size'] = st.number_input(
                    "Effect Size (f)",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.25,
                    step=0.05,
                    help="0.1=small, 0.25=medium, 0.4=large"
                )
                params['alpha'] = st.selectbox(
                    "Significance Level (α)",
                    options=[0.01, 0.05, 0.10],
                    index=1
                )
        
        elif test_type == 'correlation':
            col1, col2 = st.columns(2)
            with col1:
                params['n'] = st.number_input(
                    "Sample Size",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10
                )
                params['correlation'] = st.slider(
                    "Expected Correlation (r)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.05
                )
            with col2:
                params['method'] = st.radio(
                    "Correlation Method",
                    options=['pearson', 'spearman'],
                    format_func=lambda x: x.capitalize()
                )
                params['alpha'] = st.selectbox(
                    "Significance Level (α)",
                    options=[0.01, 0.05, 0.10],
                    index=1
                )
        
        else:
            # Generic parameters for other tests
            params['n_total'] = st.number_input(
                "Sample Size",
                min_value=10,
                max_value=10000,
                value=100,
                step=10
            )
            params['alpha'] = st.selectbox(
                "Significance Level (α)",
                options=[0.01, 0.05, 0.10],
                index=1
            )
            st.warning(f"Parameter input for {test_type} is using generic settings.")
        
        return params


class ParameterPresetsForm:
    """Manages parameter presets for common study designs."""
    
    PRESETS = {
        'phase_2_oncology': {
            'name': 'Phase II Oncology',
            'description': 'Single-arm efficacy study in cancer',
            'test_type': 'one_sample_proportion',
            'parameters': {
                'n': 40,
                'p0': 0.20,  # Historical response rate
                'p1': 0.35,  # Target response rate
                'alpha': 0.05,
                'power': 0.80
            }
        },
        'phase_3_superiority': {
            'name': 'Phase III Superiority',
            'description': 'Two-arm superiority trial',
            'test_type': 'two_sample_t_test',
            'parameters': {
                'n_total': 300,
                'effect_size': 0.3,
                'alpha': 0.05,
                'power': 0.80
            }
        },
        'phase_3_non_inferiority': {
            'name': 'Phase III Non-Inferiority',
            'description': 'Non-inferiority comparison',
            'test_type': 'two_sample_t_test',
            'parameters': {
                'n_total': 400,
                'effect_size': 0.0,  # Null difference
                'margin': 0.15,  # Non-inferiority margin
                'alpha': 0.025,  # One-sided
                'power': 0.80
            }
        },
        'dose_finding': {
            'name': 'Dose-Finding Study',
            'description': 'Multiple dose levels comparison',
            'test_type': 'one_way_anova',
            'parameters': {
                'n_total': 120,
                'n_groups': 4,  # Placebo + 3 doses
                'effect_size': 0.35,
                'alpha': 0.05,
                'power': 0.80
            }
        },
        'biomarker_association': {
            'name': 'Biomarker Association',
            'description': 'Correlation between biomarker and outcome',
            'test_type': 'correlation',
            'parameters': {
                'n': 150,
                'correlation': 0.4,
                'method': 'pearson',
                'alpha': 0.05
            }
        },
        'pilot_feasibility': {
            'name': 'Pilot/Feasibility',
            'description': 'Small pilot study for feasibility',
            'test_type': 'two_sample_t_test',
            'parameters': {
                'n_total': 30,
                'effect_size': 0.8,  # Large effect expected
                'alpha': 0.10,  # More lenient
                'power': 0.70  # Lower power acceptable
            }
        }
    }
    
    def render_preset_selector(self) -> Optional[Dict[str, Any]]:
        """
        Render preset selector and return selected preset.
        
        Returns:
            Selected preset dictionary or None
        """
        preset_names = ['None'] + [p['name'] for p in self.PRESETS.values()]
        
        selected = st.selectbox(
            "Load Study Design Preset",
            options=preset_names,
            help="Select a common study design to pre-populate parameters"
        )
        
        if selected != 'None':
            preset_key = [k for k, v in self.PRESETS.items() if v['name'] == selected][0]
            preset = self.PRESETS[preset_key]
            
            # Display preset information
            st.info(f"**{preset['name']}**: {preset['description']}")
            
            # Show preset parameters
            with st.expander("Preset Parameters", expanded=True):
                params_str = json.dumps(preset['parameters'], indent=2)
                st.code(params_str, language='json')
            
            return preset
        
        return None
    
    def apply_preset(self, preset: Dict[str, Any]) -> None:
        """
        Apply preset values to session state.
        
        Args:
            preset: Preset dictionary to apply
        """
        if preset:
            st.session_state.preset_applied = preset['name']
            st.session_state.manual_test_type = preset['test_type']
            for key, value in preset['parameters'].items():
                st.session_state[f'preset_{key}'] = value


class ResultsDisplayForm:
    """Manages results display configuration."""
    
    def render_display_options(self) -> Dict[str, bool]:
        """
        Render display options for results.
        
        Returns:
            Dictionary of display flags
        """
        st.markdown("### Display Options")
        
        col1, col2, col3 = st.columns(3)
        
        options = {}
        
        with col1:
            options['show_visualizations'] = st.checkbox(
                "Show Visualizations",
                value=True
            )
            options['show_confidence_intervals'] = st.checkbox(
                "Show Confidence Intervals",
                value=True
            )
        
        with col2:
            options['show_effect_sizes'] = st.checkbox(
                "Show Effect Sizes",
                value=True
            )
            options['show_interpretations'] = st.checkbox(
                "Show Interpretations",
                value=True
            )
        
        with col3:
            options['show_research'] = st.checkbox(
                "Show Literature",
                value=False
            )
            options['show_scenarios'] = st.checkbox(
                "Show Scenarios",
                value=False
            )
        
        return options
    
    def render_export_options(self) -> None:
        """Render export options for results."""
        st.markdown("### Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export as PDF", use_container_width=True):
                st.info("PDF export will be implemented")
        
        with col2:
            if st.button("📈 Export Plots", use_container_width=True):
                st.info("Plot export will be implemented")
        
        with col3:
            if st.button("📋 Copy Parameters", use_container_width=True):
                st.info("Parameter copy will be implemented")