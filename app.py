# app.py — Clean UI + per‑source sliders + bell‑curve scenarios + reference table
# Drop this into your repo root to replace the current Streamlit app.

import os
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# If you keep local fallbacks for t‑test calculations
try:
    import sys
    sys.path.insert(0, 'src')
    from statistical_utils import calculate_p_value_from_N_d, calculate_power_from_N_d
except Exception:
    calculate_p_value_from_N_d = None
    calculate_power_from_N_d = None

# ---------------- Config ----------------
BASE_URL = os.getenv("CTPE_BACKEND_URL", "http://localhost:8123")
URL_PROCESS = f"{BASE_URL}/process_idea"
URL_AVAILABLE = f"{BASE_URL}/available_tests"
URL_SCENARIOS = f"{BASE_URL}/analyze_scenarios"

st.set_page_config(layout="wide", page_title="Clinical Study Designer (AI)", page_icon="🧪")
st.title("🧪 Clinical Study Designer (AI)")
st.caption("Describe a study → get the right statistical test, sample size hints, and (optionally) real citations.")

# ---------------- Session defaults ----------------
def _d(key, val):
    if key not in st.session_state: st.session_state[key] = val

for k, v in {
    'study_analysis': {},
    'suggested_test_type': 'two_sample_t_test',
    'selected_test_type': 'two_sample_t_test',
    'available_tests': [],
    'test_parameters': {},
    'analysis_results': {},
    'calculated_statistics': {},
    'multi_scenarios': {},
    'scenario_analysis_complete': False,
    'recommended_scenario': None,
    'llm_provider_used': None,
    'references': [],
    'reference_view': 'list',
    'lit_search_was_enabled': True,
    'current_N': 100,
    'current_d': 0.5,
}.items():
    _d(k, v)

# ---------------- Helpers ----------------
def fetch_available_tests():
    try:
        r = requests.get(URL_AVAILABLE, timeout=30)
        if r.status_code == 200:
            data = r.json()
            # If data is a list, it's the tests directly
            if isinstance(data, list):
                # Convert to expected format
                formatted_tests = [{"test_id": t["value"], "name": t["label"]} for t in data]
                return [], formatted_tests
            # Otherwise try the old format
            return data.get('enhanced_test_info', []), data.get('available_tests', [])
        return [], []
    except Exception:
        return [], []

TEST_DISPLAY = {
    "two_sample_t_test": "Two-Sample t-Test",
    "chi_square": "Chi-Square Test",
    "one_way_anova": "One-Way ANOVA",
    "correlation": "Correlation Analysis",
    "logistic_regression": "Logistic Regression",
    "fisher_exact": "Fisher's Exact Test",
    "ancova": "ANCOVA",
}

def get_test_display_name(t):
    return TEST_DISPLAY.get(t, t.replace('_',' ').title())

# Minimal frontend helper functions used by tests
def get_effect_size_interpretation(test_type, effect_size):
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


def fetch_available_tests():
    try:
        r = requests.get(URL_AVAILABLE, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return data.get('enhanced_test_info', []), data.get('available_tests', [])
        return [], []
    except Exception:
        return [], []


def calculate_test_specific_statistics(test_type, parameters):
    try:
        payload = {
            "study_description": f"Calculate {test_type} statistics with parameters: {json.dumps(parameters)}",
            "test_override": test_type,
            "parameters_override": parameters,
        }
        r = requests.post(URL_PROCESS, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return {
                "p_value": data.get('calculated_p_value'),
                "power": data.get('calculated_power'),
                "effect_size": data.get('parameters', {}).get('effect_size_value') or data.get('parameters', {}).get('cohens_d'),
                "test_statistic": data.get('test_statistic'),
                "sample_size": data.get('parameters', {}).get('total_n'),
                "error": data.get('calculation_error')
            }
        else:
            return {"error": f"Backend calculation failed: {r.status_code}"}
    except Exception as e:
        if test_type == "two_sample_t_test":
            try:
                N_total = parameters.get('N_total') or parameters.get('total_n')
                cohens_d = parameters.get('cohens_d') or parameters.get('effect_size') or parameters.get('effect_size_value')
                alpha = parameters.get('alpha', 0.05)
                if N_total and cohens_d is not None:
                    p_val, p_msg = calculate_p_value_from_N_d(N_total, cohens_d)
                    power_val, power_msg = calculate_power_from_N_d(N_total, cohens_d, alpha)
                    return {
                        "p_value": p_val,
                        "power": power_val,
                        "effect_size": cohens_d,
                        "sample_size": N_total,
                        "error": p_msg or power_msg
                    }
            except Exception:
                pass
        return {"error": f"Error calculating statistics: {str(e)}"}


def format_test_results(test_type, results):
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
    if test_type == 'chi_square':
        formatted['cramers_v'] = results.get('cramers_v')
        formatted['contingency_table'] = results.get('contingency_table')
        formatted['expected_frequencies'] = results.get('expected_frequencies')
    elif test_type == 'one_way_anova':
        formatted['f_statistic'] = results.get('f_statistic')
        formatted['eta_squared'] = results.get('eta_squared')
        formatted['group_means'] = results.get('group_means')
    elif test_type == 'correlation':
        formatted['correlation_coefficient'] = results.get('correlation_coefficient')
        formatted['correlation_type'] = results.get('correlation_type')
    return formatted

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    st.caption("Literature search option moved to the main input area")

    # Per‑source sliders (your older UX)
    with st.expander("🔬 Sources & Limits", expanded=True):
        pubmed_papers = st.slider("PubMed papers", 0, 20, 4)
        clinicaltrials_papers = st.slider("ClinicalTrials.gov papers", 0, 20, 3)
        arxiv_papers = st.slider("arXiv papers", 0, 20, 2)
        st.session_state.max_papers = pubmed_papers + clinicaltrials_papers + arxiv_papers
        st.session_state.pubmed_papers = pubmed_papers
        st.session_state.clinicaltrials_papers = clinicaltrials_papers
        st.session_state.arxiv_papers = arxiv_papers
        st.caption(f"Total papers: {st.session_state.max_papers}")

    llm = st.selectbox("LLM Provider", ["Default","GEMINI","OPENAI","AZURE_OPENAI","ANTHROPIC"], index=0)
    st.session_state.selected_llm_provider = None if llm == "Default" else llm

# ---------------- Section 1: Describe idea ----------------
st.header("1) Describe Your Research Idea")
idea = st.text_area("Be specific about groups/arms, outcome, comparisons:", height=150)
# Inline literature search toggle (moved from sidebar)
do_lit = st.checkbox("Include literature search", value=st.session_state.get('lit_search_was_enabled', True))
colA, colB = st.columns([0.8,0.2])
with colA:
    run = st.button("🔍 Analyze Study", type="primary")
with colB:
    reset = st.button("↺ Reset")

if reset:
    for k in list(st.session_state.keys()):
        if k not in ("selected_llm_provider", "pubmed_papers", "clinicaltrials_papers", "arxiv_papers", "max_papers"):
            del st.session_state[k]
    st.rerun()

    st.experimental_rerun()

# ---------------- Backend calls ----------------
if run:
    if not idea or not idea.strip():
        st.error("Please enter a research idea.")
    else:
        st.session_state.lit_search_was_enabled = do_lit
        payload = {
            "study_description": idea.strip(),
            "llm_provider": st.session_state.get('selected_llm_provider'),
            "include_research": bool(do_lit),
            "max_papers": int(st.session_state.get('max_papers', 6)),
            "pubmed_papers": int(st.session_state.get('pubmed_papers', 0)),
            "arxiv_papers": int(st.session_state.get('arxiv_papers', 0)),
            "clinicaltrials_papers": int(st.session_state.get('clinicaltrials_papers', 0)),
        }
        with st.spinner("Thinking through your design…"):
            try:
                r = requests.post(URL_PROCESS, json=payload, timeout=120)
                if r.status_code != 200:
                    st.error(f"Backend error {r.status_code}")
                    st.write(r.text)
                else:
                    data = r.json()
                    st.session_state.study_analysis = data
                    
                    # Display prominent warning if LLM was unavailable
                    if data.get("llm_warning"):
                        st.error(f"🚨 {data.get('llm_warning')}")
                    
                    if data.get("default_used"):
                        st.warning("⚠️ Could not detect specific test type from your description. Defaulted to two-sample t-test. Please manually select the appropriate test below if this is incorrect.")
                    elif data.get("fallback_mode"):
                        st.info("ℹ️ Using pattern matching to detect test type (LLM unavailable). Please verify the suggestion is appropriate.")
                    
                    st.session_state.llm_provider_used = data.get("llm_provider_used")
                    st.session_state.suggested_test_type = data.get("suggested_study_type", "two_sample_t_test")
                    st.session_state.selected_test_type = st.session_state.suggested_test_type
                    st.session_state.test_parameters = data.get("parameters", {})
                    # Prefer backend-provided references when include_research True
                    if payload.get('include_research'):
                        st.session_state.references = data.get("references", [])
                    else:
                        st.session_state.references = []
                    # Kick scenarios only if user wants the richer view right away
                    sc_payload = {k: payload[k] for k in ("study_description","llm_provider")}
                    sr = requests.post(URL_SCENARIOS, json=sc_payload, timeout=60)
                    if sr.status_code == 200:
                        st.session_state.multi_scenarios = (sr.json() or {}).get("scenarios", {})
                        st.session_state.recommended_scenario = (sr.json() or {}).get("recommended_scenario")
                        st.session_state.scenario_analysis_complete = True
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------- Section 2: Results ----------------
analysis = st.session_state.get("study_analysis") or {}
if analysis:
    st.header("2) Results")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.4,1,1])
        with col1:
            st.subheader(f"Suggested Test: {get_test_display_name(st.session_state.get('suggested_test_type'))}")
            st.caption(analysis.get("rationale") or "No rationale provided.")
            if st.session_state.get("llm_provider_used"):
                st.caption(f"Provider: {st.session_state['llm_provider_used']}")
        with col2:
            p = analysis.get("calculated_p_value")
            pw = analysis.get("calculated_power")
            st.metric("Calculated Power", f"{(pw or 0)*100:.0f}%")
            st.metric("p‑value", f"{p if p is not None else '—'}")
        with col3:
            params = analysis.get("parameters") or {}
            st.metric("Total N (est.)", params.get("total_n", "—"))
            st.metric("Effect Size", params.get("effect_size_value", "—"))

    with st.expander("Details & Alternatives", expanded=False):
        st.json({
            "parameters": analysis.get("parameters"),
            "study_design": analysis.get("study_design"),
            "data_type": analysis.get("data_type"),
            "alternative_tests": analysis.get("alternative_tests")
        })

    # ---- References: list <-> table toggle ----
    if st.session_state.get('lit_search_was_enabled'):
        refs = st.session_state.get('references') or []
        papers = analysis.get('research_papers_data') or []
        # Show references if either simple markdown refs or structured paper data exists
        if refs or papers:
            # Entire references section in a single expander so toggling view doesn't collapse the whole section unintentionally
            with st.expander("📚 References", expanded=True):
                counts_msg = f"Found {len(papers)} structured records" if papers else f"Found {len(refs)} markdown references"
                st.caption(counts_msg)

                # Default to Table view (index=1)
                toggle = st.radio("View as", ["List","Table"], horizontal=True, index=1)

                if toggle == "List":
                    # Prefer markdown list if available, otherwise render titles from structured data
                    if refs:
                        for r in refs:
                            st.markdown(f"- {r}")
                    else:
                        for p in papers:
                            auth = ', '.join((p.get('authors') or [])[:2]) + (' et al.' if p.get('authors') and len(p.get('authors'))>2 else '')
                            title_line = f"**{p.get('title')}** {auth} ({p.get('year')}). *{p.get('journal')}*"
                            if p.get('url'):
                                title_line += f" [View]({p.get('url')})"
                            st.markdown(f"- {title_line}")
                else:
                    # Table view: prefer structured records
                    wrap_text = st.checkbox("Wrap text in title column", value=False)
                    
                    if not papers and refs:
                        # best effort: try to parse minimal fields from markdown-ish strings
                        df = pd.DataFrame({"Reference": refs})
                    else:
                        rows = []
                        has_text_extracted = False
                        for p in papers:
                            authors = p.get('authors') or []
                            if isinstance(authors, list):
                                auth = ', '.join(authors[:2]) + (' et al.' if len(authors)>2 else '')
                            else:
                                auth = authors or ''
                            
                            # Format sample size with superscript if text-extracted
                            sample_size = p.get('sample_size')
                            if sample_size and p.get('sample_size_method') == 'text_extraction':
                                sample_size = f"{sample_size}¹"
                                has_text_extracted = True
                            
                            # Determine source for grouping/filtering
                            source = "Unknown"
                            if "ClinicalTrials.gov" in p.get('journal', ''):
                                source = "CT.gov"
                            elif p.get('url', '').startswith('https://pubmed'):
                                source = "PubMed"
                            elif 'arxiv' in p.get('url', '').lower():
                                source = "arXiv"
                            
                            rows.append({
                                "Title": p.get('title'),
                                "Year": p.get('year'),
                                "Authors": auth,
                                "Journal": p.get('journal'),
                                "Source": source,
                                "N": sample_size,
                                "p-value": p.get('p_value'),
                                "Study Signal": p.get('study_signal', 'Unknown'),
                                "Link": p.get('url')
                            })
                        df = pd.DataFrame(rows)
                        
                        # Configure columns based on wrap_text setting
                        column_config = {
                            "Link": st.column_config.LinkColumn(
                                "Link",
                                help="Click to open paper",
                                display_text="View"
                            )
                        }
                        
                        if not wrap_text:
                            # Limit title width when not wrapping
                            column_config["Title"] = st.column_config.TextColumn(
                                "Title",
                                max_chars=60,
                                help="Full title (toggle wrap to see full text)"
                            )
                        
                        st.data_editor(
                            df,
                            column_config=column_config,
                            use_container_width=True,
                            hide_index=True,
                            disabled=True  # Make read-only
                        )
                        
                        if has_text_extracted:
                            st.caption("¹ Sample size extracted from abstract text (may be more prone to incorrect N citation)")

# ---------------- Section 3: Scenarios ----------------
sc_ok = st.session_state.get('scenario_analysis_complete') and isinstance(st.session_state.get('multi_scenarios'), dict) and st.session_state['multi_scenarios']
if sc_ok:
    st.header("3) Multi‑Scenario Statistical Design Dashboard")

    scenarios = st.session_state['multi_scenarios']
    rec = st.session_state.get('recommended_scenario')

    # Comparison dashboard (sample size, alpha, power, effect size)
    def comparison_chart(sc_map: dict, recommended: str|None):
        names, Ns, alphas, powers, effects = [], [], [], [], []
        for key, sc in sc_map.items():
            p = sc.get('parameters', {}) if isinstance(sc, dict) else {}
            names.append(sc.get('name', key).title())
            Ns.append(p.get('total_n'))
            alphas.append(sc.get('target_p_value', p.get('alpha', 0.05)))
            powers.append(p.get('power', 0.8))
            effects.append(p.get('effect_size_value', 0.5))
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Sample Size','α level','Power','Effect Size'))
        fig.add_scatter(x=names,y=Ns,mode='lines+markers',name='N',row=1,col=1)
        fig.add_scatter(x=names,y=alphas,mode='lines+markers',name='alpha',row=1,col=2)
        fig.add_scatter(x=names,y=[p*100 for p in powers],mode='lines+markers',name='power %',row=2,col=1)
        fig.add_scatter(x=names,y=effects,mode='lines+markers',name='effect',row=2,col=2)
        fig.update_layout(height=520, showlegend=False)
        return fig

    # Bell‑curve view (what you asked for)
    def bell_curve_chart(sc_map: dict):
        # Treat each scenario's effect size as the mean of a unit‑variance normal for visualization
        x = np.linspace(-2.5, 2.5, 300)
        fig = go.Figure()
        for key, sc in sc_map.items():
            p = sc.get('parameters', {}) if isinstance(sc, dict) else {}
            mu = float(p.get('effect_size_value', 0.5) or 0.0)
            y = norm.pdf(x, loc=mu, scale=1.0)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=sc.get('name', key).title()))
        fig.update_layout(title="Scenario Bell Curves (effect size distributions — illustrative)", height=420, xaxis_title="Effect size (Cohen's d)", yaxis_title="Density")
        return fig

    tab1, tab2 = st.tabs(["📈 Comparison","🔔 Bell Curves"])
    with tab1:
        st.plotly_chart(comparison_chart(scenarios, rec), use_container_width=True)
    with tab2:
        st.plotly_chart(bell_curve_chart(scenarios), use_container_width=True)

# --------------- Manual override & params ---------------
if analysis:
    st.header("4) Select & Configure Statistical Test")
    # Always refresh test catalog to ensure we have all tests
    enh, avail = fetch_available_tests()
    if avail and len(avail) > 1:  # If we got the full list
        st.session_state['available_tests'] = avail
    elif not st.session_state['available_tests']:  # Fallback if nothing loaded
        st.session_state['available_tests'] = [{"test_id":"two_sample_t_test","name":"Two‑Sample t‑test"}]
    opts = st.session_state['available_tests']
    ids = [o['test_id'] for o in opts]
    names = [o.get('name', get_test_display_name(o['test_id'])) for o in opts]
    idx = ids.index(st.session_state['suggested_test_type']) if st.session_state['suggested_test_type'] in ids else 0
    sel_idx = st.selectbox("Choose statistical test:", range(len(names)), index=idx, format_func=lambda i: f"{names[i]} {'🤖' if ids[i]==st.session_state['suggested_test_type'] else ''}")
    st.session_state['selected_test_type'] = ids[sel_idx]

    # Show appropriate parameters for each test type
    test_type = st.session_state['selected_test_type']
    
    # Toggle for advanced mode
    advanced_mode = st.checkbox("🔧 Advanced Parameters", help="Show additional parameters for statistical experts")
    
    if test_type == 'two_sample_t_test':
        st.subheader("Two-Sample T-Test Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=3, step=1, value=int(st.session_state.get('current_N',100)))
        with c2:
            d = st.number_input("Cohen's d", value=float(st.session_state.get('current_d',0.5)), step=0.01, format="%.3f")
        
        if advanced_mode:
            st.markdown("### 🔧 Advanced T-Test Settings")
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01)
                power_target = st.number_input("Target Power", value=0.80, min_value=0.5, max_value=0.99, step=0.05)
                allocation_ratio = st.number_input("Allocation ratio (n1:n2)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
            with adv_c2:
                equal_var = st.checkbox("Assume equal variances", value=True)
                one_tailed = st.checkbox("One-tailed test", value=False)
                continuity_correction = st.checkbox("Continuity correction", value=False)
                
    elif test_type == 'mixed_effects':
        st.subheader("Mixed Effects Model (MMRM) Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N (subjects)", min_value=10, step=1, value=200)
            n_timepoints = st.number_input("Number of timepoints", min_value=2, max_value=20, value=4)
            effect_size = st.number_input("Effect size", value=0.5, step=0.1)
        with c2:
            n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=2)
            dropout_rate = st.slider("Expected dropout rate", 0.0, 0.5, 0.2, 0.05)
        
        # Advanced parameters section (outside columns)
        if advanced_mode:
            st.markdown("### 🔧 Advanced MMRM Settings")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                correlation = st.slider("Within-subject correlation", 0.0, 0.9, 0.5, 0.1)
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01)
                n_covariates = st.number_input("Number of covariates", min_value=0, max_value=20, value=3)
            with adv_c2:
                covariance = st.selectbox("Covariance structure", 
                    ["Unstructured", "Compound Symmetry", "AR(1)", "Toeplitz", "Heterogeneous CS", "Ante-dependence"])
                missing_data = st.selectbox("Missing data method",
                    ["MAR (Missing at Random)", "MCAR", "MNAR", "Multiple Imputation"])
                baseline_adjustment = st.checkbox("Baseline as covariate", value=True)
            with adv_c3:
                random_slope = st.checkbox("Include random slopes", value=False)
                random_intercept = st.checkbox("Include random intercepts", value=True)
                center_effects = st.checkbox("Include center effects", value=False)
                interaction_test = st.checkbox("Test treatment × time", value=True)
                
    elif test_type == 'chi_square':
        st.subheader("Chi-Square Test Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=20, step=1, value=200)
            n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=2)
        with c2:
            n_categories = st.number_input("Number of categories", min_value=2, max_value=10, value=2)
            effect_size = st.number_input("Cramér's V", value=0.3, min_value=0.1, max_value=1.0, step=0.05)
        
        # Advanced parameters (outside columns for proper display)  
        if advanced_mode:
            st.markdown("### 🔧 Advanced Chi-Square Settings")
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                expected_freq = st.number_input("Min expected frequency", value=5, min_value=1)
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="chi_alpha")
            with adv_c2:
                continuity = st.checkbox("Yates continuity correction", value=False)
                monte_carlo = st.checkbox("Use Monte Carlo simulation", value=False)
            
    elif test_type == 'logistic_regression':
        st.subheader("Logistic Regression Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=50, step=10, value=300)
            n_predictors = st.number_input("Number of predictors", min_value=1, max_value=50, value=5)
            event_rate = st.slider("Baseline event rate", 0.05, 0.95, 0.5, 0.05)
        with c2:
            odds_ratio = st.number_input("Target odds ratio", value=2.0, min_value=0.1, max_value=10.0, step=0.1)
            r_squared = st.slider("Pseudo R²", 0.0, 0.5, 0.15, 0.05)
        
        # Advanced parameters (outside columns for proper display)
        if advanced_mode:
            st.markdown("### 🔧 Advanced Logistic Regression Settings")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                interaction_terms = st.number_input("Interaction terms", min_value=0, max_value=10, value=0)
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01)
                multicollinearity = st.slider("Max VIF allowed", 1.0, 10.0, 5.0, 0.5)
            with adv_c2:
                link_function = st.selectbox("Link function", ["Logit", "Probit", "Complementary log-log"])
                regularization = st.selectbox("Regularization", ["None", "L1 (Lasso)", "L2 (Ridge)", "Elastic Net"])
                if regularization == "Elastic Net":
                    l1_ratio = st.slider("L1 ratio", 0.0, 1.0, 0.5, 0.1)
            with adv_c3:
                bootstrap_ci = st.checkbox("Bootstrap confidence intervals", value=False)
                robust_se = st.checkbox("Robust standard errors", value=False)
                clustered_se = st.checkbox("Clustered standard errors", value=False)
                
    elif test_type == 'cox_regression':
        st.subheader("Cox Proportional Hazards Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=50, step=10, value=500)
            n_events = st.number_input("Expected events", min_value=10, step=5, value=100)
            follow_up = st.number_input("Median follow-up (months)", min_value=1, value=24)
        with c2:
            hazard_ratio = st.number_input("Target hazard ratio", value=0.7, min_value=0.1, max_value=5.0, step=0.1)
            n_covariates = st.number_input("Number of covariates", min_value=0, max_value=30, value=5)
        
        # Advanced parameters (outside columns for proper display)
        if advanced_mode:
            st.markdown("### 🔧 Advanced Cox PH Settings")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                censoring_rate = st.slider("Censoring rate", 0.0, 0.8, 0.3, 0.05)
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="cox_alpha")
                stratification = st.checkbox("Use stratification", value=False)
            with adv_c2:
                ph_assumption = st.checkbox("Check proportional hazards", value=True)
                time_varying = st.checkbox("Time-varying covariates", value=False)
                interaction_terms = st.checkbox("Include interactions", value=False)
            with adv_c3:
                competing_risks = st.checkbox("Account for competing risks", value=False)
                frailty_model = st.checkbox("Include frailty term", value=False)
                recurrent_events = st.checkbox("Handle recurrent events", value=False)
            
    elif test_type == 'one_way_anova':
        st.subheader("One-Way ANOVA Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=30, step=5, value=150)
            n_groups = st.number_input("Number of groups", min_value=3, max_value=10, value=3)
        with c2:
            effect_size = st.number_input("Cohen's f", value=0.25, min_value=0.1, max_value=1.0, step=0.05,
                                        help="Small=0.10, Medium=0.25, Large=0.40")
        
        # Advanced parameters (outside columns for proper display)
        if advanced_mode:
            st.markdown("### 🔧 Advanced ANOVA Settings")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="anova_alpha")
                equal_variance = st.checkbox("Assume equal variances", value=True)
                balanced_design = st.checkbox("Balanced design", value=True)
            with adv_c2:
                post_hoc = st.selectbox("Post-hoc test", 
                    ["None", "Tukey HSD", "Bonferroni", "Scheffe", "Duncan", "Newman-Keuls"])
                effect_type = st.selectbox("Effect size type", ["Cohen's f", "Eta-squared", "Omega-squared"])
            with adv_c3:
                sphericity_correction = st.checkbox("Apply sphericity correction", value=False)
                welch_anova = st.checkbox("Use Welch's ANOVA", value=False, 
                                        help="For unequal variances")
                
    elif test_type == 'correlation':
        st.subheader("Correlation Analysis Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=10, step=5, value=100)
            correlation_type = st.selectbox("Correlation type", ["Pearson", "Spearman", "Kendall"])
        with c2:
            effect_size = st.number_input("Expected correlation (r)", 
                                        value=0.3, min_value=-1.0, max_value=1.0, step=0.05,
                                        help="Small=0.1, Medium=0.3, Large=0.5")
        
        # Advanced parameters (outside columns for proper display)
        if advanced_mode:
            st.markdown("### 🔧 Advanced Correlation Settings")
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="corr_alpha")
                test_type_corr = st.selectbox("Test type", ["Two-tailed", "One-tailed (positive)", "One-tailed (negative)"])
                confidence_level = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)
            with adv_c2:
                fisher_transform = st.checkbox("Use Fisher's z transformation", value=True)
                bootstrap_ci = st.checkbox("Bootstrap confidence intervals", value=False)
                partial_correlation = st.checkbox("Partial correlation", value=False)
                if partial_correlation:
                    n_control_vars = st.number_input("Control variables", min_value=1, max_value=10, value=1)
                    
    elif test_type in ['mann_whitney', 'wilcoxon_signed', 'kruskal_wallis']:
        st.subheader(f"{test_type.replace('_', ' ').title()} Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=10, step=1, value=100)
            if test_type == 'kruskal_wallis':
                n_groups = st.number_input("Number of groups", min_value=3, max_value=10, value=3)
        with c2:
            effect_size = st.number_input("Effect size (probability of superiority)", 
                                        value=0.65, min_value=0.5, max_value=1.0, step=0.05)
        
        # Advanced parameters (outside columns for proper display)
        if advanced_mode:
            st.markdown("### 🔧 Advanced Non-Parametric Settings")
            adv_c1, adv_c2 = st.columns(2)
            with adv_c1:
                ties_method = st.selectbox("Method for handling ties", ["average", "min", "max", "dense", "ordinal"])
                alpha = st.number_input("Alpha (α)", value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="nonparam_alpha")
            with adv_c2:
                continuity_correction = st.checkbox("Apply continuity correction", value=True)
                exact_test = st.checkbox("Use exact test (slower)", value=False)
                
    else:
        # Default parameters for other tests
        st.subheader(f"{test_type.replace('_', ' ').title()} Parameters")
        c1, c2 = st.columns(2)
        with c1:
            N = st.number_input("Total N", min_value=20, step=1, value=100)
        with c2:
            effect_size = st.number_input("Effect size", value=0.5, step=0.1)
        if advanced_mode:
            st.info("Advanced parameters coming soon for this test type.")
    
    # Store parameters for calculation
    st.session_state['current_N'] = N
    st.session_state['current_effect_size'] = locals().get('effect_size', locals().get('d', 0.5))
    
    # Calculate and show results
    if st.button("📊 Calculate", type="primary"):
        params = {
            'total_n': N,
            'effect_size_value': st.session_state['current_effect_size'],
            'alpha': locals().get('alpha', 0.05)
        }
        
        # Add test-specific parameters
        if test_type == 'mixed_effects':
            params.update({
                'n_timepoints': n_timepoints,
                'n_groups': n_groups,
                'dropout_rate': dropout_rate,
                'correlation': locals().get('correlation', 0.5)
            })
        elif test_type == 'logistic_regression':
            params.update({
                'n_predictors': n_predictors,
                'event_rate': event_rate,
                'odds_ratio': odds_ratio
            })
            
        # Call calculation
        calc_result = calculate_statistics(test_type, params)
        
        if calc_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("P-value", f"{calc_result.get('p_value', 'N/A'):.4f}" if calc_result.get('p_value') else "N/A")
            with col2:
                st.metric("Power", f"{calc_result.get('power', 0)*100:.1f}%" if calc_result.get('power') else "N/A")
            with col3:
                st.metric("Sample Size", calc_result.get('sample_size', N))
            
            if calc_result.get('error'):
                st.info(calc_result['error'])

st.markdown("---")
st.caption("Exploratory tool — consult a biostatistician for study planning.")
