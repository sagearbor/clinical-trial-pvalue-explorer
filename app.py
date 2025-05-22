# app.py
import streamlit as st
import requests
import numpy as np
from scipy import stats

# Configuration for the FastAPI backend URL
# Assumes FastAPI is running on localhost:8000
BACKEND_URL = "http://localhost:8000/process_idea"

def calculate_p_value_from_N_d(N_total, cohens_d):
    """
    Calculates the two-sided p-value for a two-sample t-test,
    given the total sample size (N_total, assumed to be equally split into two groups)
    and an observed Cohen's d.
    """
    if not isinstance(N_total, (int, float)) or N_total <= 2: # N_total must be > 2 for df > 0
        return None, "Total N must be a number greater than 2."
    if not isinstance(cohens_d, (int, float)):
        return None, "Cohen's d must be a number."
    
    N_total_int = int(N_total)

    if N_total_int <= 2: 
        return None, "Total N must be greater than 2 for valid degrees of freedom."

    n_per_group = N_total_int / 2.0
    if n_per_group <= 1: 
         return None, "Sample size per group (N_total / 2) must be greater than 1."

    if cohens_d == 0:
        return 1.0, "With Cohen's d = 0, the t-statistic is 0, leading to a p-value of 1.0 (no effect observed)."

    try:
        if (n_per_group / 2.0) <= 0:
            return None, "Invalid internal calculation for t-statistic (sqrt of non-positive)."
        t_statistic = cohens_d * np.sqrt(n_per_group / 2.0) 
    except FloatingPointError: 
        return None, "Numerical instability in t-statistic calculation."

    df = N_total_int - 2 
    if df <= 0: 
        return None, "Degrees of freedom (N_total - 2) must be positive."

    try:
        p_value = (1 - stats.t.cdf(abs(t_statistic), df)) * 2 
    except Exception as e:
        return None, f"Error during SciPy p-value calculation: {str(e)}"
        
    return p_value, None


st.set_page_config(layout="wide")
st.title("Clinical Trial P-Value Explorer (AI-Assisted)")

# Initialize session state variables if they don't exist
default_N = 100
default_d = 0.5

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
if 'llm_provider_used' not in st.session_state: # New session state variable
    st.session_state.llm_provider_used = ""
if 'references' not in st.session_state:
    st.session_state.references = []

if 'current_N' not in st.session_state:
    st.session_state.current_N = st.session_state.initial_N
if 'current_d' not in st.session_state:
    st.session_state.current_d = st.session_state.initial_cohens_d


with st.sidebar:
    st.header("About")
    st.info(
        "This tool helps explore the relationship between sample size (N), "
        "effect size (Cohen's d), and p-values for a hypothetical two-sample t-test. "
        "You can get an initial AI-generated estimate for N, Cohen's d, and a brief justification "
        "based on your research idea."
        "\n\n**Disclaimer:** The AI's estimates are very rough and for exploratory purposes only. "
        "Actual sample size calculations require careful statistical planning."
    )
    st.header("P-Value Calculation Formulas")
    st.markdown(
        r"""
        The p-value is calculated for a **two-sample, two-sided t-test** with equal group sizes.
        
        1.  **Sample size per group ($n_{\text{group}}$):**
            $n_{\text{group}} = N_{\text{total}} / 2$
        
        2.  **Cohen's d (Effect Size):**
            This is your input. It represents the standardized difference between two means:
            $d = (M_1 - M_2) / \sigma_{\text{pooled}}$
            
        3.  **t-statistic (from Cohen's d):**
            For two independent groups of equal size $n_{\text{group}}$:
            $t = d \cdot \sqrt{n_{\text{group}} / 2}$
            Substituting $n_{\text{group}} = N_{\text{total}} / 2$:
            $t = d \cdot \sqrt{(N_{\text{total}} / 2) / 2} = d \cdot \sqrt{N_{\text{total}} / 4} = d \cdot \frac{\sqrt{N_{\text{total}}}}{2}$

        4.  **Degrees of Freedom (df):**
            $df = N_{\text{total}} - 2$
            
        5.  **P-value:**
            Calculated from the t-distribution using the absolute $t$-statistic and $df$.
        """
    )


st.header("1. Describe Your Research Idea")
text_idea = st.text_area("Enter your clinical trial research idea here:", height=150, key="text_idea_input_v3") # Incremented key
url_idea = st.text_input("Or provide a URL with relevant content (optional):", key="url_idea_input_v3") # Incremented key

if st.button("Get AI Estimate for N, Cohen's d, and Justification", key="get_estimate_button_v3"): # Incremented key
    if not text_idea and not url_idea:
        st.error("Please provide either a research idea text or a URL.")
    else:
        payload = {"text_idea": text_idea, "url_idea": url_idea}
        st.session_state.llm_provider_used = "" # Reset before new call
        try:
            with st.spinner("Processing your idea with AI... this may take a moment."):
                response = requests.post(BACKEND_URL, json=payload, timeout=90) # Slightly increased timeout for LLM calls
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.llm_provider_used = data.get("llm_provider_used", "Unknown")  # Store provider

                if data.get("error"):
                    st.error(f"Error from backend (Provider: {st.session_state.llm_provider_used}): {data['error']}")
                    st.session_state.processed_idea_text = data.get("processed_idea", st.session_state.processed_idea_text)
                    st.session_state.estimation_justification = ""
                    st.session_state.references = []
                elif data.get("initial_N") is not None and data.get("initial_cohens_d") is not None:
                    st.session_state.initial_N = data["initial_N"]
                    st.session_state.initial_cohens_d = data["initial_cohens_d"]
                    st.session_state.current_N = data["initial_N"]
                    st.session_state.current_d = data["initial_cohens_d"]
                    st.session_state.estimation_justification = data.get("estimation_justification", "No justification provided by AI.")
                    st.session_state.processed_idea_text = data.get("processed_idea", "Idea processed successfully.")
                    st.session_state.references = data.get("references", [])
                    st.success(f"AI estimation received! (Provider: {st.session_state.llm_provider_used})")
                else:
                    st.error(f"AI estimation received (Provider: {st.session_state.llm_provider_used}), but data is incomplete. Using previous or default values.")
                    st.session_state.processed_idea_text = data.get("processed_idea", st.session_state.processed_idea_text)
                    st.session_state.estimation_justification = ""
                    st.session_state.references = []
            else:
                st.error(f"Failed to get estimation from backend. Status code: {response.status_code}")
                try:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Details: {error_detail}")
                except: 
                    st.error(f"Details: {response.text}")
                st.session_state.processed_idea_text = "Failed to process idea."
                st.session_state.estimation_justification = ""
                st.session_state.references = []
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Is it running at " + BACKEND_URL + "?")
        except requests.exceptions.Timeout:
            st.error("The request to the backend API timed out. The AI might be taking too long or the server is busy.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.estimation_justification = ""
            st.session_state.references = []

if st.session_state.processed_idea_text:
    with st.expander("View Processed Idea Sent to AI", expanded=False):
        st.caption("This is the text that was (or would be) sent to the AI model for estimation:")
        display_text = st.session_state.processed_idea_text
        if len(display_text) > 1500:
            display_text = display_text[:1500] + "..."
        st.markdown(f"```\n{display_text}\n```")

# Display LLM provider if available, outside the button click logic for persistence
if st.session_state.llm_provider_used and st.session_state.estimation_justification: # Only show if justification is also present
    st.caption(f"Estimation based on input from: **{st.session_state.llm_provider_used}**")

if st.session_state.estimation_justification:
     with st.expander("AI's Justification for N and Cohen's d Estimates", expanded=True):
        st.info(st.session_state.estimation_justification)

if st.session_state.references:
    with st.expander("References", expanded=False):
        for ref in st.session_state.references:
            st.markdown(f"- {ref}")


st.header("2. Adjust Parameters and See P-Value")
st.markdown(
    "Use the sliders or input boxes below to adjust the Total Number of Participants (N) "
    "and the Expected Cohen's d. The p-value will update automatically. "
)

col1, col2 = st.columns(2)

with col1:
    N_total_input = st.number_input(
        "Total Number of Participants (N)", 
        min_value=4, 
        value=st.session_state.current_N, 
        step=2, 
        key="N_total_slider_v3", # Incremented key
        help="Total participants, assumed to be split equally into two groups."
    )
    st.session_state.current_N = N_total_input

with col2:
    cohens_d_input = st.number_input(
        "Expected Cohen's d (Effect Size)", 
        min_value=0.0, 
        value=st.session_state.current_d, 
        step=0.01, 
        format="%.2f",
        key="cohens_d_slider_v3", # Incremented key
        help="Standardized mean difference. Common values: 0.2 (small), 0.5 (medium), 0.8 (large)."
    )
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

st.markdown("---")
st.caption("Remember: This tool is for educational and exploratory purposes. Always consult with a qualified statistician for actual clinical trial design and sample size calculations.")

