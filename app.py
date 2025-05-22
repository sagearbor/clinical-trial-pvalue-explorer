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

    # For a two-sample t-test, each group must have at least 2 participants for variance calculation,
    # so n_per_group > 1. This means N_total / 2 > 1, so N_total > 2.
    # Degrees of freedom df = (n1 - 1) + (n2 - 1) = n1 + n2 - 2 = N_total - 2.
    # df must be > 0.
    
    if N_total_int <= 2: # Stricter check, ensuring df > 0
        return None, "Total N must be greater than 2 for valid degrees of freedom."

    n_per_group = N_total_int / 2.0
    if n_per_group <= 1: # Each group must have more than 1 participant
         return None, "Sample size per group (N_total / 2) must be greater than 1."


    # If Cohen's d is 0, t-statistic is 0, p-value is 1.
    if cohens_d == 0:
        return 1.0, "With Cohen's d = 0, the t-statistic is 0, leading to a p-value of 1.0 (no effect observed)."

    # Calculate the t-statistic from Cohen's d for two independent groups of equal size n_per_group
    # t = d * sqrt(n_per_group / 2)
    #   where n_per_group = N_total / 2
    #   So, t = d * sqrt((N_total / 2) / 2) = d * sqrt(N_total / 4) = d * (sqrt(N_total) / 2)
    try:
        # Ensure n_per_group / 2 is not zero or negative, though previous checks should cover this.
        if (n_per_group / 2.0) <= 0:
            return None, "Invalid internal calculation for t-statistic (sqrt of non-positive)."
        t_statistic = cohens_d * np.sqrt(n_per_group / 2.0) 
    except FloatingPointError: # Handles potential issues with very small numbers if any
        return None, "Numerical instability in t-statistic calculation."


    df = N_total_int - 2 # Degrees of freedom
    if df <= 0: # Should be caught by N_total_int > 2
        return None, "Degrees of freedom (N_total - 2) must be positive."

    try:
        p_value = (1 - stats.t.cdf(abs(t_statistic), df)) * 2 # Two-sided p-value
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

# Use values from session state for inputs to ensure they persist after AI call
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
        """
        The p-value is calculated for a **two-sample, two-sided t-test** with equal group sizes.
        
        1.  **Sample size per group ($n_{group}$):**
            $n_{group} = N_{total} / 2$
        
        2.  **Cohen's d (Effect Size):**
            This is your input. It represents the standardized difference between two means:
            $d = (M_1 - M_2) / \sigma_{pooled}$
            
        3.  **t-statistic (from Cohen's d):**
            For two independent groups of equal size $n_{group}$:
            $t = d \cdot \sqrt{n_{group} / 2}$
            Substituting $n_{group} = N_{total} / 2$:
            $t = d \cdot \sqrt{(N_{total} / 2) / 2} = d \cdot \sqrt{N_{total} / 4} = d \cdot \frac{\sqrt{N_{total}}}{2}$

        4.  **Degrees of Freedom (df):**
            $df = N_{total} - 2$
            
        5.  **P-value:**
            Calculated from the t-distribution using the absolute $t$-statistic and $df$.
        """
    )


st.header("1. Describe Your Research Idea")
text_idea = st.text_area("Enter your clinical trial research idea here:", height=150, key="text_idea_input_v2")
url_idea = st.text_input("Or provide a URL with relevant content (optional):", key="url_idea_input_v2")

if st.button("Get AI Estimate for N, Cohen's d, and Justification", key="get_estimate_button_v2"):
    if not text_idea and not url_idea:
        st.error("Please provide either a research idea text or a URL.")
    else:
        payload = {"text_idea": text_idea, "url_idea": url_idea}
        try:
            # Reset to defaults before new call, or keep current user settings?
            # For now, let's update session state with AI values if successful.
            
            with st.spinner("Processing your idea with AI... this may take a moment."):
                response = requests.post(BACKEND_URL, json=payload, timeout=60) # Increased timeout
            
            if response.status_code == 200:
                data = response.json()
                if data.get("error"):
                    st.error(f"Error from backend: {data['error']}")
                    st.session_state.processed_idea_text = data.get("processed_idea", st.session_state.processed_idea_text) # Keep old if new is empty
                    st.session_state.estimation_justification = "" # Clear justification on error
                elif data.get("initial_N") is not None and data.get("initial_cohens_d") is not None:
                    st.session_state.initial_N = data["initial_N"]
                    st.session_state.initial_cohens_d = data["initial_cohens_d"]
                    # Update current sliders to AI estimates
                    st.session_state.current_N = data["initial_N"]
                    st.session_state.current_d = data["initial_cohens_d"]

                    st.session_state.estimation_justification = data.get("estimation_justification", "No justification provided by AI.")
                    st.session_state.processed_idea_text = data.get("processed_idea", "Idea processed successfully.")
                    st.success("AI estimation received!")
                else:
                    st.error("AI estimation received, but data is incomplete. Using previous or default values.")
                    st.session_state.processed_idea_text = data.get("processed_idea", st.session_state.processed_idea_text)
                    st.session_state.estimation_justification = ""


            else:
                st.error(f"Failed to get estimation from backend. Status code: {response.status_code}")
                try:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Details: {error_detail}")
                except: # Fallback if response is not JSON
                    st.error(f"Details: {response.text}")
                st.session_state.processed_idea_text = "Failed to process idea."
                st.session_state.estimation_justification = ""
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API. Is it running at " + BACKEND_URL + "?")
        except requests.exceptions.Timeout:
            st.error("The request to the backend API timed out. The AI might be taking too long or the server is busy.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.session_state.estimation_justification = "" # Clear on other errors too

# This ensures that even if the button wasn't pressed, if there's an idea, it's shown
if st.session_state.processed_idea_text:
    with st.expander("View Processed Idea Sent to AI", expanded=False):
        st.caption("This is the text that was (or would be) sent to the AI model for estimation:")
        # Display only a portion if too long
        display_text = st.session_state.processed_idea_text
        if len(display_text) > 1500:
            display_text = display_text[:1500] + "..."
        st.markdown(f"```\n{display_text}\n```")

if st.session_state.estimation_justification:
     with st.expander("AI's Justification for N and Cohen's d Estimates", expanded=True):
        st.info(st.session_state.estimation_justification)


st.header("2. Adjust Parameters and See P-Value")
st.markdown(
    "Use the sliders or input boxes below to adjust the Total Number of Participants (N) "
    "and the Expected Cohen's d. The p-value will update automatically. "
)

col1, col2 = st.columns(2)

with col1:
    N_total_input = st.number_input(
        "Total Number of Participants (N)", 
        min_value=4, # Min N for df > 0 and n_per_group > 1 (N_total/2 > 1 => N_total > 2; df = N_total - 2 > 0 => N_total > 2. So 4 is safe min for n_per_group=2)
        value=st.session_state.current_N, 
        step=2, 
        key="N_total_slider_v2",
        help="Total participants, assumed to be split equally into two groups."
    )
    st.session_state.current_N = N_total_input


with col2:
    cohens_d_input = st.number_input(
        "Expected Cohen's d (Effect Size)", 
        min_value=0.0, 
        value=st.session_state.current_d, 
        step=0.01, # Finer step for Cohen's d
        format="%.2f",
        key="cohens_d_slider_v2",
        help="Standardized mean difference. Common values: 0.2 (small), 0.5 (medium), 0.8 (large)."
    )
    st.session_state.current_d = cohens_d_input


# Recalculate p-value based on current slider/input values
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

