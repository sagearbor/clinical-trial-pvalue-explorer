# Clinical Trial P-Value Explorer (AI-Assisted)

This application allows users to input a clinical trial research idea (or a URL with relevant content) and receive an AI-generated initial estimate for the total number of participants (N) and a plausible Cohen's d effect size. Users can then interactively adjust these parameters (N and Cohen's d) to see the corresponding p-value, calculated assuming a two-sample t-test with equal group sizes.

**Disclaimer:** The AI's estimates for N and Cohen's d are very rough approximations and intended for exploratory and educational purposes only. They should not be used for actual clinical trial planning without consultation with a qualified statistician. The primary utility of this tool is the interactive p-value calculator.

## Project Structure

.├── api.py              # FastAPI backend application├── app.py              # Streamlit frontend application├── requirements.txt    # Python dependencies├── .env                # For API Key (you need to create this)└── README.md           # This file
## Setup Instructions

1.  **Clone the Repository (or create files):**
    If this were a git repo, you'd clone it. For now, create the files `api.py`, `app.py`, and `requirements.txt` with the content provided.

2.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Gemini API Key:**
    * You need a Gemini API key to use the AI estimation feature. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create a file named `.env` in the root directory of the project (the same directory as `api.py`).
    * Add your API key to the `.env` file like this:
        ```
        API_KEY="YOUR_API_KEY_TO_ANY_LLM_HERE"
        ```
    * Replace `"YOUR_API_KEY_TO_ANY_LLM_HERE"` with your actual key.

## Running the Application

You need to run two components separately: the FastAPI backend and the Streamlit frontend.

1.  **Run the FastAPI Backend:**
    Open a terminal, navigate to the project directory, and run:
    ```bash
    uvicorn api:app --reload --port 8000
    ```
    This will start the backend server, typically at `http://localhost:8000`. Keep this terminal window open.

2.  **Run the Streamlit Frontend:**
    Open a *new* terminal window, navigate to the project directory, and run:
    ```bash
    streamlit run app.py
    ```
    This will open the Streamlit application in your web browser, usually at `http://localhost:8051`.

## How to Use

1.  Open the Streamlit application in your browser.
2.  Enter your research idea in the text box or provide a URL to relevant content.
3.  Click "Get AI Estimate for N and Cohen's d". The application will contact the FastAPI backend, which in turn queries the Gemini API.
    * The AI-estimated Total N and Cohen's d will populate the input fields.
    * A snippet of the text processed by the AI will be shown in an expander.
4.  Adjust the "Total Number of Participants (N)" and "Expected Cohen's d" using the input fields or sliders.
5.  The "Calculated P-Value" will update automatically based on your adjustments.
    * The tool will also indicate if the p-value is typically considered statistically significant (< 0.05).

## Important Considerations for the P-Value Calculation

* The p-value is calculated assuming a **two-sample, two-sided t-test** with **equal sample sizes** in each of the two groups (N_total / 2 per group).
* It answers the question: "If a study with `N_total` participants observed an effect size of `Cohen's d`, what would the p-value be?"
* This is **not** a power calculation (which determines the probability of detecting an effect of a certain size).
