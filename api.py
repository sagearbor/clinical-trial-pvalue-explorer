# api.py (New Refactored Structure)
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Import LLM SDKs
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

load_dotenv()
app = FastAPI()

# --- Pydantic Models (IdeaInput, EstimationOutput) remain the same ---
class IdeaInput(BaseModel):
    text_idea: str | None = None
    url_idea: str | None = None

class EstimationOutput(BaseModel):
    initial_N: int | None = None
    initial_cohens_d: float | None = None
    estimation_justification: str | None = None
    processed_idea: str | None = None
    llm_provider_used: str | None = None # Add this to know which LLM responded
    error: str | None = None

# --- URL Fetching (fetch_url_content) remains the same ---
def fetch_url_content(url: str) -> str:
    try:
        # ... (implementation from previous version) ...
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'body', 'li', 'span'])
        text_content = "\n".join([para.get_text(separator=" ", strip=True) for para in paragraphs])
        return text_content[:5000]
    except Exception as e:
        return f"Error fetching/parsing URL: {str(e)}"


# --- LLM Interaction Logic ---
ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "GEMINI").upper()

# --- System Prompt for JSON Output ---
# This prompt needs to be robust for all models to encourage JSON output.
SYSTEM_PROMPT_FOR_JSON = """You are a biostatistical assistant. A researcher is planning a clinical trial.
Assume this will be a two-arm study (e.g., treatment vs. control) with equal participants per arm (N_total / 2 per arm), and the primary outcome is continuous. The goal is to detect a difference between the groups.
Based on the research idea provided by the user, provide plausible estimates for:
1.  Total number of participants (N_total).
2.  A typical Cohen's d effect size.
3.  A brief justification for your N_total and Cohen's d estimates.

Consider that researchers often aim for 80% power at a significance level (alpha) of 0.05 (two-sided).
Your response MUST be a valid JSON object with three keys: "N_total" (an integer), "cohens_d" (a float), and "justification" (a string).
Example:
{
  "N_total": 128,
  "cohens_d": 0.5,
  "justification": "Medium effect size is common for this type of behavioral intervention, and N is estimated for 80% power."
}
Provide only the JSON object. Do not include any other text, greetings, or explanations outside of the JSON structure.
"""

# --- Provider-Specific Functions ---

async def get_llm_estimate_gemini(text_input: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") # Or your preferred model
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            # Define the schema for the expected JSON output
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
        system_instruction=SYSTEM_PROMPT_FOR_JSON # Gemini SDK uses system_instruction
    )
    # Note: For Gemini, the SYSTEM_PROMPT_FOR_JSON itself might not be needed directly in the user prompt
    # if the response_schema and system_instruction handle it well.
    # However, the user prompt still needs to contain the research idea.
    prompt_parts = [f"Research Idea: '{text_input[:3000]}'"]
    
    response = await model.generate_content_async(prompt_parts) # Use async version
    return json.loads(response.text) # response.text should be the JSON string

async def get_llm_estimate_openai(text_input: str, is_azure: bool = False) -> dict:
    if is_azure:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not all([api_key, endpoint, deployment_name]):
            raise ValueError("Azure OpenAI environment variables (API_KEY, ENDPOINT, DEPLOYMENT_NAME) not fully set.")
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        model_to_call = deployment_name
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        client = OpenAI(api_key=api_key)
        model_to_call = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_JSON},
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    
    # For newer OpenAI models/API versions, you can request JSON output directly
    # Check documentation for your specific model and API version.
    try:
        completion = client.chat.completions.create(
            model=model_to_call,
            messages=messages,
            response_format={"type": "json_object"}, # Enforces JSON output if supported
            temperature=0.2,
            max_tokens=400 # Adjust as needed
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e: # Catch specific OpenAI errors if needed
        # If response_format={"type": "json_object"} fails or is not supported,
        # you might need to remove it and parse JSON from a string more carefully,
        # or ensure the prompt is extremely clear.
        print(f"OpenAI API error: {e}")
        # Fallback if JSON mode is not supported or fails
        try:
            print("Attempting OpenAI call without explicit JSON mode...")
            completion = client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.2,
                max_tokens=400
            )
            response_content = completion.choices[0].message.content
            # Try to extract JSON from potentially messy string output
            # This is less reliable than json_object mode
            match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                raise ValueError(f"No JSON object found in OpenAI response: {response_content}")
        except Exception as fallback_e:
            print(f"OpenAI API fallback error: {fallback_e}")
            raise fallback_e


async def get_llm_estimate_anthropic(text_input: str) -> dict:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
    
    client = Anthropic(api_key=api_key)
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    # Anthropic's system prompt is a specific parameter
    # The user message forms the main part of the prompt.
    
    # Claude 3 models are good at following structured output instructions like JSON.
    # Pre-fill the assistant's turn to guide it towards JSON.
    messages = [
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]

    response = client.messages.create(
        model=model_name,
        system=SYSTEM_PROMPT_FOR_JSON, # System prompt for Claude
        max_tokens=1024, # Claude needs enough tokens for JSON + explanation
        messages=messages,
        temperature=0.2
    ).content[0].text # Get the text content from the first TextBlock

    # Anthropic might not have a strict "JSON mode" like OpenAI's latest,
    # so robust prompting and parsing are key.
    # The SYSTEM_PROMPT_FOR_JSON asks it to *only* provide the JSON.
    return json.loads(response)


# --- Main Endpoint ---
@app.post("/process_idea", response_model=EstimationOutput)
async def process_idea(item: IdeaInput):
    if not item.text_idea and not item.url_idea:
        raise HTTPException(status_code=400, detail="Either text_idea or url_idea must be provided.")

    combined_idea = item.text_idea if item.text_idea else ""
    if item.url_idea:
        url_content = fetch_url_content(item.url_idea)
        combined_idea += f"\n\nContent from URL ({item.url_idea}):\n{url_content}"
    
    if not combined_idea.strip():
        return EstimationOutput(error="Could not extract any content to process.")

    try:
        llm_response_data = {}
        provider_used = ACTIVE_LLM_PROVIDER

        if ACTIVE_LLM_PROVIDER == "GEMINI":
            llm_response_data = await get_llm_estimate_gemini(combined_idea)
        elif ACTIVE_LLM_PROVIDER == "OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=False)
        elif ACTIVE_LLM_PROVIDER == "AZURE_OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=True)
        elif ACTIVE_LLM_PROVIDER == "ANTHROPIC":
            llm_response_data = await get_llm_estimate_anthropic(combined_idea)
        else:
            return EstimationOutput(error=f"Unsupported LLM provider: {ACTIVE_LLM_PROVIDER}", llm_provider_used=ACTIVE_LLM_PROVIDER)

        # Validate and return
        initial_N = llm_response_data.get("N_total")
        initial_cohens_d = llm_response_data.get("cohens_d")
        justification = llm_response_data.get("justification")

        if isinstance(initial_N, int) and \
           isinstance(initial_cohens_d, (float, int)) and \
           isinstance(justification, str):
            return EstimationOutput(
                initial_N=initial_N,
                initial_cohens_d=float(initial_cohens_d),
                estimation_justification=justification,
                processed_idea=combined_idea,
                llm_provider_used=provider_used
            )
        else:
            return EstimationOutput(error="LLM returned data in unexpected format or missing fields.", processed_idea=combined_idea, llm_provider_used=provider_used)

    except ValueError as ve: # Catch config errors
        return EstimationOutput(error=str(ve), processed_idea=combined_idea, llm_provider_used=ACTIVE_LLM_PROVIDER)
    except json.JSONDecodeError as je:
        return EstimationOutput(error=f"Failed to parse JSON response from LLM: {str(je)}", processed_idea=combined_idea, llm_provider_used=ACTIVE_LLM_PROVIDER)
    except Exception as e:
        # Log the full error for debugging
        print(f"An unexpected error occurred with {ACTIVE_LLM_PROVIDER}: {e}")
        import traceback
        traceback.print_exc()
        return EstimationOutput(error=f"An unexpected error occurred with {ACTIVE_LLM_PROVIDER}: {str(e)}", processed_idea=combined_idea, llm_provider_used=ACTIVE_LLM_PROVIDER)

# ... (if __name__ == "__main__": uvicorn.run(...) )