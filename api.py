# api.py (New Refactored Structure with Debug Prints)
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests # Ensure this is imported if used by fetch_url_content
from bs4 import BeautifulSoup # Ensure this is imported if used by fetch_url_content
from dotenv import load_dotenv
import re # Added for the OpenAI fallback logic

# Import LLM SDKs
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

# --- CRITICAL: load_dotenv() must be called BEFORE accessing os.getenv for .env variables ---
load_dotenv(override=True)
print(f"--- DOTENV LOADED ---") # Confirmation that load_dotenv() was called

app = FastAPI()

# --- LLM Interaction Logic ---
# Let's see what os.getenv is actually returning
raw_active_llm_provider_from_env = os.getenv("ACTIVE_LLM_PROVIDER")
print(f"--- API STARTUP: Raw ACTIVE_LLM_PROVIDER from .env is: '{raw_active_llm_provider_from_env}' ---")

ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "GEMINI").upper()
print(f"--- API STARTUP: Effective ACTIVE_LLM_PROVIDER after default and .upper() is: '{ACTIVE_LLM_PROVIDER}' ---")


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
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'body', 'li', 'span'])
        text_content = "\n".join([para.get_text(separator=" ", strip=True) for para in paragraphs])
        return text_content[:5000]
    except Exception as e:
        return f"Error fetching/parsing URL: {str(e)}"


# --- System Prompt for JSON Output ---
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
    print("--- Attempting to use GEMINI provider ---")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here": # Added check for placeholder
        raise ValueError("GEMINI_API_KEY not found or is a placeholder in environment variables.")
    genai.configure(api_key=api_key)
    
    model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash-latest") 
    model = genai.GenerativeModel(
        model_name,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
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
        system_instruction=SYSTEM_PROMPT_FOR_JSON 
    )
    prompt_parts = [f"Research Idea: '{text_input[:3000]}'"]
    response = await model.generate_content_async(prompt_parts) 
    return json.loads(response.text) 

async def get_llm_estimate_openai(text_input: str, is_azure: bool = False) -> dict:
    if is_azure:
        print("--- Attempting to use AZURE_OPENAI provider ---")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01") # Defaulting to a known stable version
        if not all([api_key, endpoint, deployment_name]) or api_key == "REDACTED": # Added check for placeholder
            raise ValueError("Azure OpenAI environment variables (API_KEY, ENDPOINT, DEPLOYMENT_NAME) not fully set or API_KEY is placeholder.")
        client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        model_to_call = deployment_name
    else:
        print("--- Attempting to use OPENAI (direct) provider ---")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here": # Added check for placeholder
            raise ValueError("OPENAI_API_KEY not found or is a placeholder in environment variables.")
        client = OpenAI(api_key=api_key)
        model_to_call = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_JSON},
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    
    try:
        print(f"--- Calling OpenAI/Azure model: {model_to_call} ---")
        completion = client.chat.completions.create(
            model=model_to_call,
            messages=messages,
            response_format={"type": "json_object"}, 
            temperature=0.2,
            max_tokens=400 
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content)
    except Exception as e: 
        print(f"OpenAI API error with JSON mode (model: {model_to_call}): {e}")
        print("--- Attempting OpenAI/Azure call without explicit JSON mode (fallback) ---")
        try:
            completion = client.chat.completions.create(
                model=model_to_call,
                messages=messages,
                temperature=0.2,
                max_tokens=400
            )
            response_content = completion.choices[0].message.content
            match = re.search(r"\{.*\}", response_content, re.DOTALL) # Using re import
            if match:
                return json.loads(match.group(0))
            else:
                print(f"--- Fallback failed: No JSON object found in OpenAI response: {response_content[:500]}... ---")
                raise ValueError(f"No JSON object found in OpenAI response: {response_content}")
        except Exception as fallback_e:
            print(f"OpenAI API fallback error (model: {model_to_call}): {fallback_e}")
            raise fallback_e


async def get_llm_estimate_anthropic(text_input: str) -> dict:
    print("--- Attempting to use ANTHROPIC provider ---")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here": # Added check for placeholder
        raise ValueError("ANTHROPIC_API_KEY not found or is a placeholder in environment variables.")
    
    client = Anthropic(api_key=api_key)
    model_name = os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307")
    messages = [
        {"role": "user", "content": f"Research Idea: '{text_input[:3000]}'"}
    ]
    print(f"--- Calling Anthropic model: {model_name} ---")
    response_obj = client.messages.create( # Corrected variable name
        model=model_name,
        system=SYSTEM_PROMPT_FOR_JSON, 
        max_tokens=1024, 
        messages=messages,
        temperature=0.2
    )
    # Ensure content is a list and not empty, and first item is a TextBlock
    if response_obj.content and isinstance(response_obj.content, list) and hasattr(response_obj.content[0], 'text'):
        response_text = response_obj.content[0].text
        return json.loads(response_text)
    else:
        print(f"--- Anthropic unexpected response structure: {response_obj.content} ---")
        raise ValueError("Anthropic response content not in expected format.")


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

    # This uses the ACTIVE_LLM_PROVIDER defined at the module level
    provider_used_for_this_request = ACTIVE_LLM_PROVIDER 
    print(f"--- Processing request with provider: {provider_used_for_this_request} ---")

    try:
        llm_response_data = {}
        
        if provider_used_for_this_request == "GEMINI":
            llm_response_data = await get_llm_estimate_gemini(combined_idea)
        elif provider_used_for_this_request == "OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=False)
        elif provider_used_for_this_request == "AZURE_OPENAI":
            llm_response_data = await get_llm_estimate_openai(combined_idea, is_azure=True)
        elif provider_used_for_this_request == "ANTHROPIC":
            llm_response_data = await get_llm_estimate_anthropic(combined_idea)
        else:
            # This case should ideally not be hit if ACTIVE_LLM_PROVIDER is validated at startup,
            # but good as a safeguard.
            print(f"--- ERROR: Unsupported LLM provider configured: {provider_used_for_this_request} ---")
            return EstimationOutput(error=f"Unsupported LLM provider: {provider_used_for_this_request}", llm_provider_used=provider_used_for_this_request)

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
                llm_provider_used=provider_used_for_this_request
            )
        else:
            print(f"--- ERROR: LLM ({provider_used_for_this_request}) returned data in unexpected format: {llm_response_data} ---")
            return EstimationOutput(error="LLM returned data in unexpected format or missing fields.", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)

    except ValueError as ve: 
        print(f"--- ValueError during LLM call ({provider_used_for_this_request}): {ve} ---")
        return EstimationOutput(error=str(ve), processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except json.JSONDecodeError as je:
        print(f"--- JSONDecodeError from LLM ({provider_used_for_this_request}): {je} ---")
        return EstimationOutput(error=f"Failed to parse JSON response from LLM: {str(je)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)
    except Exception as e:
        print(f"--- An unexpected error occurred with {provider_used_for_this_request}: {e} ---")
        import traceback
        traceback.print_exc()
        return EstimationOutput(error=f"An unexpected error occurred with {provider_used_for_this_request}: {str(e)}", processed_idea=combined_idea, llm_provider_used=provider_used_for_this_request)

if __name__ == "__main__":
    # This block is for direct execution (python api.py), not when run by uvicorn
    # Uvicorn handles the app loading.
    print("--- Starting API directly (not via uvicorn, for testing only if needed) ---")
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
