# api.py
import os
import json
import httpx # For making HTTP requests to Gemini API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests # For fetching URL content
from bs4 import BeautifulSoup # For parsing HTML content
from dotenv import load_dotenv

# Load environment variables from .env file (for GEMINI_API_KEY)
load_dotenv()

app = FastAPI()

class IdeaInput(BaseModel):
    text_idea: str | None = None
    url_idea: str | None = None

class EstimationOutput(BaseModel):
    initial_N: int | None = None
    initial_cohens_d: float | None = None
    estimation_justification: str | None = None # Added field
    processed_idea: str | None = None
    error: str | None = None

def fetch_url_content(url: str) -> str:
    """Fetches and extracts text content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from common tags, could be improved
        paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'article', 'body', 'li', 'span'])
        text_content = "\n".join([para.get_text(separator=" ", strip=True) for para in paragraphs])
        return text_content[:5000] # Limit content length
    except requests.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return f"Error fetching content from URL: {e}"
    except Exception as e:
        print(f"Error parsing URL content {url}: {e}")
        return f"Error parsing content from URL: {e}"

@app.post("/process_idea", response_model=EstimationOutput)
async def process_idea(item: IdeaInput):
    """
    Processes a research idea (text or URL) and uses Gemini to estimate
    initial N, Cohen's d, and provide a justification.
    """
    if not item.text_idea and not item.url_idea:
        raise HTTPException(status_code=400, detail="Either text_idea or url_idea must be provided.")

    combined_idea = item.text_idea if item.text_idea else ""
    if item.url_idea:
        url_content = fetch_url_content(item.url_idea)
        if "Error fetching content" in url_content or "Error parsing content" in url_content:
             # Append error to idea so user and LLM are aware
            combined_idea += f"\n\nNote: Could not fully process URL ({item.url_idea}): {url_content}"
        else:
            combined_idea += f"\n\nContent from URL ({item.url_idea}):\n{url_content}"
    
    if not combined_idea.strip():
        return EstimationOutput(error="Could not extract any content to process.")

    api_key = os.getenv("API_KEY")
    # In a managed environment, Canvas might inject the key. For local dev, it's from .env
    # No explicit check for empty string here, as the API call will fail if it's truly missing or invalid.
    # The "" is a placeholder for Canvas to inject if needed.
    # If running locally, API_KEY must be in .env
    if not api_key and not (os.getenv("IS_CANVAS_ENV") == "true"): # Simple check if in Canvas like env
         print("API_KEY not found in environment variables for local execution.")
         return EstimationOutput(error="API_KEY not configured on the server for local execution.", processed_idea=combined_idea)


    prompt = f"""You are a biostatistical assistant. A researcher is planning a clinical trial.
Research Idea: '{combined_idea[:3000]}' (Content might be truncated if very long)

Assume this will be a two-arm study (e.g., treatment vs. control) with equal participants per arm (N_total / 2 per arm), and the primary outcome is continuous. The goal is to detect a difference between the groups.
Based on this idea, provide plausible estimates for:
1.  Total number of participants (N_total) that might be needed.
2.  A typical Cohen's d effect size that might be targeted or observed in such a study.
3.  A brief justification for your N_total and Cohen's d estimates. This could include reasoning based on common practices for the type of intervention or outcome described, or typical effect sizes seen in related research areas. Avoid specific citations unless directly available and highly relevant from the provided text. Focus on general rationale.

Consider that researchers often aim for 80% power at a significance level (alpha) of 0.05 (two-sided).
Your response should be a JSON object with three keys: 'N_total' (an integer, e.g., 128), 'cohens_d' (a float, e.g., 0.5), and 'justification' (a string, e.g., "Medium effect size is common for this type of behavioral intervention, and N is estimated for 80% power.").
Provide only the JSON object. Do not include any other text or explanation."""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "N_total": {"type": "INTEGER", "description": "Estimated total number of participants."},
                    "cohens_d": {"type": "NUMBER", "description": "Estimated Cohen's d effect size."},
                    "justification": {"type": "STRING", "description": "Brief justification for the N and Cohen's d estimates."}
                },
                "required": ["N_total", "cohens_d", "justification"]
            }
        }
    }
    
    # Use the placeholder for API key which will be replaced by Canvas in managed env.
    # For local, it relies on the .env loaded key.
    actual_api_key = api_key if api_key else "" 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={actual_api_key}"

    try:
        async with httpx.AsyncClient(timeout=40.0) as client: # Increased timeout slightly
            response = await client.post(api_url, json=payload, headers={'Content-Type': 'application/json'})
            response.raise_for_status() # Raise HTTP errors
            
        result = response.json()

        if (result.get("candidates") and 
            result["candidates"][0].get("content") and
            result["candidates"][0]["content"].get("parts") and
            result["candidates"][0]["content"]["parts"][0].get("text")):
            
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(json_text)
            
            initial_N = parsed_json.get("N_total")
            initial_cohens_d = parsed_json.get("cohens_d")
            justification = parsed_json.get("justification")

            if isinstance(initial_N, int) and \
               isinstance(initial_cohens_d, (float, int)) and \
               isinstance(justification, str): # Ensure justification is a string
                return EstimationOutput(
                    initial_N=initial_N, 
                    initial_cohens_d=float(initial_cohens_d), # Ensure float
                    estimation_justification=justification,
                    processed_idea=combined_idea
                )
            else:
                error_msg = "LLM returned data in unexpected format or missing fields."
                print(f"{error_msg} Received: N={initial_N}, d={initial_cohens_d}, justification={justification}")
                return EstimationOutput(error=error_msg, processed_idea=combined_idea)
        else:
            error_msg = "Unexpected response structure from LLM."
            # Check for safety ratings or blocks
            if result.get("promptFeedback") and result["promptFeedback"].get("blockReason"):
                 error_msg = f"LLM request blocked: {result['promptFeedback']['blockReason']}. This might be due to the content of the research idea or URL."
            elif not result.get("candidates"):
                 error_msg = "LLM returned no candidates. The prompt might have been filtered due to safety settings or other reasons."
            print(f"{error_msg} Full response: {result}")
            return EstimationOutput(error=error_msg, processed_idea=combined_idea)

    except httpx.HTTPStatusError as e:
        error_detail = "No details available."
        try:
            error_detail = e.response.json().get("error", {}).get("message", e.response.text)
        except: # Fallback if response is not JSON or structure is different
            error_detail = e.response.text
        print(f"HTTP error calling Gemini API: {e}. Response: {error_detail}")
        return EstimationOutput(error=f"Error calling LLM service: {e.response.status_code} - {error_detail}", processed_idea=combined_idea)
    except httpx.RequestError as e:
        print(f"Request error calling Gemini API: {e}")
        return EstimationOutput(error=f"Network error calling LLM service: {e}", processed_idea=combined_idea)
    except json.JSONDecodeError as e:
        print(f"JSON decode error from LLM response: {e}. Response text: {json_text if 'json_text' in locals() else 'N/A'}")
        return EstimationOutput(error="Error parsing LLM response (invalid JSON).", processed_idea=combined_idea)
    except Exception as e:
        print(f"An unexpected error occurred in process_idea: {e}")
        return EstimationOutput(error=f"An unexpected server error occurred: {str(e)}", processed_idea=combined_idea)

if __name__ == "__main__":
    import uvicorn
    # This is for local testing of the API if needed, but usually run via `uvicorn api:app --reload`
    # Ensure .env file with API_KEY is present in the same directory for local run.
    print("Attempting to run FastAPI server locally on port 8000...")
    print("Ensure your .env file with API_KEY is in the same directory as api.py if running locally and not in Canvas.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
