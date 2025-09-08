import requests
import json

# Test LLM processing
data = {
    "idea": "I want to test if a new drug reduces blood pressure more effectively than the standard treatment. We have 100 patients in each group.",
    "context": "Clinical trial for hypertension medication"
}

try:
    response = requests.post("http://localhost:8000/process_idea", json=data)
    result = response.json()
    print("LLM Analysis Result:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print(f"Error: {e}")
    print(f"Response: {response.text if 'response' in locals() else 'No response'}")
