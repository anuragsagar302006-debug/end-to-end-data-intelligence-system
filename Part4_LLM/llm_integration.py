import json
from openai import OpenAI

# 1. Initialize OpenAI Client pointing to Local Ollama Server
# Ollama doesn't require a real API key, so we pass a dummy string "ollama"
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

# 2. Extract Key Model Diagnostics & EDA Summaries from Parts 1-3
model_summary_data = {
    "dataset": "Telco Customer Churn",
    "total_records": 7043,
    "churn_rate": "26.5%",
    "best_model": "Tuned Random Forest Ensemble Classifier",
    "metrics": {
        "accuracy": 0.805,
        "roc_auc": 0.852,
        "f1_score": 0.598
    },
    "top_predictive_features": ["tenure", "MonthlyCharges", "Contract_Month-to-month", "InternetService_Fiber_optic"]
}

# 3. Create Prompt Asking for Structured JSON Response
prompt = f"""
You are an expert Data Science & AI Analyst. Analyze the following Machine Learning model metrics and produce a structured analytical summary.

Model Diagnostics Data:
{json.dumps(model_summary_data, indent=2)}

Return your output EXCLUSIVELY in valid JSON format with the following top-level keys:
- "executive_summary": A concise business summary of the model performance.
- "key_churn_drivers": A list of top features driving customer churn.
- "model_diagnostics": Insights into model strengths and potential limitations.
- "recommended_actions": 3 strategic recommendations for business stakeholders to reduce churn.
"""

# 4. Invoke Local Ollama API (using llama3)
response = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "system", "content": "You are a helpful data intelligence assistant that responds strictly in valid JSON format."},
        {"role": "user", "content": prompt}
    ]
)

# 5. Extract and Output JSON
json_output = response.choices[0].message.content
print("--- Generated Structured JSON Output from Ollama ---")
print(json_output)

# Save output to a JSON file
with open("llm_insights.json", "w") as f:
    # Clean up markdown code blocks if the model wrapped the JSON in them
    cleaned_output = json_output.replace("```json", "").replace("```", "").strip()
    parsed_json = json.loads(cleaned_output)
    json.dump(parsed_json, f, indent=4)

print("\nSaved LLM analysis to Part4_LLM/llm_insights.json!")