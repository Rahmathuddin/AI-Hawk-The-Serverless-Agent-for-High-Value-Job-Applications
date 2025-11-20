import os
import json
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from google.cloud import firestore

# --- Initialization ---
app = Flask(__name__)
# The Gemini client automatically handles authentication on Cloud Run 
# using the service account's permissions (if Vertex AI User role is granted).
try:
    gemini_client = genai.Client()
    db = firestore.Client()
except Exception as e:
    print(f"Failed to initialize clients: {e}")
    # Continue initialization; health check will fail if critical.

# Load the structured schema
with open("data/schema/score_schema.json", "r") as f:
    SCORE_SCHEMA = json.load(f)

# Load master candidate data for RAG context
# In a real app, this would be retrieved from Firestore
with open("data/master_profile.json", "r") as f:
    MASTER_PROFILE = f.read()

SYSTEM_INSTRUCTION = """
You are the AI Hawk Relevance Scorer. Your task is to analyze a Job Description (JD) and a candidate profile.
Score the job based on the following weighted criteria: Cloud Run Usage (20%), GCP Database Usage (20%),
Google AI Usage (20%), Functional Demo (15%), Blog Excellence (15%), Industry Impact (10%).
You MUST return the output as a JSON object matching the provided schema.
"""

# --- Core API Route ---
@app.route("/", methods=["POST"])
def analyze_job_posting():
    """
    Receives Job Data, calls Gemini to score and customize the application.
    """
    try:
        job_data = request.get_json()
        job_id = job_data.get("job_id")
        job_description = job_data.get("job_description")

        if not job_id or not job_description:
            return jsonify({"status": "error", "message": "Missing job_id or job_description"}), 400

        # --- Gemini Function Call for Structured Scoring (Phase 3) ---
        prompt = f"""
        Candidate Profile (Context): {MASTER_PROFILE}

        Job Description to Score: {job_description}

        Perform the scoring and identify missing keywords based on the six criteria.
        """
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=SCORE_SCHEMA
            )
        )
        
        # Parse the structured JSON response
        score_result = json.loads(response.text)
        weighted_score = score_result.get("weighted_score", 0)

        # --- Conditional Logic (The Application Gate) ---
        if weighted_score < 80:
            db.collection("application_log").document(job_id).set({
                "status": "REJECTED",
                "score": weighted_score,
                "reason": "Score below 80% threshold",
                "timestamp": firestore.SERVER_TIMESTAMP
            })
            return jsonify({"status": "rejected", "score": weighted_score}), 200

        # --- Customization (Simplified for this example) ---
        # In a real app, you'd make a second Gemini call here to generate the cover letter
        # using a prompt like: "Rewrite the resume with {score_result.missing_keywords}..."
        
        # Log successful vetting
        db.collection("application_log").document(job_id).set({
            "status": "READY_FOR_SUBMISSION",
            "score": weighted_score,
            "gemini_details": score_result,
            "job_data": job_data,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        # --- Dispatch Submission Task (In a real app, this would call a Cloud Run JOB) ---
        # print("Dispatching submission job...") 
        
        return jsonify({"status": "success", "score": weighted_score, "details": score_result}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Health Check and Run ---
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
