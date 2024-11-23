from fastapi import FastAPI, Depends, HTTPException, Header
import openai
import json
import requests
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# Define the data model for incoming queries
class QueryModel(BaseModel):
    query: str

# Function to fetch JSON keys dynamically from Elasticsearch
def fetch_json_structure(elastic_url: str) -> dict:
    try:
        # Query Elasticsearch to retrieve all unique keys
        response = requests.post(
            f"{elastic_url}/_search",
            headers={"Content-Type": "application/json"},
            json={"size": 0, "aggs": {"keys": {"terms": {"field": "_source"}}}}
        )
        response.raise_for_status()
        keys = response.json()
        # Assuming keys are extracted here (customize if needed based on ES response format)
        json_structure = keys  # Replace with correct extraction logic
        return json_structure
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching JSON structure: {str(e)}")

# Function to parse natural language queries using GPT
def parse_natural_query(natural_query: str, json_structure: dict, openai_api_key: str) -> dict:
    try:
        openai.api_key = openai_api_key
        # Construct the prompt dynamically
        prompt = (
            "You are an assistant that parses natural language queries into structured JSON based on a specific format. "
            "Extract parameters from the user's query to match this structure exactly, preserving all nesting. "
            "Separate any overall experience requirement (total_experience) from job-level requirements within 'work_experience'. "
            "Your response **must** be **only** the JSON object, without any additional text or explanations."
            "Match the exact JSON structure, including nested fields, as described here:\n\n"
            f"{json.dumps(json_structure, indent=2)}\n\n"
            "Extract parameters from the user's query to match this structure exactly, preserving all nesting. "
            "Your response **must** be **only** the JSON object, without any additional text or explanations."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract parameters from this query and return a valid JSON with the exact nested structure. Query: {natural_query}"}
            ],
            temperature=0,
            max_tokens=500
        )
        parsed_query_str = response['choices'][0]['message']['content'].strip()
        parsed_query_str = parsed_query_str.replace("'", '"')  # Ensure valid JSON
        parsed_json = json.loads(parsed_query_str)
        return parsed_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing natural query: {str(e)}")

# Function to construct Elasticsearch query
def generate_elasticsearch_query(parsed_query: dict) -> dict:
    try:
        # Example query generation logic (customize as needed)
        elastic_query = {"query": {"bool": {"must": []}}}

        # Example: Add filters based on parsed query
        if "skills" in parsed_query:
            for skill in parsed_query["skills"].get("technical_skills", []):
                elastic_query["query"]["bool"]["must"].append({"match": {"skills.technical_skills": skill}})
            for skill in parsed_query["skills"].get("soft_skills", []):
                elastic_query["query"]["bool"]["must"].append({"match": {"skills.soft_skills": skill}})

        if "total_experience" in parsed_query:
            elastic_query["query"]["bool"]["must"].append(
                {"range": {"total_experience": {"gte": parsed_query["total_experience"]}}}
            )

        return elastic_query
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Elasticsearch query: {str(e)}")

@app.post("/process-query/")
def process_query(
    query: QueryModel,
    elastic_url: str = Header(...),
    openai_api_key: str = Header(...)
):
    # Log query received time
    query_sent = datetime.now(timezone.utc)

    # Fetch dynamic JSON structure from Elasticsearch
    json_structure = fetch_json_structure(elastic_url)

    # Parse the natural query using GPT
    parsed_query = parse_natural_query(query.query, json_structure, openai_api_key)

    # Generate Elasticsearch query from parsed query
    elastic_query = generate_elasticsearch_query(parsed_query)

    # Return the results
    return {
        "parsed_query": parsed_query,
        "elastic_query": elastic_query,
        "query_sent": query_sent.isoformat()
    }
