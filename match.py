import json
import logging
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Matching API",
    description="API to match candidates' resumes with job descriptions using GPT-4, without database dependency.",
    version="3.1.0"
)

# Scoring weights
MATCH_SCORES = {
    "Full match": 1.0,
    "Partial match": 0.5,
    "No match": 0.0,
    "Error": 0.0  # In case of errors, score as zero
}

# Matching function
def match_item(job_item: str, resume_details: dict, openai_api_key: str) -> dict:
    """
    Matches a single job requirement against the resume details.
    """
    prompt = f"""
    Job Requirement: "{job_item}"
    Candidate Resume Details: {json.dumps(resume_details, indent=2)}

    Provide:
    - Match Level ("Full match", "Partial match", "No match")
    - Reason for the match.
    - Evidence from the resume.

    Output strictly in JSON format:
    {{
        "match_level": "",
        "reason": "",
        "evidence": ""
    }}
    """
    try:
        logger.info(f"Matching item: {job_item}")
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant analyzing job requirements against resumes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=750,
            temperature=0.5
        )
        output = response.choices[0].message.content.strip()
        logger.debug(f"GPT Response: {output}")
        if not output:
            raise ValueError("Empty response from OpenAI.")
        return json.loads(output)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error during matching: {e}")
        logger.error(f"Raw GPT Response: {output}")
        raise HTTPException(status_code=500, detail="Error parsing JSON response from OpenAI.")
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI API Error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API error.")
    except Exception as e:
        logger.error(f"Unexpected error during matching: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/match/")
async def unified_match_endpoint(
    request: Request,
    openai_api_key: str = Header(...)
):
    """
    Unified matching endpoint to compare job description and resume provided in the request.
    """
    try:
        # Parse request body
        body = await request.json()
        job_description_details = body.get("job_description", {}).get("parsed_data", {})
        resume_details = body.get("resume", {}).get("response", {})

        if not job_description_details or not resume_details:
            raise HTTPException(status_code=400, detail="Job description or resume data is missing.")

        logger.info("Received job description and resume details successfully.")

        # Initialize the output structure
        matching_results = {}
        category_scores = {}
        total_score = 0
        total_items = 0

        # Validate and process job description details
        if not isinstance(job_description_details, dict):
            logger.error(f"Invalid format for job description details: {type(job_description_details)}")
            raise HTTPException(status_code=500, detail="Job description details must be a dictionary.")

        for category, subcategories in job_description_details.items():
            if not isinstance(subcategories, dict):
                logger.error(f"Invalid format for subcategories: {type(subcategories)}")
                continue

            matching_results[category] = {}
            category_score = 0
            category_items = 0

            for subcategory, items in subcategories.items():
                if not isinstance(items, list):
                    logger.error(f"Invalid format for items in subcategory '{subcategory}': {type(items)}")
                    raise HTTPException(status_code=500, detail=f"Invalid items format in subcategory '{subcategory}'.")

                matching_results[category][subcategory] = {}
                for item in items:
                    try:
                        match_data = match_item(item, resume_details, openai_api_key)
                        matching_results[category][subcategory][item] = match_data

                        # Calculate score for this item
                        match_level = match_data.get("match_level", "No match")
                        item_score = MATCH_SCORES.get(match_level, 0)
                        category_score += item_score
                        category_items += 1
                        total_score += item_score
                        total_items += 1
                    except HTTPException as e:
                        logger.error(f"Error matching item '{item}': {e.detail}")
                        matching_results[category][subcategory][item] = {
                            "match_level": "Error",
                            "reason": f"Error during matching: {e.detail}",
                            "evidence": ""
                        }

            # Calculate category-level score
            if category_items > 0:
                category_scores[category] = category_score / category_items

        # Calculate overall score
        overall_score = total_score / total_items if total_items > 0 else 0

        return JSONResponse(content={
            "message": "Matching completed successfully.",
            "matching_results": matching_results,
            "category_scores": category_scores,
            "overall_score": overall_score
        })

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in unified_match_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")
