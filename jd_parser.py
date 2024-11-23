import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Form
from fastapi.responses import JSONResponse
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Description Parser API",
    description="API to parse job descriptions and return parsed data",
    version="1.1.0"
)

# Function to parse job descriptions using GPT
def extract_dynamic_requirements_from_jd(job_description_text: str, openai_api_key: str) -> dict:
    prompt = f"""
    Analyze the following job description and dynamically identify the main categories and their corresponding requirements.
    Structure the output in JSON format with categories based on the content of the job description.
    Each category should be accompanied by relevant requirements under it.

    Job Description:
    {job_description_text}
    """
    try:
        logger.info("Sending request to OpenAI API for JD parsing...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts dynamic categories and requirements from job descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
            api_key=openai_api_key
        )

        content = response.choices[0].message.content.strip()
        logger.debug(f"Raw GPT response for JD parsing: {content}")
        
        extracted_data = json.loads(content)
        return extracted_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error during JD parsing: {e}")
        raise HTTPException(status_code=500, detail="Error parsing JSON response from OpenAI.")
    except Exception as e:
        logger.error(f"Unexpected error during JD parsing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during JD parsing: {str(e)}")

# API Endpoint to parse JD
@app.post("/parse_jd/")
async def parse_jd(
    job_description: str = Form(...),
    job_id: str = Header(...),
    authorization: str = Header(...)
):
    try:
        # Parse the job description using GPT
        parsed_data = extract_dynamic_requirements_from_jd(job_description, authorization)
        
        # Return the parsed data along with the job_id
        return JSONResponse(content={
            "message": "Job description parsed successfully.",
            "job_id": job_id,
            "parsed_data": parsed_data
        })
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error in parse_jd: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
