# resume_parser2.py

from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Header
from fastapi.responses import JSONResponse
import json
import logging
import openai
from experience_calculator import calculate_experience
from text_extractor import extract_text_from_pdf, extract_text_from_docx

app = FastAPI(debug=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_gpt_insights(resume_text, openai_api_key):
    """
    Generate insights from resume text using OpenAI API.
    """
    insights_prompt = f"""
    Analyze the following resume text and extract the details as JSON.

    Instructions:
    - Output must be valid JSON.
    - Do not include comments or additional text outside the JSON.
    - Use double quotes for all keys and string values.
    - Ensure numerical values are numbers, not strings.

    Expected JSON structure:
    {{
        "personal_information": {{
            "name": "",
            "contact_information": {{
                "phone": "",
                "email": "",
                "linkedin_profile": ""
            }},
            "photo": ""
        }},
        "professional_summary_or_objective": "",
        "work_experience": [
            {{
                "job_title": "",
                "company_name": "",
                "employment_dates": "",
                "job_description": "",
                "skills_applied": "",
                "achievements": "",
                "experience_years": ""
            }}
        ],
        "total_experience": "",
        "education": [
            {{
                "degree": "",
                "institution": "",
                "graduation_date": ""
            }}
        ],
        "skills": {{
            "technical_skills": [],
            "soft_skills": []
        }},
        "certifications": [],
        "languages": [],
        "projects": [],
        "publications": [],
        "awards": [],
        "volunteer_experience": [],
        "affiliations": [],
        "interests": [],
        "references": [],
        "portfolio_links": [],
        "social_media_profiles": {{
            "linkedin": "",
            "github": "",
            "behance": "",
            "dribble": "",
            "medium": "",
            "other": ""
        }},
        "location_preferences": "",
        "desired_job_title_and_industry": ""
    }}

    Resume Text:
    {resume_text}
    """
    try:
        logger.info("Sending request to OpenAI API...")
        insights_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": insights_prompt}
            ],
            max_tokens=2000,
            temperature=0.7,
            api_key=openai_api_key
        )

        content = insights_response.choices[0]['message']['content'].strip()
        logger.debug(f"Raw GPT response: {content}")

        # Ensure the content is valid JSON
        if not content.startswith('{') or not content.endswith('}'):
            logging.error("GPT response is not a valid JSON object.")
            raise HTTPException(status_code=500, detail="Invalid JSON response from OpenAI.")

        # Parse the JSON response
        gpt_insights = json.loads(content)

        # Calculate experience based on work experience
        work_experience = gpt_insights.get("work_experience", [])
        total_years, modified_work_experience = calculate_experience(work_experience)

        # Update GPT response with total experience as a single decimal value
        gpt_insights["total_experience"] = total_years
        gpt_insights["work_experience"] = modified_work_experience

        return gpt_insights

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.debug(f"Content that failed to parse: {content}")
        raise HTTPException(status_code=500, detail="Error parsing JSON response from OpenAI.")
    except Exception as e:
        logger.error(f"Unexpected error in generate_gpt_insights: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/upload_resume/")
async def upload_resume(
    file: UploadFile = File(...),
    openai_api_key: str = Header(...)
):
    """
    Endpoint to upload and process a resume file.
    """
    try:
        # Read the resume file and extract text
        if file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file.file)
        elif file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or DOCX file.")

        # Generate GPT insights
        gpt_insights = generate_gpt_insights(resume_text, openai_api_key)

        # Return the insights directly as JSON along with the GPT response
        return JSONResponse(content={
            "message": "Resume parsed successfully.",
            "gpt_response": gpt_insights
        })
    
    except Exception as e:
        logger.error(f"Error in upload_resume: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
