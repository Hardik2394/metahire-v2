# experience_calculator.py

from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging


def calculate_experience(work_experience):
    total_years = 0  # Track cumulative experience in years with decimal
    modified_work_experience = []

    for job in work_experience:
        employment_dates = job.get("employment_dates", "")
        
        # Assume employment_dates format like "Jan 2015 - Dec 2020" or "Jan 2015 - Present"
        try:
            start_date_str, end_date_str = employment_dates.split(" - ")
            start_date = datetime.strptime(start_date_str, "%B %Y")  # Full month name
            end_date = datetime.now() if end_date_str.lower() in ["present", "current"] else datetime.strptime(end_date_str, "%B %Y")

            # Calculate the difference for this job and convert to years with decimal
            job_experience = relativedelta(end_date, start_date)
            experience_years = job_experience.years + (job_experience.months / 12)
            
            # Update the cumulative total years
            total_years += experience_years

            # Update the job entry with calculated years in decimal
            job["experience_years"] = round(experience_years, 1)  # Rounded to one decimal place

            # Add modified job experience to the list
            modified_work_experience.append(job)

        except ValueError:
            # Handle any parsing errors (e.g., unexpected date format)
            logging.warning(f"Could not parse dates for job: {employment_dates}")
            continue

    # Round total years to one decimal place for final output
    total_years = round(total_years, 1)

    return total_years, modified_work_experience