from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
import openai_key
import pandas as pd
import numpy as np
import io
from io import StringIO
import math
import json
import asyncio
import logging
import re

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cochran's Sample Size Formula
# We apply dataset sampling in order to limit token consumption and improve response time
# We also apply a hardcoded limit of 300 samples in case the formula returns a higher value
def cochran_sample_size(N, confidence=1.96, margin_error=0.05, p=0.5, capped_size=300):
    n0 = (confidence**2 * p * (1 - p)) / (margin_error**2)
    n = n0 / (1 + (n0 - 1) / N) if N < n0 else n0
    return min(math.ceil(n), N, capped_size)

async def load_file(file: UploadFile) -> pd.DataFrame:
    # Check if the file is CSV or JSON based on the content type
    if file.content_type == "text/csv":
        # If it's a CSV file
        contents = await file.read()
        data_str = StringIO(contents.decode("utf-8", errors="ignore"))
        return pd.read_csv(data_str)
    
    elif file.content_type == "application/json":
        # If it's a JSON file
        contents = await file.read()
        data_str = StringIO(contents.decode("utf-8", errors="ignore"))
        return pd.read_json(data_str)  # Using pd.read_json to load the JSON directly into a DataFrame
    
    else:
        # Raise an error if the file is neither CSV nor JSON
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and JSON files are allowed.")

# Function to get cleaned dataset from OpenAI
async def get_treated_dataset(df_sample):
    # Preparing dataset into a simple to process format to send to OpenAI
    df_json = df_sample.to_json(orient="records", lines=False)
    df_ai = json.loads(df_json)  # List of records
    # Creating Open AI request
    client = OpenAI(api_key=openai_key.api_key)
    response = await asyncio.to_thread(client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Your job is to take an input dataset and return it treated as a JSON-formatted list of records. Focus on handling missing values, encoding categorical variables, and ensuring data integrity."},
            {"role": "user", "content": f"Here is my dataset:\n{df_ai}\n"
                                        "Apply cleaning steps and return the treated dataset in JSON format. Must return only the treated dataset in JSON format, nothing else!"},
        ]
    )

    if not response.choices:
        logger.error("OpenAI API returned an empty response")
        return "OpenAI API returned an empty response. Ensure Input fits requirements."
    else:
        return response.choices[0].message.content  # Should contain JSON-formatted string

# Function to generate a preprocessing summary
async def get_preprocessing_summary(df_sample):
    # Preparing a dataset summary to send to OpenAI
    dataset_summary = {
        "columns": list(df_sample.columns),
        "column_types":  df_sample.dtypes.to_dict(),
        "sample_data": df_sample.to_dict(orient="records")
    }
    # Creating Open AI request
    client = OpenAI(api_key=openai_key.api_key)
    response = await asyncio.to_thread(client.chat.completions.create,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Your job is to analyze the dataset and provide a step-by-step preprocessing guide."},
            {"role": "user", "content": f"Here is my dataset summary:\n{dataset_summary}\n"
                                        "Provide a tailored list of preprocessing steps you would recommend for each column of the dataset (must be specific).\n"
                                        "This list must be specific and tailored to the dataset in question. In short, all recommendations must be based on the dataset summary."}
        ]
    )

    if not response.choices:
        logger.error("OpenAI API returned an empty response")
        return "OpenAI API returned an empty response. Ensure Input fits requirements."
    else:
        return response.choices[0].message.content

@app.post("/treated_sample/")
async def treated_sample(file: UploadFile = File(...)):
    df = await load_file(file)
    
    # Sample dataset using Cochran’s formula
    sample_size = cochran_sample_size(len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Get treated dataset from OpenAI
    raw_response = await get_treated_dataset(df_sample)

    match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if match:
        json_part = match.group(0)
        cleaned_json = json.loads(json_part)
    else:
        cleaned_json = None

    # getting final JSON output
    if cleaned_json is None:
        return {"error": "Failed to get treated dataset from OpenAI"}
    else:
        return StreamingResponse(io.StringIO(json.dumps(cleaned_json)), media_type="text/plain")


@app.post("/treatment_summary/")
async def treatment_summary(file: UploadFile = File(...)):
    df = await load_file(file)

    # Limit sample size to 180 for token efficiency
    sample_size = min(cochran_sample_size(len(df)), 180)
    df_sample = df.sample(n=sample_size, random_state=42)

    # Get preprocessing summary from OpenAI
    summary = await get_preprocessing_summary(df_sample)

    if summary is None:
        return {"error": "Failed to get preprocessing summary from OpenAI"}

    return StreamingResponse(io.StringIO(summary), media_type="text/plain")
