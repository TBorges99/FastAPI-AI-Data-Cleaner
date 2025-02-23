from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from openai import OpenAI
import openai_key
import pandas as pd
import numpy as np
import io
from io import StringIO
import math
import json

app = FastAPI()

# Cochran's Sample Size Formula
def cochran_sample_size(N, confidence=1.96, margin_error=0.05, p=0.5, max_sample_size=5000):
    """Compute Cochran’s sample size with an optional max cap."""
    n0 = (confidence**2 * p * (1 - p)) / (margin_error**2)
    n = n0 / (1 + (n0 - 1) / N) if N < n0 else n0
    return min(math.ceil(n), N, max_sample_size)  # Ensure it doesn't exceed dataset size or max cap


@app.post("/treated_sample/")
async def treated_sample(file: UploadFile = File(...)):
    contents = await file.read()
    data_str = StringIO(contents.decode("utf-8", errors="ignore"))
    df = pd.read_csv(data_str)

    # Sample the dataset using Cochran’s approach
    N = len(df)
    sample_size = cochran_sample_size(N)
    df_sample = df.sample(n=sample_size, random_state=42)

    # Convert sampled DataFrame to JSON for OpenAI
    df_json = df_sample.to_json(orient="records", lines=False)

    # Converting JSON string to Python dict/list for better handling (not essential but has ran better so far)
    df_ai = json.loads(df_json)  

    client = OpenAI(api_key=openai_key.api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Your job is to take an input dataset and return it treated as a JSON-formatted list of records. Focus on handling missing values, encoding categorical variables, and ensuring data integrity."},
            {"role": "user", "content": f"Here is my dataset:\n{df_ai}\n"
                                        "Apply cleaning steps and return the treated dataset in JSON format."},
        ]
    )

    cleaned_json = response.choices[0].message.content  # Already in JSON format

    # Convert JSON response back to DataFrame and then to CSV
    df_cleaned = pd.DataFrame(json.loads(cleaned_json))
    output = io.StringIO()
    df_cleaned.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(output, media_type="text/csv")


@app.post("/treatment_summary/")
async def treatment_summary(file: UploadFile = File(...)):
    contents = await file.read()
    data_str = StringIO(contents.decode("utf-8", errors="ignore"))
    df = pd.read_csv(data_str)

    N = len(df)
    sample_size = min(cochran_sample_size(N), 180) #enforcing 180 max sample size to keep token use in check
    df_sample = df.sample(n=sample_size, random_state=42)

    dataset_summary = {
        "columns": list(df.columns),
        "column_types": df.dtypes.astype(str).to_dict(),
        "sample_data": df_sample.to_dict(orient="records")
    }
    df_json = json.dumps(dataset_summary)

    client = OpenAI(api_key=openai_key.api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Your job is to analyze the dataset and provide a step-by-step preprocessing guide. Priorities: handling missing values, encoding categorical variables, and outlier detection."},
            {"role": "user", "content": f"Here is my dataset summary:\n{df_json}\n"
                                        "Provide a tailored list of preprocessing steps you would recommend for the information provided in the dataset summary.\n"
                                        "This list must be specific and tailored to the dataset in question. In short, all recommendations must be based on the dataset summary."}
        ]
    )
    
    summary = response.choices[0].message.content
    return StreamingResponse(summary, media_type="text/csv")
