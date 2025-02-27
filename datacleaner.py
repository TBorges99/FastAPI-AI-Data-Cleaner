import asyncio
import io
import json
import logging
import math
import re
from io import StringIO

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from openai import OpenAI

import settings

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHATGPT_MODEL = "gpt-4o-mini"
CLEAN_DATASET_SYSTEM_MESSAGE = "Your job is to take an input dataset and return it treated as a JSON-formatted list of records. Focus on handling missing values, encoding categorical variables, and ensuring data integrity."
CLEAN_DATASET_USER_MESSAGE = "Here is my dataset:\n{dataset}\nApply cleaning steps and return the treated dataset in JSON format. Must return only the treated dataset in JSON format, nothing else!"

PREPROCESSING_SYSTEM_MESSAGE = "Your job is to analyze the dataset and provide a step-by-step preprocessing guide."
PREPROCESSING_USER_MESSAGE = "Here is my dataset summary:\n{dataset}\nProvide a tailored list of preprocessing steps you would recommend for each column of the dataset (must be specific).\nThis list must be specific and tailored to the dataset in question. In short, all recommendations must be based on the dataset summary."


# Cochran's Sample Size Formula
# We apply dataset sampling in order to limit token consumption and improve response time
# We also apply a hardcoded limit of 300 samples in case the formula returns a higher value
def cochran_sample_size(N, confidence=1.96, margin_error=0.05, p=0.5, capped_size=300):
    n0 = (confidence**2 * p * (1 - p)) / (margin_error**2)
    n = n0 / (1 + (n0 - 1) / N) if N < n0 else n0
    return min(math.ceil(n), N, capped_size)


async def get_chatgpt_completion(
        system_message: str,
        user_message: str,
) -> str:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=CHATGPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    if not response.choices:
        logger.error("OpenAI API returned an empty response")
        raise HTTPException(
            status_code=400,
            detail="OpenAI API returned an empty response. Ensure Input fits requirements",
        )
    else:
        return response.choices[0].message.content


async def load_file(file: UploadFile) -> pd.DataFrame:
    contents = await file.read()
    data_str = StringIO(contents.decode("utf-8", errors="ignore"))
    if file.content_type == "text/csv":
        return pd.read_csv(data_str)
    elif file.content_type == "application/json":
        return pd.read_json(data_str)
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV and JSON files are allowed.",
        )


async def get_cleaned_dataset(df_sample):
    dataframe_json = df_sample.to_json(orient="records", lines=False)
    dataset = json.loads(dataframe_json)
    chatgpt_completion = await get_chatgpt_completion(
        CLEAN_DATASET_SYSTEM_MESSAGE,
        CLEAN_DATASET_USER_MESSAGE.format(dataset=dataset),

    )

    return chatgpt_completion


async def get_preprocessing_summary(df_sample):
    dataset_summary = {
        "columns": list(df_sample.columns),
        "column_types":  df_sample.dtypes.to_dict(),
        "sample_data": df_sample.to_dict(orient="records")
    }
    response = await get_chatgpt_completion(
        PREPROCESSING_SYSTEM_MESSAGE,
        PREPROCESSING_USER_MESSAGE.format(dataset=json.dumps(dataset_summary)),
    )

    return response


@app.post("/treated_sample/")
async def treated_sample(file: UploadFile = File(...)):
    df = await load_file(file)
    
    # Sample dataset using Cochran’s formula
    sample_size = cochran_sample_size(len(df))
    df_sample = df.sample(n=sample_size, random_state=42)

    # Get treated dataset from OpenAI
    raw_response = await get_cleaned_dataset(df_sample)

    match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if match:
        json_part = match.group(0)
        cleaned_json = json.loads(json_part)
    else:
        cleaned_json = None

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
