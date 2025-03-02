# FastAPI-AI-Data-Cleaner
This FastAPI application is designed for preprocessing and data cleaning purposes. The user can either preprocess a sample dataset automatically, or request a step-by-step guide/summary of preprocessing steps. This guide/summary is tailored to the users dataset using OpenAI’s GPT model.


## Overall Workflow 
1.	Users upload a CSV or JSON file.
2. A sample of the dataset is extracted using Cochran’s formula.  
3.	The app uses OpenAI’s GPT model to clean the data or generate a preprocessing summary.  
4.	The cleaned data or summary is returned to the user as a text response.

## Endpoints:  
•	`/treated_sample/`: Upload a file and get the cleaned and processed dataset in JSON format.  
•	`/treatment_summary/`: Upload a file and get a detailed preprocessing summary that suggests steps for handling the data.

## Error Handling:  
•	If the OpenAI API returns an empty or invalid response, appropriate error messages are returned.

## Installing:
1. Clone the repository:

```bash
git clone https://github.com/TBorges99/FastAPI-AI-Data-Cleaner
```
2. Change to the project's directory:
```bash
cd FastAPI-AI-Data-Cleaner
```

3. Once you're in the desired directory, run the following command to create a virtual environment:
```bash
python -m venv venv
```
4. Activate the virtual environment:

On macOS and Linux:

```bash
source venv/bin/activate
```
On Windows:
```bash
venv\Scripts\activate
```

5. Install the dependencies

```bash
pip install -r requirements.txt
```

6. Start the development server

```bash
fastapi run datacleaner.py
```


Personal Note: I created this app to apply some of my latest learnings about FastAPI and the Open AI API. Although this is my first complete project using these techs, I really believe this app has real world application, and can greatly assist data scientists with preprocessing Data. I tested some datasets from SKlearn, such as the Wine and Forest covertype datasets, and got very reliable preprocessing recommendations in the final output.
