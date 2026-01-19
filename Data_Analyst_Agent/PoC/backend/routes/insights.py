from fastapi import FastAPI, UploadFile, File
import pandas as pd
from io import StringIO
import os
from fastapi.responses import JSONResponse
from openai import OpenAI
from fastapi import APIRouter
from tools.dataframe_summary import DataFrameSummaryTool
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()


UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
file_path = os.path.join(UPLOAD_DIR, "uploaded_data.csv")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@router.post("/insights/")
async def data_insight(csv_file: UploadFile= File(...)):
    try:
        content = await csv_file.read()
        decoded = content.decode("utf-8")
        df = pd.read_csv(StringIO(decoded))

        df.to_csv(file_path, index=False)

        prompt = DataFrameSummaryTool(df)
        messages = [
            {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return {"insight": response.choices[0].message.content}
    
    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})