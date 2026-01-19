from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import os
from openai import OpenAI
from typing import List
from tools.code_writingtool import CodeWritingTool
from tools.plot_codegenerator import PlotCodeGeneratorTool
from tools.query_understandingtool import QueryUnderstandingTool
from tools.utils import extract_first_code_block
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
file_path = os.path.join(UPLOAD_DIR, "uploaded_data.csv")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CodeRequest(BaseModel):
    query: str

@router.post("/codegeneration/")
async def CodeGenerationAgent(payload: CodeRequest):
    """Selects the appropriate code generation tool and gets code from the LLM for the user's query."""

    should_plot = QueryUnderstandingTool(payload.query)
    df = pd.read_csv(file_path)
    prompt = PlotCodeGeneratorTool(df.columns.tolist(), payload.query) if should_plot else CodeWritingTool(df.columns.tolist(), payload.query)

    messages = [
        {"role": "system", "content": "detailed thinking off. You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after. Ensure your solution is correct, handles edge cases, and follows best practices for data analysis."},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    return {"code": code, "should_plot": should_plot}

def CodeWritingTool(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Analyze actual values in categorical columns
    categorical_analysis = {}
    for col in df.select_dtypes(include=['object']).columns:
        value_counts = df[col].value_counts().head(10).to_dict()  # Get top 10 values
        categorical_analysis[col] = value_counts

    # Calculate data profile
    data_profile = {
        "row_count": len(df),
        "column_counts": {col: df[col].nunique() for col in df.columns},
        "missing_values": {col: df[col].isna().sum() for col in df.columns},
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "categorical_values": categorical_analysis
    }

    return f"""
    Given DataFrame `df` with columns: {', '.join(cols)}

    Data Profile:
    - Row count: {data_profile['row_count']}
    - Column value counts: {data_profile['column_counts']}
    - Missing values: {data_profile['missing_values']}
    - Data types: {data_profile['data_types']}
    
    Actual Values in Categorical Columns:
    {data_profile['categorical_values']}

    Write Python code (pandas **only**, no plotting) to answer:
    "{query}"

    Rules
    -----
    1. Use pandas operations on `df` only.
    2. Assign the final result to `result`.
    3. Handle missing values and data type issues appropriately.
    4. When using fillna(), always specify a value.
    5. IMPORTANT: Use the actual values shown above for any categorical operations.
    6. Do not assume values exist in the data - check the actual values first.
    7. Wrap the snippet in a single ```python code fence (no extra prose).
    """