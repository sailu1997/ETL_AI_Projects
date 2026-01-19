import os
import io
import pandas as pd
import uuid
from typing import List, Any
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from fastapi.responses import JSONResponse
import re
import streamlit as st
from matplotlib.figure import Figure
import base64
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
from datetime import datetime, date
import plotly.express as px
import uuid

app = FastAPI()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
#file_path = os.path.join(UPLOAD_DIR, "uploaded_data.csv")
cleaned_csv = os.path.join(UPLOAD_DIR, "cleaned_data.csv")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def QueryUnderstandingTool(query: str) -> bool:
    """Return True if the query seems to request a visualisation based on keywords."""
    # Use LLM to understand intent instead of keyword matching
    messages = [
        {"role": "system", "content": "You are a smart assistant that determines if a query is requesting a data visualization. "
            "Return only 'true' if the query implies or explicitly asks for a plot, chart, trend over time, "
            "comparison between values, or any visual representation of data. Return 'false' if the query is asking only for text or numbers."},
        {"role": "user", "content": query}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=5  
    )
    
    # Extract the response and convert to boolean
    intent_response = response.choices[0].message.content.strip().lower()
    return intent_response == "true"

def PlotCodeGeneratorTool_ex(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas+plotly code for a plot based on the query and columns."""
    # Read the data
    df = pd.read_csv(cleaned_csv, low_memory=False)

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
            You are a senior data analyst. Given the following information about a pandas DataFrame named `df`, write Python code that uses **pandas** for manipulation and **plotly.graph_objects** (as `go`) for plotting.

            ### Valid Columns in the DataFrame:
            {', '.join(cols)}

            ### Data Profile:
            - Row count: {data_profile['row_count']}
            - Unique value counts: {data_profile['column_counts']}
            - Missing values: {data_profile['missing_values']}
            - Data types: {data_profile['data_types']}

            ### Top 10 Values in Categorical Columns:
            {data_profile['categorical_values']}

            ---
            Actual Values in Categorical Columns:
                {data_profile['categorical_values']}
                
            ### Query:
            "{query}"

            ---

            ### Instructions:
            1. Use **only** the columns listed above. ❗ **Do NOT invent or assume column names**.
            2. If the query requires a column that doesn't exist, raise: `raise ValueError("Column 'X' not found")`.
            3. Use **pandas** for data manipulation.
            4. Use **plotly.graph_objects (go)** for plotting.
            5. Assign the final output to a variable named `result`.
            6. Create only **one meaningful plot** with:
                - Proper axis labels
                - Clear and informative title
                - Clean tick formatting (rotate where needed)
            7. Axis rules:
                - Date index: format using `.dt.strftime('%Y-%m')`
                - Categorical: ensure proper ordering, group rare items into `'Other'` if >10 categories
                - Numeric: ensure readable scale
            8. Always validate inputs before using:
                - Check column existence: if not found, raise `ValueError(f"Column '{col}' not found")`
                - For `pd.cut()` or histogram bins, ensure bins are sorted and strictly increasing.
                - For division or ratio calculations, check denominator for 0 before dividing.
                - When grouping by categorical variables, always check the number of unique values. If >10, group least frequent into `'Other'`.

            9. When filtering or binning:
                - Use defensive coding to avoid empty DataFrames, invalid bin edges, or label mismatches.

            10. Do not assume categories, values, or ranges exist unless they are shown above.
            8. Clean string values: strip spaces, ensure consistent casing (e.g., title case or lowercase).
            9. Do NOT use `.show()` or `.write_image()` or save/display the plot. Just assign to `result`.
            10. Always assign the plot as `result = fig` (where `fig` is the `go.Figure()` object).
            11. Output your final code inside a **single code block** starting with ```python and ending with ```.

            You must return **only valid Python code**, nothing else.
            """

def DataFrameSummaryTool(df: pd.DataFrame) -> str:
    """Generate a summary prompt string for the LLM based on the DataFrame."""
    prompt = f"""
        Given a dataset with {len(df)} rows and {len(df.columns)} columns:
        Columns: {', '.join(df.columns)}
        Data types: {df.dtypes.to_dict()}
        Missing values: {df.isnull().sum().to_dict()}

        Provide:
        1. A brief description of what this dataset contains
        2. 3-4 possible data analysis questions that could be explored
        Keep it concise and focused."""
    return prompt

def CodeWritingTool_ex(cols: List[str], query: str) -> str:
    """Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)."""

    df = pd.read_csv(cleaned_csv, low_memory=False)
    
    categorical_analysis = {}
    for col in df.select_dtypes(include=['object']).columns:
        value_counts = df[col].value_counts().to_dict()  # Get top 10 values
        categorical_analysis[col] = value_counts

    # Calculate data profile
    data_profile = {
        "row_count": len(df),
        "column_counts": {col: df[col].nunique() for col in df.columns},  # Removed [:10] slice
        "missing_values": {col: df[col].isna().sum() for col in df.columns},  # Removed [:10] slice
        "data_types": {col: str(df[col].dtype) for col in df.columns},  # Removed [:10] slice
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
    4. If binning numeric columns, make sure:
       - Bins are strictly increasing
       - You add a small buffer to max() if needed
       - You check if the column has enough unique values
    5. When using fillna(), always specify a value.
    6. Do not use `pd.cut()` directly unless the values are preprocessed to be valid.
    7. Wrap the snippet in a single ```python code fence (no extra prose). 
    """

def ReasoningCurator(query: str, result: Any) -> str:
    """Builds and returns the LLM prompt for reasoning about the result."""
    is_error = isinstance(result, str) and result.startswith("Error executing code")
    is_plot = isinstance(result, (plt.Figure, plt.Axes))

    if is_error:
        desc = result
    elif is_plot:
        title = ""
        if isinstance(result, plt.Figure):
            title = result._suptitle.get_text() if result._suptitle else ""
        elif isinstance(result, plt.Axes):
            title = result.get_title()
        desc = f"[Plot Object: {title or 'Chart'}]"
    else:
        desc = str(result)

    if is_plot:
        prompt = f'''
        The user asked: "{query}".
        Below is a description of the plot result:
        {desc}
        Explain in 2–3 concise sentences what the chart shows (no code talk).'''
    else:
        prompt = f'''
        The user asked: "{query}".
        The result value is: {desc}
        Explain concisely what this tells about the data (no mention of charts).'''
    return prompt

def preprocess_column_names(columns):
    clean_columns = []
    for col in columns:
        col_clean = re.sub(r'[^\w\s]', '', col)
        col_clean = re.sub(r'\s+', '_', col_clean)
        clean_columns.append(col_clean.lower())
    return clean_columns

def make_json_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        # Convert DataFrame to dict, but handle Interval objects first
        df_copy = obj.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'interval':
                # Convert interval objects to strings
                df_copy[col] = df_copy[col].astype(str)
        return df_copy.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        # Handle Series with Interval dtype
        if obj.dtype == 'interval':
            return obj.astype(str).to_dict()
        return obj.to_dict()
    elif isinstance(obj, (float, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Interval):
        return {
            'left': make_json_serializable(obj.left),
            'right': make_json_serializable(obj.right),
            'closed': obj.closed
        }
    elif isinstance(obj, go.Figure):
        fig_dict = obj.to_dict()
        serialized_fig_dict = make_json_serializable(fig_dict)
        return {
            "type": "plotly",
            "figure_json": serialized_fig_dict
        }
    elif isinstance(obj, Figure):
        buf = io.BytesIO()
        obj.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return {"image_base64": img_base64}
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (str, int, bool, type(None))):
        return obj
    else:
        return str(obj)

def standardize_categorical_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Use LLM to standardize categorical values in a column."""
    # Get unique values
    unique_values = df[column].unique().tolist()
    
    # Skip if no unique values or all null
    if not unique_values or all(pd.isna(x) for x in unique_values):
        return df
    
    prompt = f"""
    Given these unique values from a column named '{column}':
    {unique_values}

    Please standardize these values by:
    1. Identifying which values represent the same category (e.g., "5 room" and "5-room" are the same)
    2. Choosing the most appropriate standard format for each category
    3. Providing a Python dictionary mapping original values to standardized values

    Return ONLY a Python dictionary in this format:
    {{
        "original_value1": "standardized_value1",
        "original_value2": "standardized_value2",
        ...
    }}

    Rules:
    - Keep the standardization simple and consistent
    - Use the most common or clearest format as the standard
    - Handle variations in spacing, hyphens, and capitalization
    - Preserve the meaning of each category
    - If a value is None or NaN, map it to "Unknown"
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a data standardization expert. Provide only the Python dictionary mapping."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    try:
        # Extract the dictionary from the response
        mapping_str = response.choices[0].message.content.strip()
        # Find the dictionary in the response
        dict_start = mapping_str.find('{')
        dict_end = mapping_str.rfind('}') + 1
        mapping_str = mapping_str[dict_start:dict_end]
        
        # Convert string to dictionary
        mapping = eval(mapping_str)
        
        # Add mapping for None/NaN values if not present
        if None not in mapping:
            mapping[None] = "Unknown"
        if pd.NA not in mapping:
            mapping[pd.NA] = "Unknown"
        
        # Apply the mapping to the column
        df[column] = df[column].map(mapping)
        return df
    except Exception as e:
        print(f"Error in standardizing {column}: {str(e)}")
        return df

def CodeWritingTool(dfs: dict, query: str) -> str:
    """
    Generate a prompt for the LLM to write pandas-only code for a data query (no plotting)
    based on the query and multiple DataFrames.
    """
    # Gather info for each DataFrame
    df_profiles = []
    for file_id, df in dfs.items():
        categorical_analysis = {}
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts().to_dict()
            categorical_analysis[col] = value_counts

        data_profile = {
            "row_count": len(df),
            "column_counts": {col: df[col].nunique() for col in df.columns},
            "missing_values": {col: df[col].isna().sum() for col in df.columns},
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "categorical_values": categorical_analysis
        }

        df_profiles.append(f"""
### DataFrame: `{file_id}`
- Columns: {', '.join(df.columns)}
- Row count: {data_profile['row_count']}
- Column value counts: {data_profile['column_counts']}
- Missing values: {data_profile['missing_values']}
- Data types: {data_profile['data_types']}
- Actual Values in Categorical Columns: {data_profile['categorical_values']}
""")

    all_profiles = "\n".join(df_profiles)

    return f"""
You are a senior data analyst. The user has uploaded the following pandas DataFrames, available in a dictionary called `dfs` (keyed by file_id):

{all_profiles}

---

### Query:
"{query}"

---

### Instructions:
1. Decide which DataFrame(s) are relevant to answer the query. Use `dfs["file_id"]` to access a DataFrame.
2. If the query requires joining/merging, do so using pandas.
3. Use pandas operations only (no plotting).
4. Assign the final result to a variable named `result`.
5. Handle missing values and data type issues appropriately.
6. If binning numeric columns, make sure:
   - Bins are strictly increasing
   - You add a small buffer to max() if needed
   - You check if the column has enough unique values
7. When using fillna(), always specify a value.
8. Do not use `pd.cut()` directly unless the values are preprocessed to be valid.
9. Wrap the snippet in a single ```python code fence (no extra prose).
"""

def PlotCodeGeneratorTool(dfs: dict, query: str) -> str:
    """
    Generate a prompt for the LLM to write pandas+plotly code for a plot
    based on the query and multiple DataFrames.
    """
    # Gather info for each DataFrame
    df_profiles = []
    for file_id, df in dfs.items():
        categorical_analysis = {}
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts().head(10).to_dict()
            categorical_analysis[col] = value_counts

        data_profile = {
            "row_count": len(df),
            "column_counts": {col: df[col].nunique() for col in df.columns},
            "missing_values": {col: df[col].isna().sum() for col in df.columns},
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "categorical_values": categorical_analysis
        }

        df_profiles.append(f"""
        ### DataFrame: `{file_id}`
        - Columns: {', '.join(df.columns)}
        - Row count: {data_profile['row_count']}
        - Unique value counts: {data_profile['column_counts']}
        - Missing values: {data_profile['missing_values']}
        - Data types: {data_profile['data_types']}
        - Top 10 Values in Categorical Columns: {data_profile['categorical_values']}
        """)
    all_profiles = "\n".join(df_profiles)

    return f"""
    You are a senior data analyst. The user has uploaded the following pandas DataFrames, available in a dictionary called `dfs` (keyed by file_id):

    {all_profiles}

    ---

    ### Query:
    "{query}"

    ---

    ### Instructions:
    1. Use **only** the columns listed above in the all_profiles. ❗ **Do NOT invent or assume column names**.
    2. Decide which DataFrame(s) are relevant to answer the query. Use `dfs["file_id"]` to access a DataFrame.
    3. If the query requires joining/merging, do so using pandas.
    4. Use **pandas** for data manipulation and **plotly.graph_objects (go)** for plotting.
    5. Assign the final output to a variable named `result`.
    6. Create only **one or multiple meaningful plots** with:
        - Proper axis labels
        - Clear and informative title
        - Clean tick formatting (rotate where needed)
    7. Axis rules:
        - Date index: format using `.dt.strftime('%Y-%m')`
        - Categorical: ensure proper ordering, group rare items into `'Other'` if >10 categories
        - Numeric: ensure readable scale
    8. Always validate inputs before using:
        - Check column existence: if not found, raise `ValueError(f"Column '{{col}}' not found")`
        - For `pd.cut()` or histogram bins, ensure bins are sorted and strictly increasing.
        - For division or ratio calculations, check denominator for 0 before dividing.
        - When grouping by categorical variables, always check the number of unique values. If >10, group least frequent into `'Other'`.
    9. When filtering or binning:
        - Use defensive coding to avoid empty DataFrames, invalid bin edges, or label mismatches.
    10. Do not assume categories, values, or ranges exist unless they are shown above.
    11. Clean string values: strip spaces, ensure consistent casing (e.g., title case or lowercase).
    12. Do NOT use `.show()` or `.write_image()` or save/display the plot. Just assign to `result`.
    13. Always assign the plot as `result = fig` (where `fig` is the `go.Figure()` object).
    14. Output your final code inside a **single code block** starting with ```python and ending with ```.

    You must return **only valid Python code**, nothing else.
    """

# === Helper ===
def extract_first_code_block(text: str) -> str:
    """Extracts the first Python code block from a markdown-formatted string."""
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === Pydantic Models ===
class QueryRequest(BaseModel):
    query: str
    columns: List[str]

class ExecutionRequest(BaseModel):
    code: str
    should_plot: bool
    file_ids: List[str]

class ReasoningRequest(BaseModel):
    query: str
    result: Any
    file_ids: List[str]

class CodeRequest(BaseModel):
    query: str
    file_ids: List[str]

class Request(BaseModel):
    file_id: str

# === API ROUTES ===
@app.post("/codegeneration/")
async def CodeGenerationAgent(payload: CodeRequest):
    dfs = {}
    for file_id in payload.file_ids:
        file_path = os.path.join(UPLOAD_DIR, "cleaned_data_"+file_id+".csv")
        dfs[file_id] = pd.read_csv(file_path)
        print(dfs[file_id].columns)
    df_info = "\n".join(
        [f"File ID: {fid}, Columns: {', '.join(df.columns)}" for fid, df in dfs.items()]
    )

    should_plot = QueryUnderstandingTool(payload.query)
    print(f"Query: {payload.query}, Should plot: {should_plot}")

    prompt = PlotCodeGeneratorTool(dfs, payload.query) if should_plot else CodeWritingTool(dfs, payload.query)
    messages = [
        {"role": "system", "content": "You are a Python data-analysis expert who writes clean, efficient code. Solve the given problem with optimal pandas operations. Be concise and focused. Your response must contain ONLY a properly-closed ```python code block with no explanations before or after."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2
    )
    full_response = response.choices[0].message.content
    code = extract_first_code_block(full_response)
    print(f"Generated code: {code}")
    return {"code": code, "should_plot": should_plot}

@app.post("/executionagent/")
async def ExecutionAgent(payload: ExecutionRequest):
    # Load all DataFrames into a dict
    dfs = {}
    for file_id in payload.file_ids:
        file_path = os.path.join(UPLOAD_DIR, "cleaned_data_"+file_id+".csv")
        dfs[file_id] = pd.read_csv(file_path)

    # Prepare the execution environment
    env = {
        "pd": pd,
        "dfs": dfs,  # Make all DataFrames available as 'dfs'
        "px": px,
        "go": go,
        "np": np,
        "plt": plt,
        "Figure": Figure,
        "base64": base64,
        "io": io
    }

    if payload.should_plot:
        plt.rcParams["figure.dpi"] = 100
        env["plt"] = plt
        env["io"] = io

    try:
        exec(payload.code, env, env)
        result = env.get("result", None)
        print(result)
        result = make_json_serializable(result)
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        print(f"Error executing code: {exc}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error executing code: {exc}"}
        )

@app.post("/reasoningaagent/")
async def ReasoningAgent(payload: ReasoningRequest):
    # Load all DataFrames into a dict
    dfs = {}
    for file_id in payload.file_ids:
        file_path = os.path.join(UPLOAD_DIR, "cleaned_data_"+file_id+".csv")
        dfs[file_id] = pd.read_csv(file_path)

    # Optionally, you can summarize the available DataFrames for the LLM
    df_info = "\n".join(
        [f"File ID: {fid}, Columns: {', '.join(df.columns)}" for fid, df in dfs.items()]
    )

    prompt = ReasoningCurator(payload.query, payload.result)
    is_error = isinstance(payload.result, str) and payload.result.startswith("Error executing code")
    is_plot = isinstance(payload.result, (plt.Figure, plt.Axes))

    # Streaming LLM call
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "detailed thinking on. You are an insightful data analyst."},
            {"role": "user", "content": f"{df_info}\n\n{prompt}"}
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True
    )

    thinking_placeholder = st.empty()
    full_response = ""
    thinking_content = ""
    in_think = False

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            token = chunk.choices[0].delta.content
            full_response += token

            # Simple state machine to extract <think>...</think> as it streams
            if "<think>" in token:
                in_think = True
                token = token.split("<think>", 1)[1]
            if "</think>" in token:
                token = token.split("</think>", 1)[0]
                in_think = False
            if in_think or ("<think>" in full_response and not "</think>" in full_response):
                thinking_content += token
                thinking_placeholder.markdown(
                    f'<details class="thinking" open><summary>🤔 Model Thinking</summary><pre>{thinking_content}</pre></details>',
                    unsafe_allow_html=True
                )

    # After streaming, extract final reasoning (outside <think>...</think>)
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    return {"thinking_content":thinking_content, "cleaned":cleaned}

#=== For uploaded file ===
@app.post("/insights/")
async def data_insight(csv_file: UploadFile= File(...)):
    content = await csv_file.read()
    filename = csv_file.filename.lower()
    try:
        if filename.endswith(".csv"):
            decoded = content.decode("utf-8")
            df = pd.read_csv(StringIO(decoded))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(BytesIO(content), engine='openpyxl')

        file_id = uuid.uuid4()
        file_path = os.path.join(UPLOAD_DIR, "uploaded_data_" + str(file_id)+".csv")
        df.to_csv(file_path, index=False)

        nulls = df.isnull().sum()
        null_summary = nulls[nulls > 0].to_dict()

        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    date_mask = pd.to_datetime(df[col], errors='coerce').notna()
                    if date_mask.any():
                        df[f'{col}_date'] = pd.to_datetime(df[col], errors='coerce').dt.date
                        df[col] = df[col].where(~date_mask, 'converted_to_date')
                except:
                    continue

        dtypes = df.dtypes.astype(str).to_dict()

        duplicate_count = df.duplicated().sum()

        formatting_issues = {}
        for col in df.columns:
            if df[col].dtype == "object":
                invalid_values = []
                for val in df[col].dropna().unique():
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        invalid_values.append(str(val))
                if invalid_values:
                    formatting_issues[col] = {
                        "issue": "Non-numeric values found in a potentially numeric column",
                        "invalid_values": invalid_values
                    }
        health_summary = {
            "null_values": make_json_serializable(null_summary),
            "duplicate_rows": make_json_serializable(duplicate_count),
            "column_types": make_json_serializable(dtypes),
            "formatting_issues": make_json_serializable(formatting_issues)
        }

        prompt = DataFrameSummaryTool(df)
        messages = [
            {"role": "system", "content": "detailed thinking off. You are a data analyst providing brief, focused insights."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=512
        )
        return {"insight": response.choices[0].message.content, "health_summary": health_summary, "file_id": str(file_id)}
    
    except Exception as e:
        print("Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/clean_data/")
async def clean_data(request: Request):
    try:
        file_id = request.file_id
        file_path = os.path.join(UPLOAD_DIR, "uploaded_data_"+file_id+".csv")
        df = pd.read_csv(file_path, low_memory=False)
        original_df = df.copy()
        
        schema_str = df.dtypes.to_string()
        null_counts = df.isnull().sum().to_dict()
        sample_rows = df.head(10).to_csv(index=False)
        summary_stats = df.describe(include='all').to_string()
        
        duplicate_count = df.duplicated().sum()
        column_stats = {
            col: {
                "unique_values": int(df[col].nunique()),  # Convert to int
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),  # Convert to float
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(5).tolist() if df[col].dtype == 'object' else None
            }
            for col in df.columns
        }

        # Identify potential formatting issues
        formatting_issues = {}
        for col in df.columns:
            if df[col].dtype == "object":
                # Check for mixed numeric and non-numeric values
                numeric_count = 0
                non_numeric_count = 0
                for val in df[col].dropna().unique():
                    try:
                        float(val)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        non_numeric_count += 1
                
                if numeric_count > 0 and non_numeric_count > 0:
                    formatting_issues[col] = {
                        "issue": "Mixed numeric and non-numeric values",
                        "numeric_count": int(numeric_count),  # Convert to int
                        "non_numeric_count": int(non_numeric_count)  # Convert to int
                    }

        prompt = f"""
            You are a senior data cleaning expert. Analyze the dataset below and return only the Python code required to clean it. Your goal is to ensure high data integrity while preserving meaningful business information.

            Dataset Information:
            - Schema: {schema_str}
            - Missing values: {null_counts}
            - Duplicate rows: {duplicate_count}
            - Column statistics: {column_stats}
            - Formatting issues: {formatting_issues}

            Sample data:
            {sample_rows}

            Summary statistics:
            {summary_stats}

            Please return only the Python code that performs the following tasks:

            1. Handle missing values appropriately for each column:

            For **numeric columns**:
            - If missing rate < 5%: Use mean (for normal data) or median (for skewed data)
            - If 5% ≤ missing rate < 20%: Use median or mode
            - If missing rate ≥ 20%: Consider dropping the column or use advanced imputation

            For **categorical columns**:
            - If missing rate < 5%: Fill with mode
            - If 5% ≤ missing rate < 20%: Fill with mode or create a new category like 'Unknown'
            - If missing rate ≥ 20%: Drop the column unless business-critical

            For **date columns**:
            - If missing rate < 5%: Use median date or forward/backward fill
            - If 5% ≤ missing rate < 20%: Use forward/backward fill based on logic
            - If missing rate ≥ 20%: Drop unless business-critical

            For **mixed date/status columns**:
            - Preserve status values like 'In-progress', 'Unpaid', etc.
            - Extract and store valid date values separately (e.g., a new `_date` column)
            - Document assumptions clearly in comments

            Always:
            - Drop columns with 100% missing values
            - Drop columns with a single unique value unless business-critical
            - Print null counts before and after each step

            2. Fix formatting issues:
            - Convert mixed-type columns (e.g., numbers stored as strings) to appropriate types
            - Standardize date formats and use `datetime.date` (not datetime with time)
            - Clean strings (trim whitespace, fix case)
            - Avoid adding time to date fields unless required

            3. Handling naming conventions:
            - carefully check the catergorical variable naming conventions and modify them if there are discrepencies.

            4. If there are interval objects in the column cells, handle them accordingly
            5. Remove duplicate rows (check and remove if any)

            6. Return a cleaned DataFrame called `df`

            Additional Instructions:
            - Use only standard libraries: pandas, numpy, datetime
            - Do NOT use non-standard or obscure libraries
            - Be conservative with dropping data — prioritize preserving useful columns
            - Comment your code to explain each step
            - Make sure code is runnable and robust

            Return ONLY the code inside a Python code block:
            ```python
            # your code here
            """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Provide only the Python code solution."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        cleaning_code = extract_first_code_block(response.choices[0].message.content)
        
        # Execute the cleaning code
        try:
            # Create a new namespace for execution
            namespace = {
                "df": df.copy(),
                "pd": pd, 
                "np": np,
                "datetime": datetime,
                "date": date
            }
            
            # Execute the code in the namespace
            exec(cleaning_code, namespace)
            
            # Get the cleaned DataFrame from the namespace
            cleaned_df = namespace.get("df")
            if cleaned_df is None:
                raise ValueError("Cleaning code did not modify the DataFrame")
            
            # Convert any Interval objects to strings before calculating metrics
            for col in cleaned_df.columns:
                if cleaned_df[col].dtype == 'interval':
                    cleaned_df[col] = cleaned_df[col].astype(str)
                # Convert date columns to string format for comparison
                elif pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].dt.strftime('%Y-%m-%d')
                elif isinstance(cleaned_df[col].iloc[0] if not cleaned_df[col].empty else None, (date, datetime)):
                    cleaned_df[col] = cleaned_df[col].astype(str)
            
            # Calculate cleaning impact metrics
            impact_metrics = {
                "rows_removed": int(len(original_df) - len(cleaned_df)),
                "nulls_removed": {
                    col: int(original_df[col].isnull().sum() - cleaned_df[col].isnull().sum())
                    for col in original_df.columns if col in cleaned_df.columns
                },
                "duplicates_removed": int(original_df.duplicated().sum() - cleaned_df.duplicated().sum()),
                "columns_modified": [
                    col for col in original_df.columns
                    if col in cleaned_df.columns and str(original_df[col].dtype) != str(cleaned_df[col].dtype)
                ],
                "columns_dropped": [
                    col for col in original_df.columns
                    if col not in cleaned_df.columns
                ],
                "formatting_changes": {
                    col: {
                        "before": str(original_df[col].dtype),
                        "after": str(cleaned_df[col].dtype)
                    }
                    for col in original_df.columns
                    if col in cleaned_df.columns and str(original_df[col].dtype) != str(cleaned_df[col].dtype)
                }
            }
            
            cleaned_df.columns = preprocess_column_names(cleaned_df.columns)
            cleaned_df.to_csv(UPLOAD_DIR + "/cleaned_data_"+file_id+".csv", index=False)
            
            # Ensure we always return a cleaned sample, even if empty
            cleaned_sample = cleaned_df.head(10).copy()
            # Convert any remaining date columns to string format in the sample
            for col in cleaned_sample.columns:
                if pd.api.types.is_datetime64_any_dtype(cleaned_sample[col]):
                    cleaned_sample[col] = cleaned_sample[col].dt.strftime('%Y-%m-%d')
                elif isinstance(cleaned_sample[col].iloc[0] if not cleaned_sample[col].empty else None, (date, datetime)):
                    cleaned_sample[col] = cleaned_sample[col].astype(str)
            
            cleaned_sample = cleaned_sample.to_dict(orient="records") if not cleaned_sample.empty else []

            return {
                "message": "Cleaning successful",
                "analysis": response.choices[0].message.content,
                "impact_metrics": make_json_serializable(impact_metrics),
                "cleaned_sample": make_json_serializable(cleaned_sample),
                "applied_code": cleaning_code
            }
            
        except Exception as e:
            print(f"Error during cleaning: {str(e)}")
            print(f"Cleaning code:\n{cleaning_code}")
            return {
                "message": f"Error during cleaning: {str(e)}",
                "analysis": "",
                "impact_metrics": {},
                "cleaned_sample": [],
                "applied_code": cleaning_code
            }

    except Exception as e:
        # Return a minimal response with error message
        return {
            "message": f"Error during cleaning: {str(e)}",
            "analysis": "",
            "impact_metrics": {},
            "cleaned_sample": [],
            "applied_code": ""
        }

#upload a data 
#conversational data analysis
#plotly for plots

#two different sources of data - currently tried with single csv
#same variable name in different files, will the llm be able to answer ? - 

#primary key, foreign key (two datasets)
#Genie in databricks

#Configure the agent to give info about data
#splitting up the dataset and querying on the dataset
#make the system more deterministic

