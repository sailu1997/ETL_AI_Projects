"""
Clean File Node

This node handles data cleaning for uploaded files using LLM-based approach.
Based on the data cleaning functionality from the main backend.
"""

import os
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, date
from .base_node import BaseNode
from state import ConversationState, state_manager
from openai import OpenAI

def clean_for_json(obj):
    """Clean object for JSON serialization - using proven approach from old backend"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists for JSON serialization
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Interval):
        return {
            'left': clean_for_json(obj.left),
            'right': clean_for_json(obj.right),
            'closed': obj.closed
        }
    elif isinstance(obj, pd.DataFrame):
        return obj.applymap(clean_for_json).to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.apply(clean_for_json).to_list()
    elif isinstance(obj, dict):
        return {clean_for_json(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict', None)):
        # Handle Plotly figures and other objects with to_dict method
        try:
            return obj.to_dict()
        except Exception as e:
            print(f"DEBUG: Error converting object to dict: {e}")
            return str(obj)
    elif hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json', None)):
        # Handle objects with to_json method
        try:
            import json
            return json.loads(obj.to_json())
        except Exception as e:
            print(f"DEBUG: Error converting object to JSON: {e}")
            return str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

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

def standardize_categorical_values(df: pd.DataFrame, column: str, client: OpenAI) -> pd.DataFrame:
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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data standardization expert. Provide only the Python dictionary mapping."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
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

class CleanFileNode(BaseNode):
    """Node for cleaning uploaded files using LLM-based approach"""
    
    def __init__(self):
        super().__init__(
            name="clean_file",
            description="Cleans uploaded files using LLM-based intelligent data cleaning"
        )
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "file_ids"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Clean the specified file using LLM-based approach"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for clean_file node")
            
            self.log_execution(state)
            
            session_id = state["session_id"]
            file_ids = state["file_ids"]
            
            # For now, clean the first file in the list
            if not file_ids:
                raise ValueError("No file IDs provided for cleaning")
            file_id = file_ids[0]
            
            # Get file information from session
            files = state_manager.get_session_files(session_id)
            file_info = next((f for f in files if f['file_id'] == file_id), None)
            
            if not file_info:
                raise ValueError(f"File with id {file_id} not found for session {session_id}")
            
            file_path = file_info.get('uploaded_filename')
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at {file_path}")
            
            # Read the original file
            df = pd.read_csv(file_path, low_memory=False)
            original_df = df.copy()
            
            # Prepare dataset information for LLM
            schema_str = df.dtypes.to_string()
            null_counts = df.isnull().sum().to_dict()
            sample_rows = df.head(5).to_csv(index=False)
            summary_stats = df.describe(include='all').to_string()
            
            duplicate_count = df.duplicated().sum()
            column_stats = {
                col: {
                    "unique_values": int(df[col].nunique()),
                    "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
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
                            "numeric_count": int(numeric_count),
                            "non_numeric_count": int(non_numeric_count)
                        }

            # Generate LLM-based cleaning prompt
            prompt = f"""You are a senior data cleaning expert. Analyze the dataset below and return only the Python code required to clean it. Your goal is to ensure high data integrity while preserving meaningful business information.

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
- Normalize by lowercasing, trimming spaces, and replacing hyphens/underscores with spaces.
- Map similar values to a single canonical form.

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
```"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
                
                # Save cleaned file
                cleaned_filename = f"cleaned_data_{file_id}.csv"
                cleaned_file_path = os.path.join("uploaded_files", cleaned_filename)
                cleaned_df.to_csv(cleaned_file_path, index=False)
                
                # Update file information in session
                file_info["cleaned_filename"] = cleaned_file_path
                file_info["is_cleaned"] = True
                file_info["cleaning_timestamp"] = datetime.now().isoformat()
                file_info["cleaning_code"] = cleaning_code
                file_info["impact_metrics"] = impact_metrics
                
                # Clean the file_info for JSON serialization
                cleaned_file_info = clean_for_json(file_info)
                
                # Update the file in the database directly since SimpleStateManager doesn't have update_file_in_session
                try:
                    from database import db_manager
                    # Update the file record with cleaned information
                    db_manager.update_file_cleaned_status(file_id, cleaned_file_path)
                    print(f"DEBUG: CleanFileNode - Updated file {file_id} as cleaned in database")
                except Exception as e:
                    print(f"WARNING: Could not update file status in database: {e}")
                    # Continue anyway as the state is updated
                
                # Update dataframe_info in state with cleaned file metadata
                if state.get("dataframe_info") is None:
                    state["dataframe_info"] = {}
                
                # If this file_id already exists in dataframe_info, update it; otherwise add it
                if file_id in state["dataframe_info"]:
                    # Update existing entry
                    state["dataframe_info"][file_id].update({
                        "columns": cleaned_df.columns.tolist(),
                        "data_types": cleaned_df.dtypes.to_dict(),
                        "rows": int(len(cleaned_df)),  # Convert numpy.int64 to native int
                        "columns_count": int(len(cleaned_df.columns)),  # Convert numpy.int64 to native int
                        "numeric_columns": cleaned_df.select_dtypes(include=[np.number]).columns.tolist(),
                        "categorical_columns": cleaned_df.select_dtypes(include=['object']).columns.tolist(),
                        "date_columns": cleaned_df.select_dtypes(include=['datetime64']).columns.tolist(),
                        "missing_values": {col: int(count) for col, count in cleaned_df.isnull().sum().to_dict().items()},  # Convert numpy.int64 to native int
                        "sample_data": cleaned_df.head(3).to_dict(orient='records'),
                        "cleaning_timestamp": datetime.now().isoformat(),
                        "is_cleaned": True,
                        "cleaning_impact": impact_metrics
                    })
                else:
                    # Add new entry
                    state["dataframe_info"][file_id] = {
                        "file_id": file_id,
                        "filename": file_info['original_filename'],
                        "columns": cleaned_df.columns.tolist(),
                        "data_types": cleaned_df.dtypes.to_dict(),
                        "rows": int(len(cleaned_df)),  # Convert numpy.int64 to native int
                        "columns_count": int(len(cleaned_df.columns)),  # Convert numpy.int64 to native int
                        "numeric_columns": cleaned_df.select_dtypes(include=[np.number]).columns.tolist(),
                        "categorical_columns": cleaned_df.select_dtypes(include=['object']).columns.tolist(),
                        "date_columns": cleaned_df.select_dtypes(include=['datetime64']).columns.tolist(),
                        "missing_values": {col: int(count) for col, count in cleaned_df.isnull().sum().to_dict().items()},  # Convert numpy.int64 to native int
                        "sample_data": cleaned_df.head(3).to_dict(orient='records'),
                        "upload_timestamp": file_info.get('upload_timestamp', ''),
                        "cleaning_timestamp": datetime.now().isoformat(),
                        "is_cleaned": True,
                        "cleaning_impact": impact_metrics
                    }
                
                # Generate cleaning summary
                cleaning_summary = self._generate_cleaning_summary(original_df, cleaned_df, impact_metrics)
                
                # Update state
                state["cleaned_file"] = file_info
                state["cleaning_summary"] = cleaning_summary
                state["impact_metrics"] = impact_metrics
                state["current_step"] = "file_cleaned"
                state["reasoning"] = f"Successfully cleaned {file_info['original_filename']} using LLM-based approach - removed {impact_metrics['rows_removed']} rows and modified {len(impact_metrics['columns_modified'])} columns"
                
                self.log_execution(state, f"File cleaned using LLM: {file_info['original_filename']}")
                return state
                
            except Exception as exec_error:
                print(f"Error executing cleaning code: {str(exec_error)}")
                print(f"Generated code: {cleaning_code}")
                raise exec_error
                
        except Exception as e:
            return self.handle_error(state, e)
    
    def _generate_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, 
                                 impact_metrics: Dict) -> Dict[str, Any]:
        """Generate a summary of the cleaning operations"""
        return {
            "original_shape": (int(original_df.shape[0]), int(original_df.shape[1])),  # Convert numpy.int64 to native int
            "cleaned_shape": (int(cleaned_df.shape[0]), int(cleaned_df.shape[1])),  # Convert numpy.int64 to native int
            "impact_metrics": impact_metrics,
            "data_quality_improvement": {
                "null_reduction": int(sum(impact_metrics["nulls_removed"].values())),  # Convert numpy.int64 to native int
                "duplicate_removal": int(impact_metrics["duplicates_removed"]),  # Convert numpy.int64 to native int
                "columns_improved": int(len(impact_metrics["columns_modified"])),  # Convert numpy.int64 to native int
                "columns_dropped": int(len(impact_metrics["columns_dropped"]))  # Convert numpy.int64 to native int
            },
            "summary": f"LLM cleaned {int(original_df.shape[0])} rows → {int(cleaned_df.shape[0])} rows, "
                      f"removed {int(impact_metrics['duplicates_removed'])} duplicates, "
                      f"modified {int(len(impact_metrics['columns_modified']))} columns, "
                      f"dropped {int(len(impact_metrics['columns_dropped']))} columns"
        }

# Factory function for creating the node
def clean_file_node():
    """Create a clean file node instance"""
    return CleanFileNode() 