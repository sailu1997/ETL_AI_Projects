"""
Upload File Node

This node handles file uploads, session creation, and initial file processing.
Based on the file upload functionality from the main backend.
"""

import os
import uuid
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
from io import StringIO, BytesIO
from .base_node import BaseNode
from state import ConversationState, state_manager
import numpy as np

class UploadFileNode(BaseNode):
    """Node for handling file uploads and session management"""
    
    def __init__(self):
        super().__init__(
            name="upload_file",
            description="Handles file upload, session creation, and initial file processing"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "user_query"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Handle file upload and session creation"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for upload_file node")
            
            self.log_execution(state)
            
            # Get file information from result field
            if not state.get("result") or "file_content" not in state["result"]:
                raise ValueError("File content not found in state result")
            
            file_content = state["result"]["file_content"]
            filename = state["result"]["filename"]
            should_clean = state["result"].get("should_clean", True)
            
            # Get or create session ID
            session_id = state.get("session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                state["session_id"] = session_id
            
            # Parse file based on type
            filename_lower = filename.lower()
            if filename_lower.endswith(".csv"):
                decoded = file_content.decode("utf-8")
                df = pd.read_csv(StringIO(decoded))
            elif filename_lower.endswith((".xls", ".xlsx")):
                df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            else:
                raise ValueError("Unsupported file type")
            
            # Generate unique file ID and save file
            file_id = str(uuid.uuid4())
            uploaded_filename = f"uploaded_data_{file_id}.csv"
            
            # Create upload directory if it doesn't exist
            upload_dir = "uploaded_files"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, uploaded_filename)
            df.to_csv(file_path, index=False)
            
            # Create file data
            file_data = {
                "original_filename": filename,
                "uploaded_filename": file_path,
                "cleaned_filename": None,
                "file_type": filename_lower.split('.')[-1].lower(),
                "file_size": len(file_content),
                "upload_timestamp": datetime.now().isoformat(),
                "is_cleaned": False,
                "metadata": {
                    "rows": int(len(df)),  # Convert numpy.int64 to native int
                    "columns": int(len(df.columns)),  # Convert numpy.int64 to native int
                    "column_names": df.columns.tolist()
                }
            }
            
            # Add file to session (this will generate the actual file_id)
            actual_file_id = state_manager.add_file(session_id, file_data)
            
            # Generate basic insights
            insights = self._generate_file_insights(df, filename)
            
            # Update state with file information
            state["file_ids"] = [actual_file_id]
            state["current_step"] = "file_uploaded"
            
            # Store dataframe info in state for intelligent file selection
            if state.get("dataframe_info") is None:
                state["dataframe_info"] = {}
            
            state["dataframe_info"][actual_file_id] = {
                "file_id": actual_file_id,
                "filename": filename,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "rows": int(len(df)),  # Convert numpy.int64 to native int
                "columns_count": int(len(df.columns)),  # Convert numpy.int64 to native int
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
                "missing_values": {col: int(count) for col, count in df.isnull().sum().to_dict().items()},  # Convert numpy.int64 to native int
                "sample_data": df.head(3).to_dict(orient='records'),
                "upload_timestamp": datetime.now().isoformat(),
                "is_cleaned": False
            }
            
            # Store all file information in result field
            if state.get("result") is None:
                state["result"] = {}
            
            state["result"]["uploaded_file"] = file_data
            state["result"]["file_insights"] = insights
            state["result"]["reasoning"] = f"Successfully uploaded {filename} with {int(len(df))} rows and {int(len(df.columns))} columns"
            
            # Add AI message to conversation
            from langchain_core.messages import AIMessage
            ai_message = f"File '{filename}' uploaded successfully. Found {int(len(df))} rows and {int(len(df.columns))} columns."
            state["messages"].append(AIMessage(content=ai_message))
            
            self.log_execution(state, f"File uploaded: {filename}")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _generate_file_insights(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Generate basic insights about the uploaded file"""
        try:
            insights = {
                "filename": filename,
                "file_size": int(len(df)),  # Convert numpy.int64 to native int
                "columns": int(len(df.columns)),  # Convert numpy.int64 to native int
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict(),
                "missing_values": {col: int(count) for col, count in df.isnull().sum().to_dict().items()},  # Convert numpy.int64 to native int
                "duplicate_rows": int(df.duplicated().sum()),  # Convert numpy.int64 to native int
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
                "date_columns": df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Add basic statistics for numeric columns
            if insights["numeric_columns"]:
                numeric_df = df[insights["numeric_columns"]]
                insights["numeric_stats"] = {
                    col: {
                        "mean": float(numeric_df[col].mean()) if not numeric_df[col].isna().all() else None,
                        "median": float(numeric_df[col].median()) if not numeric_df[col].isna().all() else None,
                        "std": float(numeric_df[col].std()) if not numeric_df[col].isna().all() else None,
                        "min": float(numeric_df[col].min()) if not numeric_df[col].isna().all() else None,
                        "max": float(numeric_df[col].max()) if not numeric_df[col].isna().all() else None
                    }
                    for col in insights["numeric_columns"]
                }
            
            return insights
            
        except Exception as e:
            return {
                "filename": filename,
                "error": f"Failed to generate insights: {str(e)}"
            }

# Factory function for creating the node
def upload_file_node():
    """Create an upload file node instance"""
    return UploadFileNode() 