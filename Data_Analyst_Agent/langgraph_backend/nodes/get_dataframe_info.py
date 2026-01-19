"""
Get DataFrame Info Node

This node gets information about a specific DataFrame including columns, data types, and sample data.
Based on the GetDataFrameInfoTool from LangChain.
"""

import os
import pandas as pd
import io
from typing import Dict, Any
from .base_node import BaseNode
from state import ConversationState, state_manager

class GetDataFrameInfoNode(BaseNode):
    """Node for getting DataFrame information"""
    
    def __init__(self):
        super().__init__(
            name="get_dataframe_info",
            description="Gets information about a specific DataFrame, including columns, data types, and sample data"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "file_id"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Get DataFrame information for the specified file"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for get_dataframe_info node")
            
            self.log_execution(state)
            
            session_id = state["session_id"]
            file_id = state["file_id"]
            
            # Get file information from session
            files = state_manager.get_session_files(session_id)
            file_info = next((f for f in files if f['file_id'] == file_id), None)
            
            if not file_info:
                raise ValueError(f"File with id {file_id} not found for session {session_id}")
            
            # Use cleaned file if available, otherwise use uploaded file
            file_path = file_info.get('cleaned_filename') or file_info.get('uploaded_filename')
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found at {file_path}")
            
            # Read the DataFrame
            df = pd.read_csv(file_path)
            
            # Get DataFrame info
            buffer = io.StringIO()
            df.info(buf=buffer)
            info = buffer.getvalue()
            
            # Create DataFrame information
            dataframe_info = {
                "file_id": file_id,
                "columns": df.columns.tolist(),
                "info": info,
                "head": df.head().to_dict(orient='records'),
                "shape": df.shape,
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            
            # Update state with DataFrame information
            state["dataframe_info"] = dataframe_info
            state["current_step"] = "dataframe_info_retrieved"
            state["reasoning"] = f"Successfully retrieved information for {file_info['original_filename']} with {df.shape[0]} rows and {df.shape[1]} columns"
            
            self.log_execution(state, f"DataFrame info retrieved for {file_id}")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)

# Factory function for creating the node
def get_dataframe_info_node():
    """Create a get dataframe info node instance"""
    return GetDataFrameInfoNode() 