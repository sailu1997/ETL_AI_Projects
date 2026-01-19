"""
List Files Node

This node lists all available cleaned data files for the current session.
Based on the ListFilesToolLC from LangChain.
"""

import os
from typing import Dict, Any, List
from .base_node import BaseNode
from state import ConversationState
from state import state_manager

class ListFilesNode(BaseNode):
    """Node for listing available cleaned data files"""
    
    def __init__(self):
        super().__init__(
            name="list_files",
            description="Lists all available cleaned data files for the current session"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """List available cleaned data files for the session"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for list_files node")
            
            self.log_execution(state)
            
            session_id = state["session_id"]
            
            # Get files for the specific session
            files = state_manager.get_session_files(session_id)
            
            # Return all files (both uploaded and cleaned)
            file_ids = [f['file_id'] for f in files]
            
            # Update state with file information
            state["file_ids"] = file_ids
            state["available_files"] = files
            state["current_step"] = "files_listed"
            
            # Add reasoning about the operation
            state["reasoning"] = f"Found {len(file_ids)} files for session {session_id}"
            
            self.log_execution(state, f"Found {len(file_ids)} files")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)

# Factory function for creating the node
def list_files_node():
    """Create a list files node instance"""
    return ListFilesNode()



