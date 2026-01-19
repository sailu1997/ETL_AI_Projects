"""
File Selector Node

This node handles file selection using the same simple approach as the old backend:
- No scoring or complex selection logic
- Agent directly specifies file_ids to use
- Simple file loading by ID
- Works with multiple files when needed
"""

import os
import pandas as pd
from typing import Dict, Any, List
import pandas as pd
from langchain_core.messages import HumanMessage
from .base_node import BaseNode
from state import ConversationState


class FileSelectorNode(BaseNode):
    """Node for loading files and intelligently selecting the most relevant one based on metadata."""
    
    def __init__(self):
        super().__init__(
            name="file_selector",
            description="Intelligently selects the most relevant file based on user query and file metadata"
        )
    
    def get_required_fields(self) -> List[str]:
        """Return the required fields for this node."""
        return ["session_id", "user_query"]
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Load available files and intelligently select the most relevant one based on metadata."""
        session_id = state.get("session_id")
        user_query = state.get("user_query")
        
        if not session_id:
            state["error"] = "Session ID is required."
            return state
        
        print(f"DEBUG: FileSelector - Loading files for session: {session_id}")
        
        try:
            # Get available files for this session from database
            from database import db_manager
            available_files = db_manager.get_user_files(session_id)
            
            if not available_files:
                state["error"] = "No files found for this session. Please upload a file first."
                state["result"] = {
                    "query_type": "error",
                    "message": "No files available for analysis. Please upload a CSV file first."
                }
                return state
            
            print(f"DEBUG: FileSelector - Found {len(available_files)} files")
            
            # Filter for cleaned files only
            cleaned_files = [f for f in available_files if f.get('is_cleaned', False)]
            if not cleaned_files:
                state["error"] = "No cleaned files found. Please wait for file processing to complete."
                return state
            
            # Analyze each file's metadata to find the most relevant one
            best_file = self._select_best_file(cleaned_files, user_query)
            
            if not best_file:
                state["error"] = "No suitable files found for your query."
                return state
            
            # Store the selected file info
            state["selected_file"] = {
                "file_id": best_file["file_id"],
                "filename": best_file["original_filename"],
                "cleaned_filename": best_file.get("cleaned_filename"),
                "metadata": best_file.get("metadata", {}),
                "is_cleaned": best_file.get("is_cleaned", False)
            }
            
            # Update file_ids to contain only the selected file
            state["file_ids"] = [best_file["file_id"]]
            
            print(f"DEBUG: FileSelector - Selected file: {best_file['original_filename']} (ID: {best_file['file_id']})")
            
            state["result"] = {
                "message": f"Selected file '{best_file['original_filename']}' based on your query: '{user_query}'",
                "selected_file": state["selected_file"]
            }
            
            return state
            
        except Exception as e:
            print(f"ERROR: FileSelector failed: {str(e)}")
            state["error"] = f"Failed to load files: {str(e)}"
            return state
    
    def _select_best_file(self, files: List[Dict], query: str) -> Dict:
        """Select the best file based on query relevance using metadata."""
        query_lower = query.lower()
        
        # Define query categories and their relevant column patterns
        query_patterns = {
            'financial': {
                'keywords': ['rent', 'price', 'cost', 'amount', 'salary', 'income', 'revenue', 'expense'],
                'columns': ['rent', 'price', 'cost', 'amount', 'salary', 'income', 'revenue', 'expense', 'monthly_rent', 'price_per_sqft']
            },
            'property': {
                'keywords': ['flat', 'apartment', 'house', 'property', 'building', 'room', 'bedroom'],
                'columns': ['flat_type', 'apartment_type', 'property_type', 'building_type', 'type', 'room_count', 'bedroom_count']
            },
            'location': {
                'keywords': ['location', 'area', 'city', 'state', 'country', 'region', 'address', 'zipcode'],
                'columns': ['location', 'city', 'state', 'country', 'region', 'area', 'address', 'zipcode']
            },
            'temporal': {
                'keywords': ['monthly', 'yearly', 'annual', 'weekly', 'daily', 'date', 'time'],
                'columns': ['month', 'year', 'date', 'time', 'period', 'quarter', 'week']
            },
            'demographic': {
                'keywords': ['age', 'gender', 'education', 'occupation', 'marital'],
                'columns': ['age', 'gender', 'education', 'occupation', 'marital_status']
            }
        }
        
        best_score = 0
        best_file = None
        
        for file_info in files:
            metadata = file_info.get('metadata', {})
            if not metadata or 'columns' not in metadata:
                continue
            
            file_columns = [col.lower() for col in metadata['columns']]
            score = 0
            
            # Score based on query relevance
            for category, patterns in query_patterns.items():
                # Check if query contains category keywords
                if any(keyword in query_lower for keyword in patterns['keywords']):
                    # Check if file has relevant columns
                    relevant_columns = [col for col in file_columns if col in patterns['columns']]
                    if relevant_columns:
                        score += len(relevant_columns) * 2  # Bonus for multiple relevant columns
                        score += 1  # Base score for category match
            
            # Additional scoring factors
            if metadata.get('row_count', 0) > 100:  # Prefer files with more data
                score += 1
            
            if file_info.get('is_cleaned', False):  # Prefer cleaned files
                score += 1
            
            print(f"DEBUG: FileSelector - File '{file_info['original_filename']}' score: {score}")
            
            if score > best_score:
                best_score = score
                best_file = file_info
        
        # If no specific match found, return the first cleaned file
        if not best_file and files:
            best_file = files[0]
            print(f"DEBUG: FileSelector - No specific match found, using first available file: {best_file['original_filename']}")
        
        return best_file

# Factory function for creating the node
def file_selector_node():
    """Create a file selector node instance"""
    return FileSelectorNode()
