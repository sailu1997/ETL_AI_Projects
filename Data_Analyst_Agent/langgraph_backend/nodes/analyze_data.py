"""
Data Analysis Node for LangGraph

This node provides intelligent data analysis and insights for uploaded files.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_node import BaseNode
from state import ConversationState
from database import DatabaseManager
from state import SimpleStateManager
import logging

class AnalyzeDataNode(BaseNode):
    """Simple file selection and routing node - no analysis, just selection"""
    
    def __init__(self):
        super().__init__(
            name="analyze_data",
            description="Selects the best file for analysis and routes to next node"
        )
        self.db_manager = DatabaseManager()
        self.session_manager = SimpleStateManager()
        self.logger = logging.getLogger(__name__)

    def get_required_fields(self) -> list:
        return ["session_id", "user_query", "selected_file"]

    def execute(self, state: ConversationState) -> ConversationState:
        """Select best file and route to next node with context"""
        try:
            print("=== DEBUG: AnalyzeData execute method called ===")
            print(f"DEBUG: AnalyzeData - Input state keys: {list(state.keys())}")
            print(f"DEBUG: AnalyzeData - Input state type: {type(state)}")
            session_id = state["session_id"]
            user_query = state["user_query"]
            query_classification = state.get("query_classification", {})
            print(f"DEBUG: AnalyzeData - Session ID: {session_id}")
            print(f"DEBUG: AnalyzeData - User query: {user_query}")
            print(f"DEBUG: AnalyzeData - Query classification: {query_classification}")
            
            # Get selected file from file_selector (now handles file selection)
            selected_file = state.get("selected_file")
            
            # If selected_file is not available, try to construct it from file_ids
            if not selected_file and state.get("file_ids"):
                file_ids = state["file_ids"]
                if file_ids:
                    # Try to get file info from database
                    try:
                        from database import db_manager
                        session_id = state.get("session_id")
                        file_id = file_ids[0]  # Use first file
                        file_info = db_manager.get_file_by_id(file_id, session_id)
                        
                        if file_info:
                            selected_file = {
                                "file_id": file_info["file_id"],
                                "filename": file_info["original_filename"],
                                "cleaned_filename": file_info.get("cleaned_filename"),
                                "metadata": file_info.get("metadata", {}),
                                "is_cleaned": file_info.get("is_cleaned", False)
                            }
                            print(f"DEBUG: AnalyzeData - Constructed selected_file from database: {selected_file}")
                    except Exception as e:
                        print(f"DEBUG: AnalyzeData - Failed to construct selected_file from database: {e}")
            
            if not selected_file:
                # Handle case when no files are available
                if not state.get("file_ids") or len(state.get("file_ids", [])) == 0:
                    state["error"] = "No files uploaded. Please upload a file first."
                    state["result"] = {
                        "query_type": "error",
                        "message": "No files available for analysis. Please upload a CSV file first."
                    }
                    return state
                else:
                    state["error"] = "No file selected. Please ensure file_selector has run first."
                    state["result"] = {
                        "query_type": "error"
                    }
                    return state
            
            selected_file_id = selected_file["file_id"]
            print(f"DEBUG: AnalyzeData - Working with selected file: {selected_file_id}")
            
            # Simplified: Skip complex database operations for now
            print(f"DEBUG: AnalyzeData - Skipping database operations for simplicity")
            state["conversation_memory"] = {}
            
            # Early exit for conversation memory queries
            if query_classification.get("query_type") == "conversation_memory":
                state["result"] = {
                    "message": "This appears to be a conversation memory query. Please ask a specific question about your data.",
                    "query_type": "conversation_memory"
                }
                state["error"] = None  # Clear any previous errors
                return state
            
            # selected_file is already validated above, so we can proceed
            
            # Use the selected file from file_selector
            print(f"DEBUG: AnalyzeData - Using selected file: {selected_file_id}")
            
            # Update state with selected file info for next node
            # CRITICAL: Update file_ids to contain only the selected file for smart_code_generator
            state["file_ids"] = [selected_file_id]
            print(f"DEBUG: AnalyzeData - Updated file_ids to: {state['file_ids']}")
            
            # CRITICAL: Put file metadata in conversation_context for SmartCodeGenerator to access
            file_metadata = {
                selected_file_id: {
                    "file_id": selected_file_id,
                    "filename": selected_file.get("filename", "Unknown"),
                    "shape": (selected_file.get("metadata", {}).get("row_count", 0), selected_file.get("metadata", {}).get("column_count", 0)),
                    "columns": selected_file.get("metadata", {}).get("columns", []),
                    "dtypes": selected_file.get("metadata", {}).get("dtypes", {}),
                    "null_counts": {},  # Database doesn't store this yet
                    "head_sample": selected_file.get("metadata", {}).get("sample_data", []),
                    "file_path": selected_file.get("cleaned_filename", selected_file.get("uploaded_filename", "")),
                    "file_type": "csv"
                }
            }
            
            # Also put it in top-level state for backward compatibility
            state["file_metadata"] = file_metadata
            
            state["result"] = {
                "message": f"File loaded and ready for analysis: {selected_file_id}",
                "query_type": query_classification.get("query_type", "general_analysis"),
                "selected_file": selected_file,  # Use the full selected_file info
                "conversation_context": {
                    "user_query": user_query,
                    "query_classification": query_classification,
                    "session_id": session_id,
                    "previous_messages": state.get("messages", []),
                    "is_follow_up": query_classification.get("is_follow_up", False),
                    "context_clues": query_classification.get("context_clues", ""),
                    "previous_analysis": self._extract_previous_analysis_context(state),
                    "data_relationships": self._extract_data_relationships(state, None),
                    "file_metadata": file_metadata  # CRITICAL: Add file metadata here
                }
            }
            state["error"] = None  # Clear any previous errors
            
            self.log_execution(state, f"File selected for query: {user_query}")
            print(f"DEBUG: AnalyzeData - Execution completed successfully")
            print(f"DEBUG: AnalyzeData - Final state keys: {list(state.keys())}")
            return state
            
        except Exception as e:
            error_msg = f"Error in AnalyzeDataNode: {str(e)}"
            self.logger.error(error_msg)
            return self.handle_error(state, e)

    def _select_best_file(self, file_ids: List[str], query_classification: Dict[str, Any] = None, state: ConversationState = None) -> Optional[Dict[str, Any]]:
        """Select the best file for the query using simple criteria"""
        try:
            print(f"DEBUG: AnalyzeData - _select_best_file called with {len(file_ids)} file_ids: {file_ids}")
            
            # Get files from session manager instead of database
            session_id = state.get("session_id") if state else None
            if not session_id:
                print(f"DEBUG: AnalyzeData - No session_id in state")
                return None
            
            files = self.session_manager.get_session_files(session_id)
            print(f"DEBUG: AnalyzeData - Files from session manager: {files}")
            
            # Filter to only the files we're interested in
            relevant_files = [f for f in files if f.get('file_id') in file_ids]
            print(f"DEBUG: AnalyzeData - Relevant files: {len(relevant_files)}")
            
            if not relevant_files:
                print(f"DEBUG: AnalyzeData - No relevant files found, returning None")
                return None
            
            # Simple selection: prefer cleaned files, then most recent
            cleaned_files = [f for f in relevant_files if f.get('is_cleaned', False)]
            print(f"DEBUG: AnalyzeData - Cleaned files: {len(cleaned_files)}")
            
            if cleaned_files:
                # Return most recent cleaned file
                selected = max(cleaned_files, key=lambda x: x.get('upload_timestamp', ''))
                print(f"DEBUG: AnalyzeData - Selected cleaned file: {selected}")
                return selected
            else:
                # Return most recent file if no cleaned files
                selected = max(relevant_files, key=lambda x: x.get('upload_timestamp', ''))
                print(f"DEBUG: AnalyzeData - Selected most recent file: {selected}")
                return selected
            
        except Exception as e:
            error_msg = f"Error selecting file: {str(e)}"
            self.logger.error(error_msg)
            print(f"DEBUG: AnalyzeData - Error in _select_best_file: {error_msg}")
            # Don't handle error here - let the main execute method handle it
            # Just return None to indicate failure
            return None

    def _extract_previous_analysis_context(self, state: ConversationState) -> Dict[str, Any]:
        """Extract context from previous analysis in the conversation"""
        try:
            previous_context = {}
            
            # First, check conversation memory in state
            conversation_memory = state.get("conversation_memory", {})
            if conversation_memory and conversation_memory.get("last_analysis"):
                last_analysis = conversation_memory["last_analysis"]
                previous_context.update({
                    "user_query": last_analysis.get("user_query", ""),
                    "query_type": last_analysis.get("query_type", ""),
                    "data_relationship": last_analysis.get("data_relationship", ""),
                    "columns_analyzed": last_analysis.get("columns_analyzed", []),
                    "visualization_type": last_analysis.get("visualization_type", ""),
                    "result_type": last_analysis.get("result_type", ""),
                    "timestamp": last_analysis.get("timestamp", "")
                })
                
                # Extract insights if available
                if last_analysis.get("insights"):
                    previous_context["insights"] = last_analysis["insights"]
                
                print(f"DEBUG: AnalyzeData - Found conversation memory: {previous_context}")
                return previous_context
            
            # Fallback: Look for previous analysis in messages
            messages = state.get("messages", [])
            for i, msg in enumerate(messages):
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    content = msg.content.lower()
                    
                    # Look for analysis patterns
                    if "plot" in content or "chart" in content or "graph" in content:
                        previous_context["visualization_type"] = self._extract_plot_type(content)
                    
                    # PRIORITIZE flat_type analysis over year analysis
                    if "flat_type" in content and "monthly_rent" in content:
                        previous_context["data_relationship"] = "monthly_rent vs flat_type"
                        previous_context["columns_analyzed"] = ["monthly_rent", "flat_type"]
                        previous_context["categorical_analysis"] = "flat_type"
                        previous_context["primary_analysis"] = "flat_type_rent_analysis"
                    elif "monthly_rent" in content and "year" in content:
                        previous_context["data_relationship"] = "monthly_rent vs year"
                        previous_context["columns_analyzed"] = ["monthly_rent", "year"]
                    
                    if "region" in content:
                        previous_context["geographic_analysis"] = "region"
            
            # Look for previous results in state
            if state.get("result") and "execution_result" in state["result"]:
                exec_result = state["result"]["execution_result"]
                if exec_result.get("success") and "result" in exec_result:
                    result_data = exec_result["result"]
                    if result_data.get("type") == "dataframe":
                        previous_context["previous_result_type"] = "dataframe"
                        previous_context["previous_result_shape"] = result_data.get("shape")
                        previous_context["previous_result_columns"] = result_data.get("columns")
            
            return previous_context
            
        except Exception as e:
            self.logger.warning(f"Error extracting previous analysis context: {e}")
            return {}

    def _extract_data_relationships(self, state: ConversationState, selected_file: Dict[str, Any]) -> Dict[str, Any]:
        """Extract potential data relationships from file metadata"""
        try:
            relationships = {}
            
            # Get column metadata
            columns = selected_file.get("metadata", {}).get("column_names", [])
            
            # Identify potential relationships
            if "monthly_rent" in columns:
                relationships["rent_analysis"] = {
                    "primary_column": "monthly_rent",
                    "potential_relationships": []
                }
                
                # Look for time-related columns
                time_columns = [col for col in columns if any(time_word in col.lower() for time_word in ["date", "year", "month", "time"])]
                if time_columns:
                    relationships["rent_analysis"]["potential_relationships"].extend([
                        f"monthly_rent vs {col}" for col in time_columns
                    ])
                
                # Look for categorical columns
                categorical_columns = [col for col in columns if col not in ["monthly_rent"] + time_columns]
                if categorical_columns:
                    relationships["rent_analysis"]["potential_relationships"].extend([
                        f"monthly_rent by {col}" for col in categorical_columns
                    ])
            
            # Geographic relationships
            geo_columns = [col for col in columns if col in ["region", "town", "subzone", "planning_area"]]
            if geo_columns:
                relationships["geographic_analysis"] = {
                    "columns": geo_columns,
                    "potential_relationships": [f"analysis by {col}" for col in geo_columns]
                }
            
            # Property characteristics
            property_columns = [col for col in columns if col in ["flat_type", "flat_model", "floor_area_sqm", "furnished"]]
            if property_columns:
                relationships["property_analysis"] = {
                    "columns": property_columns,
                    "potential_relationships": [f"analysis by {col}" for col in property_columns]
                }
            
            return relationships
            
        except Exception as e:
            self.logger.warning(f"Error extracting data relationships: {e}")
            return {}

    def _extract_plot_type(self, content: str) -> str:
        """Extract plot type from content"""
        content_lower = content.lower()
        if "line" in content_lower:
            return "line"
        elif "bar" in content_lower:
            return "bar"
        elif "scatter" in content_lower:
            return "scatter"
        elif "histogram" in content_lower:
            return "histogram"
        elif "box" in content_lower:
            return "box"
        else:
            return "plot"

def analyze_data_node():
    return AnalyzeDataNode() 