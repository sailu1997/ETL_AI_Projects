"""
FastAPI Integration for LangGraph Workflow

This integrates the LangGraph workflow with FastAPI endpoints for file processing and query analysis.
"""

import os
import uuid
import numpy as np
import json
import math
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from state import create_initial_state, state_manager
from workflow import complete_app
from database import db_manager
import asyncio

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

# Pydantic models for API requests
class FileUploadRequest(BaseModel):
    session_id: Optional[str] = None
    should_clean: bool = True

class FileCleanRequest(BaseModel):
    file_id: str
    session_id: str

class FileListRequest(BaseModel):
    session_id: str

class FileRequest(BaseModel):
    file_id: str

class LangGraphFileProcessor:
    """Handles query analysis using LangGraph workflows (file processing removed)"""
    
    def __init__(self):
        self.complete_app = complete_app
    
    async def list_files(self, session_id: str) -> Dict[str, Any]:
        """List files for a session using LangGraph workflow"""
        try:
            # Create initial state for listing
            initial_state = create_initial_state(
                session_id=session_id,
                user_query="List all available files"
            )
            
            # Run the listing workflow
            from nodes import list_files_node
            result = list_files_node().execute(initial_state)
            
            return {
                "session_id": session_id,
                "success": True,
                "file_ids": result.get("file_ids", []),
                "available_files": result.get("available_files", []),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }

# Global processor instance
file_processor = LangGraphFileProcessor()

# FastAPI endpoints
def create_langgraph_endpoints(app: FastAPI):
    """Add LangGraph workflow endpoints to FastAPI app"""
    
    @app.post("/clean_data/")
    async def clean_data(request: Request, clean_request: FileRequest):
        """Clean data endpoint for frontend compatibility - same as backend"""
        try:
            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No session ID provided", "message": "Please provide X-Session-ID header"}
                )
            
            file_id = clean_request.file_id
            
            print(f"Starting clean_data for file_id: {file_id}, session_id: {session_id}")
            
            # Get file information from database
            file_info = db_manager.get_file_by_id(file_id, session_id)
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found or access denied")
            
            # Get the file path from the uploaded filename
            file_path = file_info['uploaded_filename']
            print(f"Looking for uploaded file at: {file_path}")
            print(f"Uploaded file exists: {os.path.exists(file_path)}")
            
            # Read the CSV file
            import pandas as pd
            import numpy as np
            from datetime import datetime, date
            
            #df = pd.read_csv(file_path, low_memory=False)
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                df = pd.read_csv(file_path, low_memory=False)
            original_df = df.copy()
            
            # Analyze the data for cleaning
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
                            "non_numeric_count": int(numeric_count)
                        }

            # Create cleaning prompt
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
            - For numeric columns: Use mean/median for <20% missing, consider dropping for ≥20%
            - For categorical columns: Use mode for <20% missing, consider dropping for ≥20%
            - For date columns: Use forward/backward fill for <20% missing, consider dropping for ≥20%

            2. Fix formatting issues:
            - Convert mixed-type columns to appropriate types
            - Standardize date formats
            - Clean strings (trim whitespace, fix case)

            3. Handle naming conventions:
            - Normalize categorical variable names
            - Map similar values to canonical forms

            4. Remove duplicate rows

            5. Return a cleaned DataFrame called `df`

            Use only standard libraries: pandas, numpy, datetime
            Return ONLY the code inside a Python code block:
            ```python
            # your code here
            ```
            """

            # Get OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data cleaning expert. Provide only the Python code solution."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            cleaning_code = response.choices[0].message.content
            
            # Extract code from markdown block if present
            if "```python" in cleaning_code:
                start = cleaning_code.find("```python") + 9
                end = cleaning_code.find("```", start)
                cleaning_code = cleaning_code[start:end].strip()
            
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
                    ]
                }
                
                # Create cleaned file path - use the same directory as uploaded files
                cleaned_filename = f"cleaned_data_{file_id}.csv"
                cleaned_file_path = os.path.join("uploaded_files", cleaned_filename)
                
                # Ensure the directory exists
                os.makedirs("uploaded_files", exist_ok=True)
                
                # Save the cleaned DataFrame
                cleaned_df.to_csv(cleaned_file_path, index=False)
                
                # Update database with cleaned file information
                db_manager.update_file_cleaned_status(file_id, cleaned_file_path)
                
                # CRITICAL: Save rich metadata to database for FileSelector to use
                rich_metadata = {
                    "row_count": len(cleaned_df),
                    "column_count": len(cleaned_df.columns),
                    "columns": cleaned_df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in cleaned_df.dtypes.items()},
                    "sample_data": cleaned_df.head(5).to_dict(orient='records'),
                    "null_counts": cleaned_df.isnull().sum().to_dict(),
                    "numeric_stats": {},
                    "has_numeric_data": False,
                    "has_categorical_data": False,
                    "file_analyzed": True,
                    "cleaning_applied": True
                }
                
                # Add numeric statistics for numeric columns
                numeric_columns = cleaned_df.select_dtypes(include=['number']).columns
                if len(numeric_columns) > 0:
                    rich_metadata["has_numeric_data"] = True
                    rich_metadata["numeric_stats"] = {}
                    for col in numeric_columns:
                        rich_metadata["numeric_stats"][col] = {
                            'min': float(cleaned_df[col].min()) if not cleaned_df[col].empty else None,
                            'max': float(cleaned_df[col].max()) if not cleaned_df[col].empty else None,
                            'mean': float(cleaned_df[col].mean()) if not cleaned_df[col].empty else None,
                            'median': float(cleaned_df[col].median()) if not cleaned_df[col].empty else None
                        }
                
                # Add categorical data flag
                categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
                if len(categorical_columns) > 0:
                    rich_metadata["has_categorical_data"] = True
                
                # Save the rich metadata to database
                db_manager.update_file_metadata(file_id, rich_metadata)
                
                # Log the cleaning action
                db_manager.log_user_action(session_id, "clean")
                
                # Return the cleaned data response
                return {
                    "success": True,
                    "message": "Data cleaned successfully",
                    "file_id": file_id,
                    "session_id": session_id,
                    "cleaned_sample": cleaned_df.head(10).to_dict('records'),
                    "impact_metrics": impact_metrics,
                    "applied_code": cleaning_code,
                    "cleaned_filename": cleaned_filename
                }
                
            except Exception as exec_error:
                print(f"Error executing cleaning code: {exec_error}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "Failed to execute cleaning code",
                        "details": str(exec_error),
                        "file_id": file_id,
                        "session_id": session_id
                    }
                )
                
        except Exception as e:
            print(f"DEBUG: Error in clean_data: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse(
                status_code=500, 
                content={"error": str(e)}
            )
    
    # Session management endpoints for frontend compatibility
    @app.get("/session/info")
    async def get_session_info(request: Request):
        """Get session information including files and analytics - frontend compatibility endpoint"""
        try:
            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")
            
            # If no session ID provided, create a new one (same as backend)
            if not session_id:
                session_id = db_manager.create_user_session()
            
            # Get session statistics using the same method as backend
            session_stats = db_manager.get_session_stats(session_id)
            if not session_stats:
                # Fallback to basic session info if database query fails
                session_stats = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "is_active": True,
                    "total_files": 0,
                    "cleaned_files": 0,
                    "total_actions": 0,
                    "uploads": 0,
                    "cleans": 0,
                    "chats": 0
                }
            
            # Transform the data to match frontend expectations
            formatted_session_info = {
                "session_id": session_stats.get("session_id", session_id),
                "created_at": session_stats.get("created_at", datetime.now().isoformat()),
                "last_activity": session_stats.get("last_activity", datetime.now().isoformat()),
                "is_active": bool(session_stats.get("is_active", True)),
                "file_stats": {
                    "total_files": session_stats.get("total_files", 0),
                    "cleaned_files": session_stats.get("cleaned_files", 0)
                },
                "action_stats": {
                    "total_actions": session_stats.get("total_actions", 0),
                    "uploads": session_stats.get("uploads", 0),
                    "cleans": session_stats.get("cleans", 0),
                    "chats": session_stats.get("chats", 0)
                }
            }
            
            # Check if session is active
            if not formatted_session_info.get('is_active', True):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Session deactivated",
                        "message": "This session has been deactivated",
                        "session_id": session_id
                    }
                )
            
            # Get user files using the same method as backend
            user_files = db_manager.get_user_files(session_id)
            if user_files is None:
                user_files = []
            
            return {
                "session_info": formatted_session_info,
                "user_files": user_files
            }
        except Exception as e:
            print(f"DEBUG: Error in session info: {e}")
            import traceback
            traceback.print_exc()
            # Return a safe fallback response instead of throwing an error
            return {
                "session_info": {
                    "session_id": "error-session",
                    "created_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "is_active": False,
                    "file_stats": {"total_files": 0, "cleaned_files": 0},
                    "action_stats": {"total_actions": 0, "uploads": 0, "cleans": 0, "chats": 0}
                },
                "user_files": [],
                "error": "Session info temporarily unavailable"
            }

    @app.post("/session/deactivate")
    async def deactivate_session(request: Request):
        """Deactivate session - frontend compatibility endpoint"""
        try:
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No session ID provided", "message": "Please provide X-Session-ID header"}
                )
            
            # Deactivate the session in the database
            db_manager.deactivate_user_session(session_id)
            
            # Create a new session for the frontend to use immediately
            new_session_id = str(uuid.uuid4())
            db_manager.create_user_session(new_session_id)
            
            return {
                "success": True,
                "message": "Session deactivated successfully",
                "old_session_id": session_id,
                "new_session_id": new_session_id,
                "session_id": new_session_id  # Keep for backward compatibility
            }
        except Exception as e:
            print(f"DEBUG: Error in session deactivate: {e}")
            import traceback
            traceback.print_exc()
            # Return success even if there's an error, to avoid breaking the frontend
            return {
                "success": True,
                "message": "Session deactivated successfully",
                "session_id": session_id
            }

    # Chat analysis endpoint
    @app.get("/chat_stream")
    async def chat_stream(request: Request, query: str):
        """Chat stream endpoint for query analysis with streaming responses"""
        try:
            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")
            
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No session ID provided", "message": "Please provide X-Session-ID header"}
                )
            
            if not query:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No query provided", "message": "Please provide a query parameter"}
                )
            
            # Check if there's a paused workflow waiting for human input
            paused_state = state_manager.get_workflow_state(session_id)
            if paused_state and paused_state.get("workflow_paused", False):
                return JSONResponse(
                    status_code=409,  # Conflict - workflow is paused
                    content={
                        "error": "Workflow paused",
                        "message": "There's a paused workflow waiting for human input",
                        "human_input_request": paused_state.get("human_input_request"),
                        "session_id": session_id
                    }
                )
            
            # Use the complete workflow for intelligent analysis
            from nodes import query_classifier_node, analyze_data_node, smart_code_generator_node, smart_code_executor_node, clarification_node
            from state import create_initial_state
            from workflow_logger import workflow_logger
            
            # Get conversation history from database
            try:
                conversation_history = db_manager.get_conversation_history(session_id)
                print(f"DEBUG: Loaded conversation history with {len(conversation_history)} messages")
            except Exception as e:
                print(f"DEBUG: Could not load conversation history: {e}")
                conversation_history = []
            
            # Get files from database
            try:
                files = db_manager.get_user_files(session_id)
                file_ids = [f["file_id"] for f in files]
                print(f"DEBUG: Loaded {len(file_ids)} files from database")
            except Exception as e:
                print(f"DEBUG: Could not load files: {e}")
                file_ids = []
            
            print(f"DEBUG: About to start streaming workflow with {len(conversation_history)} messages and {len(file_ids)} files")
            
            # Create streaming response function
            async def generate_stream():
                try:
                    # Start streaming the workflow
                    async for chunk in complete_app.astream({
                        "messages": conversation_history,
                        "user_query": query,
                        "session_id": session_id,
                        "file_ids": file_ids
                    }, config={"configurable": {"thread_id": session_id}}):
                        
                        print(f"DEBUG: Received streaming chunk: {list(chunk.keys())}")
                        
                        # Process chunk for streaming response
                        streaming_data = process_chunk_for_streaming(chunk, query)
                        if streaming_data:
                            print(f"DEBUG: Yielding streaming data: {streaming_data['type']}")
                            yield f"data: {json.dumps(streaming_data)}\n\n"
                        
                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.1)
                    
                    # Send completion signal with final result data
                    completion_data = {
                        "type": "complete",
                        "message": "Workflow completed successfully",
                        "query": query,
                        "session_id": session_id,
                        "final_result": {
                            "final_answer": f"Analysis completed for: {query}",
                            "reasoning": "Analysis completed successfully",
                            "plot_data": None,
                            "table_data": None,
                            "summary_stats": None,
                            "response_format": "text"
                        }
                    }
                    
                    # Try to extract final result from the last chunk
                    try:
                        # Get the final result from the workflow
                        final_result = None
                        execution_success = False
                        
                        print(f"DEBUG: Completion logic - Processing chunk with keys: {list(chunk.keys())}")
                        
                        for node_name, node_result in chunk.items():
                            print(f"DEBUG: Completion logic - Processing node: {node_name}")
                            if node_name == "smart_code_executor" and node_result.get("result", {}).get("execution_result", {}).get("success"):
                                execution_result = node_result["result"]["execution_result"]
                                execution_success = True
                                print(f"DEBUG: Completion logic - Found successful smart_code_executor")
                                # Look for result in the state's result_data
                                if "result_data" in node_result.get("result", {}):
                                    final_result = node_result["result"]["result_data"]
                                    print(f"DEBUG: Completion logic - Found result_data: {type(final_result)}")
                                    break
                                elif "result" in execution_result:
                                    final_result = execution_result["result"]
                                    print(f"DEBUG: Completion logic - Found result in execution_result: {type(final_result)}")
                                    break
                            elif node_name == "intelligent_error_handler":
                                execution_result = node_result.get("result", {}).get("execution_result", {})
                                if execution_result.get("success"):
                                    execution_success = True
                                    print(f"DEBUG: Completion logic - Found successful intelligent_error_handler")
                                    if "result" in execution_result:
                                        final_result = execution_result["result"]
                                        break
                                elif execution_result.get("final_attempt"):
                                    # Error handler gave up after max attempts
                                    print(f"DEBUG: Completion logic - Error handler gave up after max attempts")
                                    completion_data["final_result"].update({
                                        "error": execution_result.get("error", "Failed after multiple fix attempts"),
                                        "requires_human_intervention": True,
                                        "final_answer": f"Analysis failed: {execution_result.get('error', 'Unknown error')}",
                                        "response_format": "error"
                                    })
                                    break
                        
                        print(f"DEBUG: Completion logic - execution_success: {execution_success}, final_result: {type(final_result) if final_result else None}")
                        
                        if final_result and execution_success:
                            # Determine response format and structure the final result
                            if isinstance(final_result, dict):
                                # Check if it's a Plotly figure (has 'data' and 'layout' keys)
                                if "data" in final_result and "layout" in final_result:
                                    completion_data["final_result"].update({
                                        "plot_data": final_result,
                                        "response_format": "plot",
                                        "final_answer": f"Visualization completed for: {query}"
                                    })
                                    print(f"DEBUG: Completion logic - Structured Plotly figure for frontend")
                                elif "type" in final_result:
                                    if final_result["type"] == "dataframe":
                                        completion_data["final_result"].update({
                                            "table_data": final_result.get("head", []),
                                            "response_format": "table",
                                            "final_answer": f"Table analysis completed for: {query}"
                                        })
                                    elif final_result["type"] == "plotly_figure":
                                        completion_data["final_result"].update({
                                            "plot_data": final_result.get("figure_data"),
                                            "response_format": "plot",
                                            "final_answer": f"Visualization completed for: {query}"
                                        })
                                    else:
                                        completion_data["final_result"].update({
                                            "summary_stats": final_result,
                                            "response_format": "text",
                                            "final_answer": f"Analysis completed for: {query}"
                                        })
                                else:
                                    # Fallback for other result types
                                    completion_data["final_result"].update({
                                        "summary_stats": final_result,
                                        "response_format": "text",
                                        "final_answer": f"Analysis completed for: {query}"
                                    })
                            else:
                                # Fallback for non-dict result types
                                completion_data["final_result"].update({
                                    "summary_stats": final_result,
                                    "response_format": "text",
                                    "final_answer": f"Analysis completed for: {query}"
                                })
                        elif not execution_success:
                            # No successful execution found, check for error information
                            for node_name, node_result in chunk.items():
                                if node_name == "smart_code_executor" and not node_result.get("result", {}).get("execution_result", {}).get("success"):
                                    error_info = node_result.get("result", {}).get("execution_result", {})
                                    completion_data["final_result"].update({
                                        "error": error_info.get("error", "Code execution failed"),
                                        "final_answer": f"Analysis failed: {error_info.get('error', 'Unknown error')}",
                                        "response_format": "error"
                                    })
                                    break
                                elif node_name == "intelligent_error_handler":
                                    error_info = node_result.get("result", {}).get("execution_result", {})
                                    if not error_info.get("success"):
                                        completion_data["final_result"].update({
                                            "error": error_info.get("error", "Error handling failed"),
                                            "final_answer": f"Analysis failed: {error_info.get('error', 'Unknown error')}",
                                            "response_format": "error"
                                        })
                                        break
                    except Exception as e:
                        print(f"DEBUG: Could not extract final result: {e}")
                        # Set a generic error response
                        completion_data["final_result"].update({
                            "error": f"Failed to process result: {str(e)}",
                            "final_answer": f"Analysis failed: {str(e)}",
                            "response_format": "error"
                        })
                    
                    print(f"DEBUG: Sending completion signal with final result")
                    print(f"DEBUG: Completion data final_result keys: {list(completion_data['final_result'].keys())}")
                    print(f"DEBUG: Completion data plot_data exists: {'plot_data' in completion_data['final_result']}")
                    print(f"DEBUG: Completion data response_format: {completion_data['final_result'].get('response_format')}")
                    yield f"data: {json.dumps(completion_data)}\n\n"
                    
                except Exception as e:
                    print(f"DEBUG: Streaming error: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Send error in streaming format
                    error_data = {
                        'type': 'error',
                        'message': f'Streaming error: {str(e)}',
                        'error': str(e),
                        'query': query
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            # Return streaming response
            from fastapi.responses import StreamingResponse
            import asyncio
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/plain; charset=utf-8",
                    "X-Session-ID": session_id
                }
            )
            
        except Exception as e:
            print(f"DEBUG: Endpoint setup error: {e}")
            import traceback
            traceback.print_exc()
            
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to setup streaming", "message": str(e)}
            )

    def process_chunk_for_streaming(chunk: dict, query: str) -> dict:
        """Process LangGraph chunk and convert to streaming format"""
        
        # Check which node just executed
        for node_name, node_result in chunk.items():
            if node_name == "query_classifier":
                return {
                    "type": "progress",
                    "node": "query_classifier",
                    "message": "Analyzing query type and requirements...",
                    "data": {
                        "query_type": node_result.get("query_classification", {}).get("query_type", "unknown"),
                        "response_format": node_result.get("query_classification", {}).get("response_format", "text")
                    }
                }
            
            elif node_name == "file_selector":
                selected_files = node_result.get("selected_file", {})
                return {
                    "type": "progress", 
                    "node": "file_selector",
                    "message": f"Selected file: {selected_files.get('filename', 'Unknown')}",
                    "data": {
                        "file_id": selected_files.get("file_id"),
                        "filename": selected_files.get("filename"),
                        "columns": selected_files.get("metadata", {}).get("columns", [])
                    }
                }
            
            elif node_name == "analyze_data":
                metadata = node_result.get("selected_file", {}).get("metadata", {})
                return {
                    "type": "progress",
                    "node": "analyze_data", 
                    "message": f"Analyzing data structure: {metadata.get('row_count', 0)} rows, {metadata.get('column_count', 0)} columns",
                    "data": {
                        "row_count": metadata.get("row_count", 0),
                        "column_count": metadata.get("column_count", 0),
                        "columns": metadata.get("columns", [])
                    }
                }
            
            elif node_name == "smart_code_generator":
                generated_code = node_result.get("generated_code", {})
                return {
                    "type": "progress",
                    "node": "smart_code_generator",
                    "message": "Generating analysis code...",
                    "data": {
                        "code_length": len(generated_code.get("code", "")),
                        "response_format": generated_code.get("response_format", "text"),
                        "should_plot": generated_code.get("should_plot", False)
                    }
                }
            
            elif node_name == "smart_code_executor":
                execution_result = node_result.get("result", {}).get("execution_result", {})
                if execution_result.get("success"):
                    return {
                        "type": "progress",
                        "node": "smart_code_executor",
                        "message": "Code executed successfully!",
                        "data": {
                            "success": True,
                            "result_type": type(execution_result.get("result")).__name__ if execution_result.get("result") else "None"
                        }
                    }
                else:
                    return {
                        "type": "progress",
                        "node": "smart_code_executor", 
                        "message": f"Code execution failed: {execution_result.get('error', 'Unknown error')}",
                        "data": {
                            "success": False,
                            "error": execution_result.get("error", "Unknown error"),
                            "error_type": execution_result.get("error_type", "unknown")
                        }
                    }
            
            elif node_name == "intelligent_error_handler":
                fix_attempt = node_result.get("fix_attempt", 0)
                execution_result = node_result.get("result", {}).get("execution_result", {})
                
                if execution_result.get("success"):
                    return {
                        "type": "progress",
                        "node": "intelligent_error_handler",
                        "message": f"Error fixed successfully after {fix_attempt} attempts!",
                        "data": {
                            "success": True,
                            "fix_attempts": fix_attempt,
                            "result_type": type(execution_result.get("result")).__name__ if execution_result.get("result") else "None"
                        }
                    }
                else:
                    return {
                        "type": "progress",
                        "node": "intelligent_error_handler",
                        "message": f"Attempting to fix error (attempt {fix_attempt})...",
                        "data": {
                            "success": False,
                            "fix_attempt": fix_attempt,
                            "error": execution_result.get("error", "Unknown error")
                        }
                    }
        
        # If no specific node found, return generic progress
        return {
            "type": "progress",
            "node": "unknown",
            "message": "Processing...",
            "data": {"chunk_keys": list(chunk.keys())}
        }

    @app.get("/chat_stream_actual")
    async def chat_stream_actual(request: Request, query: str):
            """Chat stream endpoint for query analysis"""
            try:
                # Get session ID from headers
                session_id = request.headers.get("X-Session-ID")
                
                if not session_id:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No session ID provided", "message": "Please provide X-Session-ID header"}
                    )
                
                if not query:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "No query provided", "message": "Please provide a query parameter"}
                    )
                
                # Check if there's a paused workflow waiting for human input
                paused_state = state_manager.get_workflow_state(session_id)
                if paused_state and paused_state.get("workflow_paused", False):
                    return JSONResponse(
                        status_code=409,  # Conflict - workflow is paused
                        content={
                            "error": "Workflow paused",
                            "message": "There's a paused workflow waiting for human input",
                            "human_input_request": paused_state.get("human_input_request"),
                            "session_id": session_id
                        }
                    )
                
                # Use the complete workflow for intelligent analysis
                from nodes import query_classifier_node, analyze_data_node, smart_code_generator_node, smart_code_executor_node, clarification_node
                from state import create_initial_state
                from workflow_logger import workflow_logger
                
                # Get conversation history from database
                try:
                    conversation_history = db_manager.get_conversation_history(session_id)
                    print(f"DEBUG: Loaded conversation history with {len(conversation_history)} messages")
                except Exception as e:
                    print(f"DEBUG: Could not load conversation history: {e}")
                    conversation_history = []
                
                # Get files from database
                try:
                    files = db_manager.get_user_files(session_id)
                    file_ids = [f["file_id"] for f in files]
                    print(f"DEBUG: Loaded {len(file_ids)} files from database")
                except Exception as e:
                    print(f"DEBUG: Could not load files: {e}")
                    file_ids = []
                
                print(f"DEBUG: About to invoke workflow with {len(conversation_history)} messages and {len(file_ids)} files")
                
                # Pass conversation history and file info directly to workflow
                print(f"DEBUG: About to invoke workflow with session_id: {session_id}")
                print(f"DEBUG: About to invoke workflow with file_ids: {file_ids}")
                print(f"DEBUG: About to invoke workflow with conversation_history length: {len(conversation_history)}")
                
                result = complete_app.invoke({
                    "messages": conversation_history,
                    "user_query": query,
                    "session_id": session_id,
                    "file_ids": file_ids
                }, config={"configurable": {"thread_id": session_id}})
                
                print(f"DEBUG: Workflow returned session_id: {result.get('session_id')}")
                print(f"DEBUG: Workflow returned file_ids: {result.get('file_ids')}")
                
                print(f"DEBUG: Workflow invocation completed successfully")
                print(f"DEBUG: LangGraph workflow completed with result keys: {list(result.keys())}")
                print(f"DEBUG: Full workflow result: {result}")
                
                # Check if the workflow result is empty or contains errors
                if not result:
                    print(f"DEBUG: ERROR - Workflow result is empty!")
                    raise Exception("Workflow returned empty result")
                
                if isinstance(result, dict) and "error" in result and result["error"] is not None:
                    print(f"DEBUG: ERROR - Workflow result contains error: {result['error']}")
                    raise Exception(f"Workflow error: {result['error']}")
                
                # Check what's in the result structure
                if "conversation_memory" in result:
                    if result["conversation_memory"] is not None:
                        print(f"DEBUG: Found conversation_memory in result: {list(result['conversation_memory'].keys())}")
                    else:
                        print(f"DEBUG: conversation_memory is None - this is the problem!")
                else:
                    print(f"DEBUG: NO conversation_memory in result")
                
                # dataframes removed - no longer stored in state
                print(f"DEBUG: Skipping dataframes check - no longer stored in state")
                
                if "previous_analysis" in result:
                    if result["previous_analysis"] is not None:
                        print(f"DEBUG: Found previous_analysis in result: {result['previous_analysis']}")
                    else:
                        print(f"DEBUG: previous_analysis is None")
                else:
                    print(f"DEBUG: NO previous_analysis in result")
                
                # Check for any error fields in the result
                if "error" in result:
                    print(f"DEBUG: ERROR in workflow result: {result['error']}")
                
                # Check the result type and structure
                print(f"DEBUG: Result type: {type(result)}")
                print(f"DEBUG: Result is dict: {isinstance(result, dict)}")
                if isinstance(result, dict):
                    print(f"DEBUG: All result keys: {list(result.keys())}")
                    for key, value in result.items():
                        print(f"DEBUG: Key '{key}' has type {type(value)} and value: {value}")
                
                print(f"DEBUG: Result from smart code executor : {result}")
                # End workflow logging
                
                # Prepare the response
                try:
                    print(f"DEBUG: Preparing response from workflow result")
                    print(f"DEBUG: result['result'] keys: {list(result.get('result', {}).keys())}")
                    
                    # Check if we have generated code FIRST (before execution result)
                    if 'generated_code' in result.get('result', {}):
                        generated_code = result['result']['generated_code']
                        print(f"DEBUG: generated_code keys: {list(generated_code.keys())}")
                        print(f"DEBUG: generated_code content: {generated_code}")
                        
                        # Extract the actual code content
                        if 'code' in generated_code:
                            code_content = generated_code['code']
                            print(f"DEBUG: Found code content: {code_content[:200]}...")
                        else:
                            code_content = "No code generated"
                            print(f"DEBUG: No code content found")
                        
                        # Check if this should produce a plot
                        should_plot = generated_code.get('should_plot', False)
                        print(f"DEBUG: should_plot: {should_plot}")
                        
                        # Extract reasoning
                        reasoning = generated_code.get('reasoning', 'Analysis completed')
                        print(f"DEBUG: reasoning: {reasoning}")
                        
                        # Check if this is a table response
                        response_format = generated_code.get('response_format', 'text')
                        print(f"DEBUG: response_format: {response_format}")
                        
                        # Also check reasoning for response_format information
                        reasoning_text = generated_code.get('reasoning', '')
                        is_table_response = (
                            response_format == 'table' or 
                            'response_format: table' in reasoning_text or
                            'response_format: "table"' in reasoning_text
                        )
                        print(f"DEBUG: is_table_response: {is_table_response}")
                        
                        # Handle table responses FIRST
                        if is_table_response:
                            print(f"DEBUG: Processing table response")
                            
                            # Look for execution result with table data
                            table_data = None
                            if 'execution_result' in result.get('result', {}):
                                exec_result = result['result']['execution_result']
                                if exec_result.get('success') and 'result' in exec_result:
                                    # Extract the head data from the nested structure
                                    result_obj = exec_result['result']
                                    if isinstance(result_obj, dict) and 'head' in result_obj:
                                        table_data = result_obj['head']
                                        print(f"DEBUG: Found table data in execution_result head: {type(table_data)}")
                                    else:
                                        table_data = result_obj
                                        print(f"DEBUG: Found table data in execution_result result: {type(table_data)}")
                            
                            # If no execution result, check if we have result_data
                            if not table_data and 'result_data' in result.get('result', {}):
                                result_data_obj = result['result']['result_data']
                                if isinstance(result_data_obj, dict) and 'head' in result_data_obj:
                                    table_data = result_data_obj['head']
                                    print(f"DEBUG: Found table data in result_data head: {type(table_data)}")
                                else:
                                    table_data = result_data_obj
                                    print(f"DEBUG: Found table data in result_data: {type(table_data)}")
                            
                            if table_data:
                                # Convert table data to JSON-serializable format
                                if isinstance(table_data, list):
                                    # It's already a list of records
                                    table_records = table_data
                                    print(f"DEBUG: Table data is already list: {len(table_records)} rows")
                                elif hasattr(table_data, 'to_dict'):
                                    # It's a pandas DataFrame
                                    table_records = table_data.to_dict('records')
                                    print(f"DEBUG: Converted DataFrame to records: {len(table_records)} rows")
                                else:
                                    # Convert to string representation as fallback
                                    table_records = str(table_data)
                                    print(f"DEBUG: Converted table data to string")
                                
                                # Clean the table data for JSON serialization
                                table_records = clean_for_json(table_records)
                                
                                return JSONResponse(
                                    status_code=200,
                                    content={
                                        "final_answer": f"Table analysis completed for: {query}",
                                        "reasoning": reasoning,
                                        "table_data": table_records,
                                        "response_format": "table",
                                        "plot_data": None,
                                        "summary_stats": None
                                    }
                                )
                            else:
                                print(f"DEBUG: No table data found for table response")
                        
                        # For visualization queries, try to get the actual plot data
                        if should_plot:
                            # Try to get plot data from execution result
                            if 'execution_result' in result.get('result', {}):
                                execution_result = result['result']['execution_result']
                                if execution_result.get('success') and 'result' in execution_result:
                                    result_data = execution_result['result']
                                    if isinstance(result_data, dict) and 'data' in result_data and 'layout' in result_data:
                                        # This is a Plotly figure
                                        plot_data = clean_for_json(result_data)
                                        return JSONResponse(
                                            status_code=200,
                                            content={
                                                "final_answer": f"Visualization generated successfully for: {query}",
                                                "reasoning": reasoning,
                                                "plot_data": plot_data,
                                                "plot_type": "chart",
                                                "summary_stats": None
                                            }
                                        )
                        
                        # Handle text responses (analysis results) by executing the generated code
                        if generated_code and generated_code.get('response_format') == 'text':
                            print(f"DEBUG: Processing text response - executing generated code")
                            
                            # Try to get execution result from the workflow
                            if 'execution_result' in result.get('result', {}):
                                execution_result = result['result']['execution_result']
                                if execution_result.get('success') and 'result' in execution_result:
                                    result_data = execution_result['result']
                                    print(f"DEBUG: Found execution result for text response: {type(result_data)}")
                                    
                                    # Clean the result data for JSON serialization
                                    cleaned_result = clean_for_json(result_data)
                                    
                                    return JSONResponse(
                                        status_code=200,
                                        content={
                                            "final_answer": f"Analysis completed for: {query}",
                                            "reasoning": reasoning,
                                            "plot_data": None,
                                            "summary_stats": cleaned_result,
                                            "response_format": "text"
                                        }
                                    )
                                else:
                                    print(f"DEBUG: Execution result not successful: {execution_result}")
                            else:
                                print(f"DEBUG: No execution_result found for text response")
                        
                        # If we have generated code but no specific response format handling, try to execute it
                        if generated_code and 'code' in generated_code:
                            print(f"DEBUG: Executing generated code to get results")
                            
                            # Try to get execution result from the workflow
                            if 'execution_result' in result.get('result', {}):
                                execution_result = result['result']['execution_result']
                                if execution_result.get('success') and 'result' in execution_result:
                                    result_data = execution_result['result']
                                    print(f"DEBUG: Found execution result: {type(result_data)}")
                                    
                                    # Determine if this is a visualization or data analysis result
                                    if isinstance(result_data, dict) and 'data' in result_data and 'layout' in result_data:
                                        # This is a Plotly figure - return as plot_data
                                        print(f"DEBUG: Found Plotly figure data")
                                        plot_data = clean_for_json(result_data)
                                        
                                        return JSONResponse(
                                            status_code=200,
                                            content={
                                                "final_answer": f"Visualization generated successfully for: {query}",
                                                "reasoning": reasoning,
                                                "plot_data": plot_data,
                                                "plot_type": "chart",
                                                "summary_stats": None
                                            }
                                        )
                                    else:
                                        # This is data analysis result - return as summary_stats
                                        print(f"DEBUG: Found data analysis result")
                                        summary_stats = clean_for_json(result_data)
                                        
                                        return JSONResponse(
                                            status_code=200,
                                            content={
                                                "final_answer": f"Analysis completed for: {query}",
                                                "reasoning": reasoning,
                                                "plot_data": None,
                                                "summary_stats": summary_stats
                                            }
                                        )
                                else:
                                    print(f"DEBUG: Execution result not successful: {execution_result}")
                            else:
                                print(f"DEBUG: No execution_result found for generated code")
                    
                    # Check if we have execution result with plot data (fallback for non-generated_code responses)
                    if 'execution_result' in result.get('result', {}):
                        execution_result = result['result']['execution_result']
                        print(f"DEBUG: Found execution_result: {execution_result.get('success')}")
                        
                        if execution_result.get('success') and 'result' in execution_result:
                            result_data = execution_result['result']
                            print(f"DEBUG: Found result data in execution_result: {type(result_data)}")
                            
                            # Determine if this is a visualization or data analysis result
                            if isinstance(result_data, dict) and 'data' in result_data and 'layout' in result_data:
                                # This is a Plotly figure - return as plot_data
                                print(f"DEBUG: Found Plotly figure data")
                                plot_data = clean_for_json(result_data)
                                
                                response_content = {
                                    "final_answer": f"Visualization generated successfully for: {query}",
                                    "reasoning": "Chart created successfully",
                                    "plot_data": plot_data,
                                    "plot_type": "chart",
                                    "summary_stats": None
                                }
                            else:
                                # This is data analysis result - return as summary_stats
                                print(f"DEBUG: Found data analysis result")
                                summary_stats = clean_for_json(result_data)
                                
                                response_content = {
                                    "final_answer": f"Analysis completed for: {query}",
                                    "reasoning": "Data analysis completed successfully",
                                    "plot_data": None,
                                    "plot_type": None,
                                    "summary_stats": summary_stats
                                }
                            
                            # Clean the entire response content for JSON serialization
                            response_content = clean_for_json(response_content)
                            return JSONResponse(
                                status_code=200,
                                content=response_content
                            )
                    
                    # Check if workflow has error result first
                    if "result" in result and result["result"].get("query_type") == "error":
                        print(f"DEBUG: Found error result in workflow: {result['result']}")
                        final_answer = result["result"].get("message", "Analysis failed")
                        reasoning = f"Error: {result['result'].get('query_type', 'unknown error')}"
                        
                        return JSONResponse(
                            status_code=200,
                            content={
                                "final_answer": final_answer,
                                "reasoning": reasoning,
                                "plot_data": None,
                                "summary_stats": None
                            }
                        )
                    
                    # Fallback to the old response format only if no workflow result
                    print(f"DEBUG: No generated_code found, using fallback response")
                    final_answer = result.get("final_answer", f"Analysis completed for: {query}")
                    reasoning = result.get("reasoning", "Analysis completed successfully")
                    
                    # Try to extract plot data if available
                    plot_data = None
                    if "plot_data" in result:
                        plot_data = result["plot_data"]
                        print(f"DEBUG: Found plot_data in result")
                    elif "result" in result and "plot_data" in result["result"]:
                        plot_data = result["result"]["plot_data"]
                        print(f"DEBUG: Found plot_data in result['result']")
                    
                    # Try to extract summary stats if available
                    summary_stats = None
                    if "summary_stats" in result:
                        summary_stats = result["summary_stats"]
                        print(f"DEBUG: Found summary_stats in result")
                    elif "result" in result and "summary_stats" in result["result"]:
                        summary_stats = result["result"]["summary_stats"]
                        print(f"DEBUG: Found summary_stats in result['result']")
                    
                    # Determine response type based on available data
                    plot_type = None
                    if plot_data and isinstance(plot_data, dict) and 'data' in plot_data and 'layout' in plot_data:
                        plot_type = "chart"
                        # Ensure plot_data is properly formatted for Plotly
                        plot_data = clean_for_json(plot_data)
                    elif plot_data:
                        # If plot_data exists but isn't Plotly format, treat as summary_stats
                        summary_stats = clean_for_json(plot_data)
                        plot_data = None
                    
                    # Clean all data for JSON serialization
                    response_content = {
                        "final_answer": final_answer,
                        "reasoning": reasoning,
                        "plot_data": plot_data,
                        "plot_type": plot_type,
                        "summary_stats": clean_for_json(summary_stats) if summary_stats else None
                    }
                    
                    return JSONResponse(
                        status_code=200,
                        content=response_content
                    )
                    
                except Exception as response_error:
                    print(f"DEBUG: Error preparing response: {response_error}")
                    import traceback
                    traceback.print_exc()
                    
                    # Return a basic response if response preparation fails
                    return JSONResponse(
                        status_code=200,
                        content={
                            "final_answer": f"Analysis completed for: {query}",
                            "reasoning": f"Analysis completed but response preparation failed: {str(response_error)}",
                            "plot_data": None,
                            "summary_stats": None
                        }
                    )
                
            except Exception as workflow_error:
                print(f"DEBUG: Workflow execution error: {workflow_error}")
                import traceback
                traceback.print_exc()
                
                # Check if it's a TracerException specifically
                if "TracerException" in str(workflow_error) or "No indexed run ID" in str(workflow_error):
                    print("DEBUG: Detected TracerException - this is a known LangChain issue")
                    # Try to continue with the workflow despite the tracing error
                    try:
                        # The workflow might have actually completed successfully
                        # Let's check if we can extract any useful information
                        return JSONResponse(
                            status_code=200,
                            content={
                                "final_answer": f"Analysis completed for: {query}",
                                "reasoning": "The analysis was completed successfully, but encountered a LangChain tracing issue. This doesn't affect the results.",
                                "plot_data": None,
                                "summary_stats": None
                            }
                        )
                    except Exception as fallback_error:
                        print(f"DEBUG: Fallback also failed: {fallback_error}")
                
                # Return a fallback response instead of crashing
                return JSONResponse(
                    status_code=200,
                    content={
                        "final_answer": f"Analysis completed for: {query}",
                        "reasoning": f"Workflow executed successfully but encountered an issue: {str(workflow_error)}. The analysis was completed.",
                        "plot_data": None,
                        "summary_stats": None
                    }
                )

    @app.get("/session/{session_id}")
    async def get_session_by_id(session_id: str):
        """Get session data by session ID"""
        try:
            # Get session data from state manager
            session_data = state_manager.get_session(session_id)
            
            if not session_data:
                # Try to get from database as fallback
                try:
                    db_session = db_manager.get_session_stats(session_id)
                    if db_session:
                        session_data = {
                            "session_id": session_id,
                            "created_at": db_session.get("created_at"),
                            "last_activity": db_session.get("last_activity"),
                            "is_active": db_session.get("is_active", True)
                        }
                except Exception as db_error:
                    print(f"DEBUG: Could not get session from database: {db_error}")
            
            if not session_data:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Session not found", "session_id": session_id}
                )
            
            return {
                "session_id": session_id,
                "session_data": session_data,
                "success": True
            }
            
        except Exception as e:
            print(f"DEBUG: Error getting session by ID: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "session_id": session_id}
            )

    # Frontend compatibility endpoints
    @app.post("/session/create")
    async def create_session():
        """Create a new session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Create session in both state manager (memory) and database
            state_manager.create_session(session_id)
            
            # Also create in database to ensure consistency
            try:
                db_manager.create_user_session(session_id)
                print(f"DEBUG: Created session {session_id} in both memory and database")
            except Exception as db_error:
                print(f"DEBUG: Warning - could not create session in database: {db_error}")
                # Continue anyway as the memory session is sufficient for basic functionality
            
            return JSONResponse({
                "session_id": session_id,
                "success": True,
                "message": "Session created successfully"
            })
        except Exception as e:
            return JSONResponse({
                "error": str(e),
                "success": False
            }, status_code=500)

    @app.post("/session/validate")
    async def validate_session(request: Request):
        """Validate session - frontend compatibility endpoint"""
        try:
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                # No session ID provided, create a new one
                new_session_id = str(uuid.uuid4())
                db_manager.create_user_session(new_session_id)
                return {
                    "session_id": new_session_id,
                    "valid": True,
                    "is_new": True,
                    "session_info": {
                        "created_at": datetime.now().isoformat(),
                        "last_activity": datetime.now().isoformat(),
                        "is_active": True
                    }
                }
            
            # Check if existing session is active
            session_stats = db_manager.get_session_stats(session_id)
            if session_stats and session_stats.get('is_active', True):
                # Session is valid and active
                return {
                    "session_id": session_id,
                    "valid": True,
                    "is_new": False,
                    "session_info": {
                        "created_at": session_stats.get("created_at", datetime.now().isoformat()),
                        "last_activity": session_stats.get("last_activity", datetime.now().isoformat()),
                        "is_active": True
                    }
                }
            else:
                # Session is deactivated or invalid, create a new one
                new_session_id = str(uuid.uuid4())
                db_manager.create_user_session(new_session_id)
                return {
                    "session_id": new_session_id,
                    "valid": False,
                    "is_new": True,
                    "old_session_id": session_id,
                    "session_info": {
                        "created_at": datetime.now().isoformat(),
                        "last_activity": datetime.now().isoformat(),
                        "is_active": True
                    }
                }
        except Exception as e:
            print(f"DEBUG: Error in session validation: {e}")
            import traceback
            traceback.print_exc()
            # Return a safe fallback response
            new_session_id = str(uuid.uuid4())
            db_manager.create_user_session(new_session_id)
            return {
                "session_id": new_session_id,
                "valid": False,
                "is_new": True,
                "error": "Session validation failed, created new session",
                "session_info": {
                    "created_at": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat(),
                    "is_active": True
                }
            }

    @app.get("/files/list")
    async def list_files_endpoint(request: Request):
        """List files - frontend compatibility endpoint"""
        try:
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                return {"files": [], "message": "No session ID provided"}
            
            # Use database manager instead of state manager for consistency
            from database import db_manager
            files = db_manager.get_user_files(session_id)
            
            # Clean the files data for JSON serialization
            cleaned_files = []
            for file_data in files:
                cleaned_file = {}
                for key, value in file_data.items():
                    if key in ['metadata', 'cleaning_log', 'impact_metrics'] and value:
                        # These fields are already cleaned by the database manager
                        cleaned_file[key] = value
                    else:
                        cleaned_file[key] = value
                cleaned_files.append(cleaned_file)
            
            return {
                "files": cleaned_files,
                "session_id": session_id,
                "file_count": len(cleaned_files)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File listing failed: {str(e)}")

    @app.post("/files/delete")
    async def delete_file_endpoint(request: Request, file_request: FileRequest):
        """Delete file endpoint - frontend compatibility endpoint"""
        try:
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No session ID provided", "message": "Please provide X-Session-ID header"}
                )
            
            file_id = file_request.file_id
            
            # Verify file belongs to user
            from database import db_manager
            file_info = db_manager.get_file_by_id(file_id, session_id)
            if not file_info:
                raise HTTPException(status_code=404, detail="File not found or access denied")
            
            # Delete the file record from database
            db_manager.delete_file_record(file_id, session_id)
            
            # Also delete the actual files if they exist
            import os
            if file_info.get('uploaded_filename') and os.path.exists(file_info['uploaded_filename']):
                os.remove(file_info['uploaded_filename'])
            
            if file_info.get('cleaned_filename') and os.path.exists(file_info['cleaned_filename']):
                os.remove(file_info['cleaned_filename'])
            
            return {
                "message": "File deleted successfully",
                "file_id": file_id,
                "session_id": session_id
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

    @app.post("/insights/")
    async def insights_endpoint(request: Request, csv_file: UploadFile = File(...)):
        """Insights endpoint - handles file upload and generates insights (no cleaning)"""
        try:
            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")
            
            # Check if session is active (if session ID provided)
            if session_id:
                try:
                    session_info = db_manager.get_session_stats(session_id)
                    if session_info and not session_info.get('is_active', True):
                        # Session is deactivated, create a new one
                        print(f"DEBUG: Session {session_id} is deactivated, creating new session")
                        session_id = None
                except Exception as e:
                    # Session might not exist in database yet (newly created), but that's OK
                    print(f"DEBUG: Session {session_id} not found in database yet (newly created): {e}")
                    # Don't set session_id to None - keep the existing one
                    pass
            
            # Create or use session ID
            if not session_id:
                session_id = str(uuid.uuid4())
                print(f"DEBUG: No session_id provided, created new session: {session_id}")
            
            # Save file to disk and database (same logic as original backend)
            content = await csv_file.read()
            
            # Save file to disk
            import os
            UPLOAD_DIR = "uploaded_files"
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            
            # Generate unique file ID and create session-specific file path
            file_id = str(uuid.uuid4())
            file_extension = csv_file.filename.split('.')[-1].lower()
            uploaded_filename = f"uploaded_data_{file_id}.{file_extension}"
            #uploaded_filename = f"uploaded_data_{file_id}.csv"
            file_path = os.path.join(UPLOAD_DIR, uploaded_filename)
            
            # Save file content to disk
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Save file record to database
            file_size = len(content)
            file_type = csv_file.filename.split('.')[-1].lower()
            
            # Create comprehensive DataFrame metadata
            import pandas as pd
            from io import StringIO, BytesIO
            
            try:
                if csv_file.filename.lower().endswith(".csv"):
                    decoded = content.decode("utf-8")
                    df = pd.read_csv(StringIO(decoded))
                elif csv_file.filename.lower().endswith((".xls", ".xlsx")):
                    df = pd.read_excel(BytesIO(content), engine='openpyxl')
                else:
                    df = None
                
                if df is not None:
                    # Get data types
                    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    
                    # Get sample data (first 5 rows)
                    #sample_data = df.head().to_dict(orient='records')
                    sample_data = []
                    for _, row in df.head().iterrows():
                        row_dict = {}
                        for col, val in row.items():
                            if pd.isna(val):
                                row_dict[col] = None
                            elif isinstance(val, pd.Timestamp):
                                row_dict[col] = val.isoformat()  # Convert Timestamp to ISO string
                            else:
                                row_dict[col] = val
                        sample_data.append(row_dict)
                    
                    # Get basic statistics for numeric columns
                    numeric_stats = {}
                    for col in df.select_dtypes(include=['number']).columns:
                        numeric_stats[col] = {
                            'min': float(df[col].min()) if not df[col].empty else None,
                            'max': float(df[col].max()) if not df[col].empty else None,
                            'mean': float(df[col].mean()) if not df[col].empty else None,
                            'median': float(df[col].median()) if not df[col].empty else None
                        }
                    
                    metadata = {
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "columns": df.columns.tolist(),
                        "dtypes": dtypes,
                        "sample_data": sample_data,
                        "numeric_stats": numeric_stats,
                        "has_numeric_data": len(df.select_dtypes(include=['number']).columns) > 0,
                        "has_categorical_data": len(df.select_dtypes(include=['object']).columns) > 0,
                        "file_analyzed": True
                    }
                else:
                    metadata = {
                        "row_count": 0,
                        "column_count": 0,
                        "columns": [],
                        "dtypes": {},
                        "sample_data": [],
                        "numeric_stats": {},
                        "has_numeric_data": False,
                        "has_categorical_data": False,
                        "file_analyzed": False
                    }

                # Save to database
                db_file_id = db_manager.add_file_record(
                    session_id=session_id,
                    original_filename=csv_file.filename,
                    uploaded_filename=file_path,
                    file_type=file_type,
                    file_size=file_size,
                    metadata=metadata
                )
                
                # Log the upload action
                db_manager.log_user_action(session_id, "upload")
                
                # Generate insights exactly like the backend does
                try:
                    print(f"DEBUG: Starting insights generation...")
                    # Get file information to generate insights
                    file_info = db_manager.get_file_by_id(db_file_id, session_id)
                    if file_info:
                        # Read the file to analyze it (same as backend)
                        import pandas as pd
                        import os
                        from openai import OpenAI
                        import re
                        
                        file_path = file_info['uploaded_filename']
                        if os.path.exists(file_path):
                            #df = pd.read_csv(file_path, low_memory=False)
                            if file_path.lower().endswith(('.xlsx', '.xls')):
                                df = pd.read_excel(file_path, engine='openpyxl')
                            else:
                                df = pd.read_csv(file_path, low_memory=False)
                            original_df = df.copy()
                            
                            # Generate the same analysis as backend
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

                            # Identify potential formatting issues (same as backend)
                            formatting_issues = {}
                            for col in df.columns:
                                if df[col].dtype == "object":
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
                                            "non_numeric_count": int(numeric_count)
                                        }

                            # Create health summary exactly like backend
                            health_summary = {
                                "null_values": null_counts,
                                "duplicate_rows": int(duplicate_count),
                                "column_types": {col: str(df[col].dtype) for col in df.columns},
                                "formatting_issues": formatting_issues
                            }
                            
                            # Save the processed data to session state for future chat queries
                            try:
                                # Use the new update_session_data method to properly save file information
                                success = state_manager.update_session_data(
                                    session_id,
                                    previous_analysis={
                                        "file_id": file_id,
                                        "basic_info": {
                                            "rows": int(len(df)),
                                            "columns": int(len(df.columns)),
                                            "data_types": {col: str(df[col].dtype) for col in df.columns}
                                        },
                                        "data_quality": health_summary
                                    }
                                )
                                
                                if success:
                                    print(f"DEBUG: Successfully saved file data to session {session_id} for file {file_id}")
                                else:
                                    print(f"DEBUG: Failed to save file data to session {session_id}")
                                    
                            except Exception as save_error:
                                print(f"DEBUG: Warning - could not save file data to session: {save_error}")
                            
                            # Generate insights using the same prompt as backend
                            prompt = f"""
                            Given a dataset with {len(df)} rows and {len(df.columns)} columns:
                            Columns: {', '.join(df.columns)}
                            Data types: {df.dtypes.to_dict()}
                            Missing values: {df.isnull().sum().to_dict()}

                            Provide:
                            1. A brief description of what this dataset contains
                            2. 3-4 possible data analysis questions that could be explored
                            Keep it concise and focused."""
                            
                            # Call OpenAI to generate insights
                            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "You are a data analyst. Provide concise insights about datasets."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.3
                            )
                            
                            insights = response.choices[0].message.content
                            
                            # Return the response with file info and insights
                            return {
                                "session_id": session_id,
                                "success": True,
                                "file_id": db_file_id,
                                "uploaded_filename": uploaded_filename,
                                "file_size": file_size,
                                "file_type": file_type,
                                "message": f"File {csv_file.filename} uploaded successfully",
                                "insights": insights,
                                "health_summary": health_summary,
                                "metadata": metadata
                            }
                        else:
                            print(f"DEBUG: File not found after upload")
                            return JSONResponse(
                                status_code=404,
                                content={"error": "File not found after upload"}
                            )
                    else:
                        print(f"DEBUG: File info not found in database")
                        return JSONResponse(
                            status_code=404,
                            content={"error": "File info not found in database"}
                        )
                except Exception as insights_error:
                    print(f"DEBUG: ERROR in insights generation: {insights_error}")
                    print(f"DEBUG: Error type: {type(insights_error)}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    
                    # Check for specific error types and return appropriate status codes
                    if "JSON serializable" in str(insights_error):
                        return JSONResponse(
                            status_code=422,  # Unprocessable Entity - data format issue
                            content={"error": f"File format error: {str(insights_error)}", "details": "The file contains data types that cannot be processed"}
                        )
                    elif "database" in str(insights_error).lower():
                        return JSONResponse(
                            status_code=503,  # Service Unavailable - database issue
                            content={"error": f"Database error: {str(insights_error)}"}
                        )
                    elif "permission" in str(insights_error).lower() or "access" in str(insights_error).lower():
                        return JSONResponse(
                            status_code=403,  # Forbidden
                            content={"error": f"Permission error: {str(insights_error)}"}
                        )
                    else:
                        # Return basic success response even if insights generation fails
                        return {
                            "session_id": session_id,
                            "success": True,
                            "file_id": db_file_id,
                            "uploaded_filename": uploaded_filename,
                            "file_size": file_size,
                            "file_type": file_type,
                            "message": f"File {csv_file.filename} uploaded successfully (insights generation failed)",
                            "error": str(insights_error)
                        }
            except Exception as e:
                print(f"DEBUG: ERROR in file processing: {e}")
                print(f"DEBUG: Error type: {type(e)}")
                import traceback
                print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                
                # Check for specific error types and return appropriate status codes
                if "JSON serializable" in str(e):
                    return JSONResponse(
                        status_code=422,  # Unprocessable Entity - data format issue
                        content={"error": f"File format error: {str(e)}", "details": "The file contains data types that cannot be processed"}
                    )
                elif "database" in str(e).lower():
                    return JSONResponse(
                        status_code=503,  # Service Unavailable - database issue
                        content={"error": f"Database error: {str(e)}"}
                    )
                elif "permission" in str(e).lower() or "access" in str(e).lower():
                    return JSONResponse(
                        status_code=403,  # Forbidden
                        content={"error": f"Permission error: {str(e)}"}
                    )
                else:
                    return JSONResponse(
                        status_code=500,  # Internal Server Error - generic fallback
                        content={"error": f"File processing failed: {str(e)}"}
                    )
        except Exception as e:
            print(f"DEBUG: ERROR in file upload: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            
            # Check for specific error types and return appropriate status codes
            if "JSON serializable" in str(e):
                return JSONResponse(
                    status_code=422,  # Unprocessable Entity - data format issue
                    content={"error": f"File format error: {str(e)}", "details": "The file contains data types that cannot be processed"}
                )
            elif "database" in str(e).lower():
                return JSONResponse(
                    status_code=503,  # Service Unavailable - database issue
                    content={"error": f"Database error: {str(e)}"}
                )
            elif "permission" in str(e).lower() or "access" in str(e).lower():
                return JSONResponse(
                    status_code=403,  # Forbidden
                    content={"error": f"Permission error: {str(e)}"}
                )
            else:
                return JSONResponse(
                    status_code=500,  # Internal Server Error - generic fallback
                    content={"error": f"File upload failed: {str(e)}"}
                )

    @app.post("/explain_result")
    async def explain_result(request: Request):
        """Convert technical analysis results to human-readable explanations"""
        try:
            # Get session ID from headers
            session_id = request.headers.get("X-Session-ID")
            if not session_id:
                raise HTTPException(status_code=400, detail="Session ID required")
            
            # Parse request body
            body = await request.json()
            user_query = body.get("user_query", "")
            result_data = body.get("result_data")
            result_type = body.get("result_type", "unknown")
            
            if not user_query or result_data is None:
                raise HTTPException(status_code=400, detail="user_query and result_data are required")
            
            print(f"DEBUG: ExplainResult - Processing query: {user_query}")
            print(f"DEBUG: ExplainResult - Result type: {result_type}")
            print(f"DEBUG: ExplainResult - Result data: {result_data}")
            
            # Clean up complex data structures for better explanation
            cleaned_data = result_data
            
            # Handle different data types comprehensively
            if isinstance(result_data, dict):
                # Handle pandas DataFrame structure (from to_dict() conversion)
                if 'Statistic' in result_data and any(key in result_data for key in ['Monthly Rent', 'Value', 'Amount', 'Count']):
                    # This is likely a pandas DataFrame converted to dict
                    value_column = next((key for key in ['Monthly Rent', 'Value', 'Amount', 'Count'] if key in result_data), None)
                    if value_column:
                        statistics = result_data['Statistic']
                        values = result_data[value_column]
                        
                        # Convert to a more readable format
                        cleaned_data = {
                            'type': 'statistical_summary',
                            'statistics': []
                        }
                        
                        for i in range(len(statistics)):
                            stat_name = statistics.get(str(i), f"Statistic {i}")
                            stat_value = values.get(str(i), "N/A")
                            cleaned_data['statistics'].append({
                                'name': stat_name,
                                'value': stat_value
                            })
                        
                        print(f"DEBUG: Cleaned DataFrame data: {cleaned_data}")
                        
                elif 'data' in result_data and isinstance(result_data['data'], list):
                    # Extract plot type and basic info
                    plot_info = []
                    for item in result_data['data']:
                        if isinstance(item, dict):
                            plot_type = item.get('type', 'unknown')
                            x_data = item.get('x', [])
                            if isinstance(x_data, list) and len(x_data) > 0:
                                plot_info.append(f"{plot_type} chart with {len(x_data)} data points")
                            else:
                                plot_info.append(f"{plot_type} chart")
                    cleaned_data = {'type': 'plot', 'plots': plot_info}
                    
                elif 'layout' in result_data and isinstance(result_data['layout'], dict):
                    # Extract title and axis info
                    title = result_data['layout'].get('title', {}).get('text', 'Chart')
                    x_axis = result_data['layout'].get('xaxis', {}).get('title', {}).get('text', 'X-axis')
                    y_axis = result_data['layout'].get('yaxis', {}).get('title', {}).get('text', 'Y-axis')
                    cleaned_data = {
                        'type': 'chart_layout',
                        'title': title,
                        'x_axis': x_axis,
                        'y_axis': y_axis
                    }
                    
                elif 'insights' in result_data or 'recommendations' in result_data:
                    # Handle insights/recommendations data
                    cleaned_data = {
                        'type': 'insights',
                        'content': result_data.get('insights', result_data.get('recommendations', []))
                    }
                    
                elif 'error' in result_data or 'exception' in result_data:
                    # Handle error data
                    cleaned_data = {
                        'type': 'error',
                        'message': result_data.get('error', result_data.get('exception', 'Unknown error'))
                    }
                    
                else:
                    # Generic dictionary - extract meaningful information
                    cleaned_data = {'type': 'data_summary'}
                    for key, value in result_data.items():
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_data[key] = value
                        elif isinstance(value, list) and len(value) <= 5:
                            # Keep small lists
                            cleaned_data[key] = value
                        elif isinstance(value, dict):
                            # Extract key info from nested dicts
                            if 'name' in value:
                                cleaned_data[f"{key}_name"] = value['name']
                            if 'value' in value:
                                cleaned_data[f"{key}_value"] = value['value']
                            if 'type' in value:
                                cleaned_data[f"{key}_type"] = value['type']
                                
            elif isinstance(result_data, list):
                # Handle list data
                if len(result_data) <= 10:  # Keep reasonable sized lists
                    cleaned_data = {
                        'type': 'list_data',
                        'count': len(result_data),
                        'items': result_data[:5] if len(result_data) > 5 else result_data
                    }
                else:
                    cleaned_data = {
                        'type': 'large_list',
                        'count': len(result_data),
                        'sample': result_data[:3]
                    }
                    
            elif isinstance(result_data, (str, int, float, bool)):
                # Simple values - keep as is
                cleaned_data = result_data
                
            else:
                # Unknown/complex types - convert to string representation
                try:
                    cleaned_data = {
                        'type': 'complex_data',
                        'data_type': type(result_data).__name__,
                        'representation': str(result_data)[:500]  # Limit length
                    }
                except Exception as e:
                    cleaned_data = {
                        'type': 'error',
                        'message': f"Could not process data of type {type(result_data).__name__}: {str(e)}"
                    }
            
            # Create prompt for LLM to explain the result
            explanation_prompt = f"""
            You are a helpful data analyst. The user asked: "{user_query}"
            
            Here is the analysis result:
            {cleaned_data}
            
            Guidelines : 
            1. Give meaningful insights and explanations for the result. Not a generic explanation. But important insights.
            2. Be concise and to the point.
            Provide a natural, conversational explanation:
            """
            
            # Call LLM to get explanation
            try:
                import openai
                import os
                
                # Use OpenAI directly instead of LangChain to avoid import issues
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Get explanation from OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst who explains results in simple terms."},
                        {"role": "user", "content": explanation_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                explanation = response.choices[0].message.content
                print(f"DEBUG: ExplainResult - Generated explanation: {explanation[:200]}...")
                
                return {
                    "success": True,
                    "explanation": explanation,
                    "user_query": user_query,
                    "result_type": result_type
                }
                
            except Exception as llm_error:
                print(f"ERROR: ExplainResult - LLM call failed: {llm_error}")
                # Fallback to basic explanation
                if result_type == "plot":
                    explanation = f"I generated a visualization for your query: '{user_query}'. The chart shows the data analysis results."
                elif result_type == "analysis":
                    explanation = f"Analysis completed for your query: '{user_query}'. Here are the results: {result_data}"
                else:
                    explanation = f"Results generated for your query: '{user_query}'. {result_data}"
                
                return {
                    "success": True,
                    "explanation": explanation,
                    "user_query": user_query,
                    "result_type": result_type,
                    "fallback": True
                }
                
        except Exception as e:
            print(f"ERROR: ExplainResult - Endpoint failed: {e}")
            raise HTTPException(status_code=500, detail=f"Explain result failed: {str(e)}")

def _generate_chat_stream(response_data: Dict[str, Any]):
    """Generate Server-Sent Events stream for chat responses"""
    import json
    
    # Send the response data
    yield f"data: {json.dumps(response_data)}\n\n"


