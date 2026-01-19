"""
Smart Code Executor Node

This node executes the generated Python code and returns results.
"""

import os
import sys
import math
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from .base_node import BaseNode
from state import ConversationState
from state import state_manager  # Added this import

class SmartCodeExecutorNode(BaseNode):
    """Node for executing generated Python code safely"""
    
    def __init__(self):
        super().__init__(
            name="smart_code_executor",
            description="Executes generated Python code safely and returns results"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "user_query"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Execute the generated Python code"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for smart_code_executor node")
            
            self.log_execution(state)
            
            # Get generated code from result field
            if not state.get("result") or "generated_code" not in state["result"]:
                raise ValueError("No generated code found in state result")
            
            generated_code = state["result"]["generated_code"]
            
            # Check if code exists (success is implied if we have code)
            if not generated_code.get("code"):
                # Code generation failed, store error
                if state.get("result") is None:
                    state["result"] = {}
                
                state["result"]["execution_result"] = {
                    "success": False,
                    "error": "Code generation failed",
                    "details": "No code found in generated_code"
                }
                state["current_step"] = "execution_failed"
                return state
            
            code = generated_code["code"]
            
            # Execute the code
            print(f"DEBUG: SmartCodeExecutor - About to execute code: {code[:200]}...")
            execution_result = self._execute_code_safely(code, state)
            print(f"DEBUG: SmartCodeExecutor - Execution result: {execution_result}")
            
            # Store execution result in result field
            if state.get("result") is None:
                state["result"] = {}
            
            state["result"]["execution_result"] = execution_result
            state["current_step"] = "execution_complete"
            
            # Add reasoning to result
            if execution_result["success"]:
                state["result"]["reasoning"] = "Code executed successfully"
                
                # Store the actual execution result data in the format the API expects
                if "result" in execution_result:
                    result_data = execution_result["result"]
                    # Ensure the result_data is fully serializable
                    result_data = self._ensure_serializable(result_data)
                    state["result"]["result_data"] = result_data
                    print(f"DEBUG: SmartCodeExecutor - Stored result data: {result_data}")
                    
                                        # Check if we need to convert DataFrame results to table visualizations
                    query_classification = state.get("query_classification", {})
                    response_format = query_classification.get("response_format", "")
                    
                    print(f"DEBUG: SmartCodeExecutor - Checking table conversion. response_format: {response_format}")
                    print(f"DEBUG: SmartCodeExecutor - result_data type: {type(result_data)}")
                    
                    if response_format == "table":
                        print(f"DEBUG: SmartCodeExecutor - Table format requested, attempting conversion...")
                        try:
                            # Import plotly
                            import plotly.graph_objects as go
                            print(f"DEBUG: SmartCodeExecutor - Plotly imported successfully")
                            
                            # Get table data - handle both raw DataFrames and processed results
                            if hasattr(result_data, 'shape') and hasattr(result_data, 'columns'):
                                # Raw DataFrame - create more meaningful summary
                                print(f"DEBUG: SmartCodeExecutor - Processing raw DataFrame with shape {result_data.shape}")
                                
                                # Get basic info
                                numeric_cols = result_data.select_dtypes(include=['number']).columns.tolist()
                                categorical_cols = result_data.select_dtypes(include=['object', 'category']).columns.tolist()
                                
                                # Create enhanced summary data
                                enhanced_summary = []
                                for col in result_data.columns:
                                    col_data = result_data[col]
                                    col_info = {
                                        'column': col,
                                        'dtype': str(col_data.dtype),
                                        'count': len(col_data.dropna()),
                                        'null_count': col_data.isnull().sum(),
                                        'unique_count': col_data.nunique()
                                    }
                                    
                                    # Add numeric statistics if applicable
                                    if col in numeric_cols:
                                        col_info.update({
                                            'mean': col_data.mean() if not col_data.empty else None,
                                            'std': col_data.std() if not col_data.empty else None,
                                            'min': col_data.min() if not col_data.empty else None,
                                            '25%': col_data.quantile(0.25) if not col_data.empty else None,
                                            '50%': col_data.quantile(0.50) if not col_data.empty else None,
                                            '75%': col_data.quantile(0.75) if not col_data.empty else None,
                                            'max': col_data.max() if not col_data.empty else None
                                        })
                                    else:
                                        # For categorical columns, add mode info
                                        mode_info = col_data.mode()
                                        col_info.update({
                                            'top': mode_info.iloc[0] if not mode_info.empty else None,
                                            'freq': (col_data == mode_info.iloc[0]).sum() if not mode_info.empty else None
                                        })
                                    
                                    enhanced_summary.append(col_info)
                                
                                table_data = {
                                    "type": "dataframe",
                                    "shape": result_data.shape,
                                    "columns": ['column', 'dtype', 'count', 'null_count', 'unique_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'top', 'freq'],
                                    "head": enhanced_summary,
                                    "dtypes": self._make_serializable(result_data.dtypes.to_dict()),
                                    "summary": enhanced_summary
                                }
                                print(f"DEBUG: SmartCodeExecutor - Created enhanced DataFrame summary with {len(enhanced_summary)} columns")
                            elif isinstance(result_data, dict) and result_data.get("type") == "dataframe":
                                # Processed DataFrame result - create enhanced summary from scratch
                                print(f"DEBUG: SmartCodeExecutor - Creating enhanced summary from processed DataFrame result")
                                
                                # Try to extract the original DataFrame from the session state
                                # Look for the original data in the session
                                session_data = state.get("session_data", {})
                                original_df = None
                                
                                # Try to find the original DataFrame in various possible locations
                                if "data" in session_data:
                                    original_df = session_data["data"]
                                elif "df" in session_data:
                                    original_df = session_data["df"]
                                elif "dataframe" in session_data:
                                    original_df = session_data["dataframe"]
                                
                                if original_df is not None and hasattr(original_df, 'shape'):
                                    print(f"DEBUG: SmartCodeExecutor - Found original DataFrame, creating enhanced summary")
                                    # Create enhanced summary from original DataFrame
                                    numeric_cols = original_df.select_dtypes(include=['number']).columns.tolist()
                                    categorical_cols = original_df.select_dtypes(include=['object', 'category']).columns.tolist()
                                    
                                    enhanced_summary = []
                                    for col in original_df.columns:
                                        col_data = original_df[col]
                                        col_info = {
                                            'column': col,
                                            'dtype': str(col_data.dtype),
                                            'count': len(col_data.dropna()),
                                            'null_count': col_data.isnull().sum(),
                                            'unique_count': col_data.nunique()
                                        }
                                        
                                        # Add numeric statistics if applicable
                                        if col in numeric_cols:
                                            col_info.update({
                                                'mean': col_data.mean() if not col_data.empty else None,
                                                'std': col_data.std() if not col_data.empty else None,
                                                'min': col_data.min() if not col_data.empty else None,
                                                '25%': col_data.quantile(0.25) if not col_data.empty else None,
                                                '50%': col_data.quantile(0.50) if not col_data.empty else None,
                                                '75%': col_data.quantile(0.75) if not col_data.empty else None,
                                                'max': col_data.max() if not col_data.empty else None
                                            })
                                        else:
                                            # For categorical columns, add mode info
                                            mode_info = col_data.mode()
                                            col_info.update({
                                                'top': mode_info.iloc[0] if not mode_info.empty else None,
                                                'freq': (col_data == mode_info.iloc[0]).sum() if not mode_info.empty else None
                                            })
                                        
                                        enhanced_summary.append(col_info)
                                    
                                    table_data = {
                                        "type": "dataframe",
                                        "shape": original_df.shape,
                                        "columns": ['column', 'dtype', 'count', 'null_count', 'unique_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'top', 'freq'],
                                        "head": enhanced_summary,
                                        "dtypes": self._make_serializable(original_df.dtypes.to_dict()),
                                        "summary": enhanced_summary
                                    }
                                    print(f"DEBUG: SmartCodeExecutor - Created enhanced DataFrame summary with {len(enhanced_summary)} columns")
                                else:
                                    print(f"DEBUG: SmartCodeExecutor - No original DataFrame found, using processed result as fallback")
                                    table_data = result_data
                            else:
                                print(f"DEBUG: SmartCodeExecutor - Unknown result_data type, skipping table conversion")
                                table_data = None
                            
                            if table_data and "columns" in table_data and "head" in table_data:
                                print(f"DEBUG: SmartCodeExecutor - Creating Plotly table with {len(table_data['columns'])} columns")
                                
                                # Clean the data to handle NaN, inf, and other non-JSON-compliant values
                                def clean_value(val):
                                    if val is None:
                                        return ''
                                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                                        return ''
                                    if str(val).lower() == 'nan':
                                        return ''
                                    return str(val)
                                
                                # Create cleaned table data
                                import math
                                cleaned_values = []
                                for i in range(min(len(table_data["head"]), 10)):
                                    row = []
                                    for col in table_data["columns"]:
                                        val = table_data["head"][i].get(col, '')
                                        row.append(clean_value(val))
                                    cleaned_values.append(row)
                                
                                # Create Plotly table
                                fig = go.Figure(data=[go.Table(
                                    header=dict(
                                        values=table_data["columns"],
                                        fill_color='paleturquoise',
                                        align='left',
                                        font=dict(size=12)
                                    ),
                                    cells=dict(
                                        values=cleaned_values,
                                        fill_color='lavender',
                                        align='left',
                                        font=dict(size=11)
                                    )
                                )])
                                
                                fig.update_layout(
                                    title=f"Data Analysis Results - {state.get('user_query', 'Query')}",
                                    width=800,
                                    height=400
                                )
                                
                                # Store the table visualization in plot_data
                                state["result"]["plot_data"] = fig.to_dict()
                                print(f"DEBUG: SmartCodeExecutor - Successfully created table visualization and stored in plot_data")
                            else:
                                print(f"DEBUG: SmartCodeExecutor - Table data missing required fields: columns={table_data.get('columns') if table_data else None}, head={table_data.get('head') if table_data else None}")
                                
                        except Exception as table_error:
                            print(f"DEBUG: SmartCodeExecutor - Error creating table visualization: {table_error}")
                            import traceback
                            traceback.print_exc()
                            # Continue without table visualization if there's an error
                
                # Update conversation memory with analysis results
                print(f"DEBUG: SmartCodeExecutor - About to call _update_conversation_memory...")
                self._update_conversation_memory(state, execution_result)
                print(f"DEBUG: SmartCodeExecutor - Finished _update_conversation_memory")
                
                # Add AI message to conversation
                from langchain_core.messages import AIMessage
                ai_message = "Code executed successfully! Here are your results."
                state["messages"].append(AIMessage(content=ai_message))
            else:
                state["result"]["reasoning"] = f"Code execution failed: {execution_result.get('error', 'Unknown error')}"
                
                # Add AI message to conversation
                from langchain_core.messages import AIMessage
                ai_message = f"I encountered an error while running the code: {execution_result.get('error', 'Unknown error')}"
                state["messages"].append(AIMessage(content=ai_message))
            
            print(f"DEBUG: SmartCodeExecutor - Final state result keys: {list(state.get('result', {}).keys())}")
            self.log_execution(state, f"Code execution {'succeeded' if execution_result['success'] else 'failed'}")
            
            # Update conversation history with this interaction
            try:
                state_manager.update_conversation_history(
                    state["session_id"],
                    state["user_query"],
                    f"Analysis completed: {str(execution_result.get('result', {}))}"
                )
            except Exception as e:
                print(f"Warning: Could not update conversation history: {e}")
            
            # Final safety check: ensure the entire state is serializable
            try:
                state = self._ensure_serializable(state)
            except Exception as e:
                print(f"Warning: Could not ensure state serialization: {e}")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _update_conversation_memory(self, state: ConversationState, execution_result: Dict[str, Any]):
        """Update conversation memory with analysis results for future context"""
        print(f"DEBUG: SmartCodeExecutor - _update_conversation_memory called")
        print(f"DEBUG: SmartCodeExecutor - State keys: {list(state.keys())}")
        print(f"DEBUG: SmartCodeExecutor - Execution result keys: {list(execution_result.keys())}")
        print(f"DEBUG: SmartCodeExecutor - Execution success: {execution_result.get('success')}")
        
        try:
            # Extract key information for memory
            memory_data = {
                "user_query": state.get("user_query", ""),
                "query_type": state.get("result", {}).get("query_type", ""),
                "timestamp": datetime.now().isoformat(),
                "execution_success": execution_result.get("success", False)
            }
            
            # Extract analysis context from result
            if execution_result.get("success") and "result" in execution_result:
                result_data = execution_result["result"]
                
                # Handle different result types
                if hasattr(result_data, 'to_dict') and callable(getattr(result_data, 'to_dict', None)):
                    # It's a Plotly Figure - extract plot information
                    plot_data = result_data.to_dict()
                    memory_data.update({
                        "result_type": "plot",
                        "plot_title": plot_data.get("layout", {}).get("title", {}).get("text", ""),
                        "plot_type": plot_data.get("data", [{}])[0].get("type", ""),
                        "x_axis": plot_data.get("data", [{}])[0].get("x", []),
                        "y_axis": plot_data.get("data", [{}])[0].get("y", [])
                    })
                    
                    # Extract data relationships from plot context
                    plot_title = plot_data.get("layout", {}).get("title", {}).get("text", "").lower()
                    if "flat type" in plot_title and "monthly rent" in plot_title:
                        memory_data["data_relationship"] = "monthly_rent vs flat_type"
                        memory_data["columns_analyzed"] = ["monthly_rent", "flat_type"]
                        memory_data["categorical_analysis"] = "flat_type"
                        memory_data["primary_analysis"] = "flat_type_rent_analysis"
                    elif "time" in plot_title or "year" in plot_title or "date" in plot_title:
                        memory_data["data_relationship"] = "monthly_rent vs time"
                        memory_data["columns_analyzed"] = ["monthly_rent", "time"]
                        memory_data["temporal_analysis"] = "time_series"
                    
                elif isinstance(result_data, (int, float, str)):
                    # It's a simple value (statistical result, text, etc.)
                    memory_data.update({
                        "result_type": "analysis",
                        "result_value": result_data,
                        "result_content": str(result_data)
                    })
                    
                    # For statistical analysis, try to extract context from the query
                    if isinstance(result_data, (int, float)) and "monthly_rent" in memory_data.get("user_query", "").lower():
                        memory_data["data_relationship"] = "monthly_rent_statistics"
                        memory_data["columns_analyzed"] = ["monthly_rent"]
                        memory_data["statistical_analysis"] = "numerical"
                    
                elif isinstance(result_data, dict):
                    # It's a dictionary - extract metadata normally
                    memory_data.update({
                        "result_type": result_data.get("type", ""),
                        "result_shape": result_data.get("shape", ""),
                        "result_columns": result_data.get("columns", []),
                        "insights": result_data.get("insights", []),
                        "summary_stats": result_data.get("summary", {})
                    })
                    
                    # Extract data relationships if available
                    if "columns" in result_data:
                        columns = result_data["columns"]
                        if "monthly_rent" in columns and any("year" in col.lower() for col in columns):
                            memory_data["data_relationship"] = "monthly_rent vs year"
                            memory_data["columns_analyzed"] = ["monthly_rent", "year"]
                        
                        if "flat_type" in columns:
                            memory_data["categorical_analysis"] = "flat_type"
                        
                        if "region" in columns:
                            memory_data["geographic_analysis"] = "region"
                else:
                    # It's some other type
                    memory_data.update({
                        "result_type": type(result_data).__name__,
                        "result_content": str(result_data)[:200]  # Truncate long content
                    })
            else:
                # No result data available
                memory_data.update({
                    "result_type": "no_data",
                    "result_content": "No execution result available"
                })
                print(f"DEBUG: SmartCodeExecutor - No result data available in execution_result")
            
            # Extract file context
            if state.get("result") and "selected_file" in state["result"]:
                selected_file = state["result"]["selected_file"]
                memory_data["file_context"] = {
                    "file_id": selected_file.get("file_id"),
                    "filename": selected_file.get("filename"),
                    "is_cleaned": selected_file.get("is_cleaned", False)
                }
            
            # Store in state for immediate use
            if state.get("conversation_memory") is None:
                state["conversation_memory"] = {}
            
            # CRITICAL: Preserve existing analysis history and append new analysis
            existing_history = state["conversation_memory"].get("analysis_history", [])
            
            # If this is the first analysis, initialize history
            if not existing_history:
                existing_history = []
                print(f"DEBUG: SmartCodeExecutor - Initializing new analysis history")
            
            # CRITICAL: Maintain limited conversation history (last 2-3 conversations)
            if "analysis_history" not in state["conversation_memory"]:
                state["conversation_memory"]["analysis_history"] = []
            
            # Append new analysis to existing history
            state["conversation_memory"]["analysis_history"].append(memory_data)
            
            # LIMIT HISTORY: Keep only last 3 conversations to avoid token limit issues
            if len(state["conversation_memory"]["analysis_history"]) > 3:
                # Remove oldest conversations, keep only last 3
                state["conversation_memory"]["analysis_history"] = state["conversation_memory"]["analysis_history"][-3:]
                print(f"DEBUG: SmartCodeExecutor - Trimmed history to last 3 conversations")
            
            # Update last_analysis
            state["conversation_memory"]["last_analysis"] = memory_data
            
            print(f"DEBUG: SmartCodeExecutor - Conversation memory keys after update: {list(state['conversation_memory'].keys())}")
            print(f"DEBUG: SmartCodeExecutor - Analysis history length: {len(state['conversation_memory']['analysis_history'])}")
            print(f"DEBUG: SmartCodeExecutor - History queries: {[item.get('user_query', 'Unknown') for item in state['conversation_memory']['analysis_history']]}")
            
            print(f"DEBUG: SmartCodeExecutor - Updated conversation memory. History now has {len(existing_history) + 1} entries")
            if existing_history:
                print(f"DEBUG: SmartCodeExecutor - First analysis: {existing_history[0].get('user_query', 'Unknown')}")
                print(f"DEBUG: SmartCodeExecutor - First data_relationship: {existing_history[0].get('data_relationship', 'Unknown')}")
            print(f"DEBUG: SmartCodeExecutor - Latest analysis: {memory_data.get('user_query', 'Unknown')}")
            print(f"DEBUG: SmartCodeExecutor - Latest data_relationship: {memory_data.get('data_relationship', 'Unknown')}")
            
            print(f"DEBUG: SmartCodeExecutor - Updated conversation memory. History now has {len(existing_history) + 1} entries")
            print(f"DEBUG: SmartCodeExecutor - First analysis: {existing_history[0] if existing_history else 'None'}")
            print(f"DEBUG: SmartCodeExecutor - Latest analysis: {memory_data}")
            
            # Also update previous_analysis field for backward compatibility
            state["previous_analysis"] = memory_data
            
            # CRITICAL: Save conversation memory to database for persistence
            try:
                print(f"DEBUG: SmartCodeExecutor - Attempting to import state_manager...")
                from state import state_manager
                print(f"DEBUG: SmartCodeExecutor - Successfully imported state_manager: {type(state_manager)}")
                print(f"DEBUG: SmartCodeExecutor - Calling update_conversation_memory with session_id: {state['session_id']}")
                # CRITICAL: Save the ENTIRE conversation memory structure, not just the new analysis
                # Create a clean copy of memory_data without circular references
                clean_memory_data = memory_data.copy()
                # Don't include analysis_history in the individual memory entry to avoid circular references
                if "analysis_history" in clean_memory_data:
                    del clean_memory_data["analysis_history"]
                
                # Ensure the memory data is fully serializable
                clean_memory_data = self._ensure_serializable(clean_memory_data)
                
                print(f"DEBUG: SmartCodeExecutor - Saving conversation memory: {clean_memory_data}")
                result = state_manager.update_conversation_memory(state["session_id"], clean_memory_data)
                print(f"DEBUG: SmartCodeExecutor - Saved conversation memory to database: {result}")
            except ImportError as import_error:
                print(f"ERROR: Could not import state_manager: {import_error}")
            except Exception as db_error:
                print(f"ERROR: Could not save conversation memory to database: {db_error}")
                import traceback
                traceback.print_exc()
            
            print(f"DEBUG: SmartCodeExecutor - Updated conversation memory with: {memory_data}")
            
        except Exception as e:
            print(f"Warning: Could not update conversation memory: {e}")
            # Don't fail the execution for memory update errors
    
    def _execute_code_safely(self, code: str, state: ConversationState) -> Dict[str, Any]:
        """Execute Python code safely with proper error handling"""
        try:
            # Create a safe execution environment with dfs dictionary (like backend approach)
            execution_env = self._get_execution_environment(state)
            
            # Execute the code
            print(f"DEBUG: SmartCodeExecutor - About to execute code with dfs keys: {list(execution_env.get('dfs', {}).keys())}")
            exec(code, execution_env)
            print(f"DEBUG: SmartCodeExecutor - Code execution completed. Available variables: {list(execution_env.keys())}")
            
            if 'result' not in execution_env and 'fig' in execution_env:
                execution_env['result'] = execution_env['fig']
            result = execution_env.get("result")
            # Get the result
            #result = execution_env.get('result', None)
            
            # Process the result based on its type
            processed_result = self._process_execution_result(result, execution_env)
            
            return {
                "success": True,
                "result": processed_result,
                "execution_env": {
                    "variables": list(execution_env.keys()),
                    "dataframes": [k for k, v in execution_env.items() if isinstance(v, pd.DataFrame)]
                }
            }
            
        except Exception as e:
            print(f"DEBUG: SmartCodeExecutor - EXECUTION ERROR: {str(e)}")
            print(f"DEBUG: SmartCodeExecutor - Error type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: SmartCodeExecutor - Full traceback: {traceback.format_exc()}")
            
            error_info = self._analyze_execution_error(e, code)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "error_analysis": error_info,
                "traceback": traceback.format_exc()
            }
    
    def _get_execution_environment(self, state: ConversationState) -> Dict[str, Any]:
        """Create a safe execution environment with necessary imports and data (EXACTLY like backend)"""
        print(f"DEBUG: SmartCodeExecutor - _get_execution_environment called")
        print(f"DEBUG: SmartCodeExecutor - State keys: {list(state.keys())}")
        print(f"DEBUG: SmartCodeExecutor - file_ids in state: {state.get('file_ids')}")
        
        env = {}
        
        # Add standard imports (EXACTLY like backend)
        env.update({
            'pd': pd,
            'np': np,
            'go': go,
            'math': math,
            'json': json
        })
        
        # Load DataFrames into dfs dictionary (EXACTLY like backend)
        if state.get("file_ids"):
            file_ids = state["file_ids"]
            print(f"DEBUG: SmartCodeExecutor - Loading DataFrames for file_ids: {file_ids}")
            
            # Load the dataframes (EXACTLY like backend)
            dfs = {}
            session_id = state.get("session_id")
            
            for file_id in file_ids:
                try:
                    # Strip quotes from file_id if present (like backend)
                    file_id = str(file_id).strip("'\"")
                    
                    from database import db_manager
                    file_info = db_manager.get_file_by_id(file_id, session_id)
                    if file_info:
                        # Use cleaned file if available, otherwise use uploaded file (like backend)
                        file_path = file_info.get('cleaned_filename') or file_info.get('uploaded_filename')
                        if file_path and os.path.exists(file_path):
                            #dfs[file_id] = pd.read_csv(file_path)
                            if file_path.lower().endswith(('.xlsx', '.xls')):
                                dfs[file_id] = pd.read_excel(file_path, engine='openpyxl')
                            else:
                                # Parse datetime columns automatically for CSV files
                                dfs[file_id] = pd.read_csv(file_path, low_memory=False)
                            print(f"DEBUG: SmartCodeExecutor - Loaded DataFrame {file_id} with shape: {dfs[file_id].shape}")
                        else:
                            print(f"WARNING: File path not found for {file_id}: {file_path}")
                    else:
                        print(f"WARNING: No file info found for {file_id}")
                except Exception as e:
                    print(f"ERROR: Failed to load DataFrame for {file_id}: {e}")
            
            # Create the execution environment (EXACTLY like backend)
            env["dfs"] = dfs
            print(f"DEBUG: SmartCodeExecutor - Added dfs dictionary with keys: {list(dfs.keys())}")
            print(f"DEBUG: SmartCodeExecutor - Available file_ids: {list(dfs.keys())}")
        
        return env
    
    def _process_execution_result(self, result: Any, execution_env: Dict[str, Any]) -> Dict[str, Any]:
        """Process the execution result into a structured format"""
        if result is None:
            return {"type": "none", "value": None, "message": "No result returned"}
        
        result_type = type(result).__name__
        
        if isinstance(result, pd.DataFrame):
            # Convert dtypes to serializable format
            dtypes_dict = result.dtypes.to_dict()
            serializable_dtypes = self._make_serializable(dtypes_dict)
            
            return {
                "type": "dataframe",
                "shape": result.shape,
                "columns": result.columns.tolist(),
                "head": result.head(5).to_dict(orient='records'),
                "dtypes": serializable_dtypes,
                "summary": result.describe().to_dict() if len(result.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
        
        elif isinstance(result, (pd.Series, pd.Index)):
            return {
                "type": "series",
                "length": len(result),
                "dtype": str(result.dtype),
                "head": result.head(10).tolist(),
                "summary": result.describe().to_dict() if hasattr(result, 'describe') else {}
            }
        
        elif isinstance(result, (int, float, str, bool)):
            return {
                "type": "scalar",
                "value": result,
                "python_type": result_type
            }
        
        elif isinstance(result, (list, tuple)):
            return {
                "type": "sequence",
                "length": len(result),
                "sample": result[:10] if len(result) > 10 else result,
                "python_type": result_type
            }
        
        elif isinstance(result, dict):
            return {
                "type": "dictionary",
                "keys": list(result.keys()),
                "sample": {k: str(v)[:100] + "..." if len(str(v)) > 100 else str(v) for k, v in list(result.items())[:5]},
                "python_type": result_type
            }
        
        elif hasattr(result, 'to_dict'):  # For plotly figures, etc.
            try:
                # For Plotly figures - return them directly like the backend does
                if hasattr(result, 'data') and hasattr(result, 'layout'):
                    # It's a Plotly figure - return it directly
                    print(f"DEBUG: SmartCodeExecutor - Found Plotly figure, returning to_dict() directly")
                    plot_data = result.to_dict()
                    
                    # Convert any non-serializable data to serializable format
                    serializable_plot_data = self._make_serializable(plot_data)
                    print(f"DEBUG: SmartCodeExecutor - Converted data to serializable format")
                    return serializable_plot_data
                else:
                    # Generic object with to_dict method
                    try:
                        dict_data = result.to_dict()
                        serializable_data = self._make_serializable(dict_data)
                        return {
                            "type": "object",
                            "class": result_type,
                            "data": serializable_data,
                            "representation": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                        }
                    except Exception as dict_error:
                        print(f"DEBUG: SmartCodeExecutor - Error converting to dict: {dict_error}")
                        return {
                            "type": "object",
                            "class": result_type,
                            "representation": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                        }
            except Exception as e:
                print(f"DEBUG: SmartCodeExecutor - Error in to_dict processing: {e}")
                pass
        
        # Default case - try to make it serializable
        try:
            serializable_result = self._make_serializable(result)
            return {
                "type": "object",
                "class": result_type,
                "data": serializable_result,
                "representation": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            }
        except Exception as e:
            print(f"DEBUG: Error in default serialization: {e}")
            return {
                "type": "object",
                "class": result_type,
                "representation": str(result)[:200] + "..." if len(str(result)) > 200 else str(result),
                "serialization_error": str(e)
            }
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert any object to a serializable format that LangGraph can handle.
        This method handles all common data types that cause serialization issues.
        
        Handles:
        - NumPy arrays, scalars, and objects
        - Pandas DataFrames, Series, and Index objects
        - Plotly figures and charts
        - Datetime objects
        - Complex numbers
        - Bytes and file objects
        - Custom objects with __dict__
        - Any other Python object type
        
        Returns a fully serializable Python object that LangGraph can checkpoint.
        """
        try:
            if obj is None:
                return None
            
            # Handle basic serializable types
            if isinstance(obj, (str, int, float, bool)):
                return obj
            
            # Handle NumPy types comprehensively
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                try:
                    return obj.tolist()
                except Exception as e:
                    print(f"DEBUG: Error converting NumPy array to list: {e}")
                    return str(obj)
            
            # Handle NumPy arrays and scalars (fallback)
            if hasattr(obj, 'tolist'):
                try:
                    return obj.tolist()
                except Exception as e:
                    print(f"DEBUG: Error converting NumPy object to list: {e}")
                    return str(obj)
            
            # Handle NumPy scalars (fallback)
            if hasattr(obj, 'item'):
                try:
                    return obj.item()
                except Exception as e:
                    print(f"DEBUG: Error converting NumPy scalar: {e}")
                    return str(obj)
            
            # Handle Pandas objects
            if hasattr(obj, 'to_dict'):
                try:
                    dict_data = obj.to_dict()
                    return self._make_serializable(dict_data)
                except Exception as e:
                    print(f"DEBUG: Error converting Pandas object to dict: {e}")
                    return str(obj)
            
            # Handle Pandas data types (dtype objects) - more comprehensive detection
            if hasattr(obj, '__class__') and ('dtype' in str(obj.__class__).lower() or 'pandas' in str(obj.__class__).lower()):
                try:
                    # Handle various Pandas dtype objects
                    dtype_info = {}
                    if hasattr(obj, 'name'):
                        dtype_info["name"] = str(obj.name)
                    if hasattr(obj, 'kind'):
                        dtype_info["kind"] = str(obj.kind)
                    if hasattr(obj, 'type'):
                        dtype_info["type"] = str(obj.type)
                    if hasattr(obj, 'dtype'):
                        dtype_info["dtype"] = str(obj.dtype)
                    
                    dtype_info.update({
                        "type": "pandas_dtype",
                        "class": str(obj.__class__),
                        "representation": str(obj)
                    })
                    return dtype_info
                except Exception as e:
                    print(f"DEBUG: Error converting Pandas dtype: {e}")
                    return {"type": "pandas_dtype", "representation": str(obj)}
            
            # Handle Pandas dtype objects specifically (like dtype('O'))
            if hasattr(obj, '__class__') and str(obj.__class__).startswith("<class 'pandas.core.dtypes.dtypes"):
                try:
                    return {
                        "type": "pandas_dtype_specific",
                        "class": str(obj.__class__),
                        "representation": str(obj),
                        "dtype_string": str(obj)
                    }
                except Exception as e:
                    print(f"DEBUG: Error converting specific Pandas dtype: {e}")
                    return {"type": "pandas_dtype_specific", "representation": str(obj)}
            
            # Handle NumPy data types
            if hasattr(obj, '__class__') and 'dtype' in str(obj.__class__).lower():
                try:
                    dtype_info = {}
                    if hasattr(obj, 'dtype'):
                        dtype_info["dtype"] = str(obj.dtype)
                    if hasattr(obj, 'type'):
                        dtype_info["type"] = str(obj.type)
                    if hasattr(obj, 'name'):
                        dtype_info["name"] = str(obj.name)
                    
                    dtype_info.update({
                        "type": "numpy_dtype",
                        "class": str(obj.__class__),
                        "representation": str(obj)
                    })
                    return dtype_info
                except Exception as e:
                    print(f"DEBUG: Error converting NumPy dtype: {e}")
                    return {"type": "numpy_dtype", "representation": str(obj)}
            
            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: self._make_serializable(v) for k, v in obj.items()}
            
            # Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                return [self._make_serializable(item) for item in obj]
            
            # Handle sets
            if isinstance(obj, set):
                return [self._make_serializable(item) for item in obj]
            
            # Handle datetime objects
            if hasattr(obj, 'isoformat'):
                try:
                    return obj.isoformat()
                except Exception as e:
                    print(f"DEBUG: Error converting datetime object: {e}")
                    return str(obj)
            
            # Handle complex numbers
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag, "type": "complex"}
            
            # Handle bytes
            if isinstance(obj, bytes):
                try:
                    return obj.decode('utf-8')
                except Exception as e:
                    print(f"DEBUG: Error decoding bytes: {e}")
                    return str(obj)
            
            # Handle file objects
            if hasattr(obj, 'read'):
                return {"type": "file_object", "name": getattr(obj, 'name', 'unknown')}
            
            # Handle any object with __dict__
            if hasattr(obj, '__dict__'):
                try:
                    dict_data = obj.__dict__.copy()
                    # Remove any non-serializable attributes
                    clean_dict = {}
                    for k, v in dict_data.items():
                        if not k.startswith('_'):  # Skip private attributes
                            clean_dict[k] = self._make_serializable(v)
                    return clean_dict
                except Exception as e:
                    print(f"DEBUG: Error converting object with __dict__: {e}")
                    return {"type": str(type(obj)), "representation": str(obj)}
            
            # Handle any other object type
            try:
                # Special handling for Pandas/NumPy objects that might have been missed
                if hasattr(obj, '__class__'):
                    class_name = str(obj.__class__).lower()
                    class_str = str(obj.__class__)
                    
                    # Check for Pandas objects including dtypes
                    if 'pandas' in class_name or 'dtype' in class_name or 'pandas.core.dtypes' in class_str:
                        return {
                            "type": "pandas_object",
                            "class": str(obj.__class__),
                            "representation": str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
                        }
                    
                    # Check for NumPy objects
                    if 'numpy' in class_name:
                        return {
                            "type": "numpy_object", 
                            "class": str(obj.__class__),
                            "representation": str(obj)[:200] + "..." if len(str(obj)) > 200 else str(obj)
                        }
                    
                    # Check for any object that looks like a dtype
                    if hasattr(obj, '__str__') and str(obj).startswith('dtype('):
                        return {
                            "type": "dtype_object",
                            "class": str(obj.__class__),
                            "representation": str(obj)
                        }
                
                # Final fallback - convert to string
                return str(obj)
                
            except Exception as e:
                print(f"DEBUG: Error in final object handling: {e}")
                return {"type": str(type(obj)), "representation": str(obj), "error": str(e)}
                
        except Exception as e:
            print(f"DEBUG: Error in _make_serializable: {e}")
            return {"type": "serialization_error", "error": str(e), "original_type": str(type(obj))}
    
    def _ensure_serializable(self, obj: Any) -> Any:
        """
        Final safety check to ensure an object is fully serializable.
        This method catches any NumPy types that might have slipped through.
        """
        try:
            import numpy as np
            
            # Check for any remaining NumPy types
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Recursively check dictionaries and lists
            if isinstance(obj, dict):
                return {k: self._ensure_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._ensure_serializable(item) for item in obj]
            
            return obj
            
        except Exception as e:
            print(f"DEBUG: Error in _ensure_serializable: {e}")
            return str(obj)
    
    def _analyze_execution_error(self, error: Exception, code: str) -> Dict[str, Any]:
        """Analyze execution errors to provide helpful feedback"""
        error_str = str(error).lower()
        
        analysis = {
            "error_category": "unknown",
            "suggestions": [],
            "common_causes": []
        }
        
        # Categorize common errors
        if "nameerror" in error_str or "name" in error_str:
            analysis["error_category"] = "name_error"
            analysis["suggestions"].extend([
                "Check that all variables are defined before use",
                "Ensure all required imports are included",
                "Verify column names match your dataset"
            ])
            analysis["common_causes"].extend([
                "Undefined variable",
                "Missing import",
                "Typo in variable name"
            ])
        
        elif "attributeerror" in error_str or "attribute" in error_str:
            analysis["error_category"] = "attribute_error"
            analysis["suggestions"].extend([
                "Check the data type of your variables",
                "Ensure you're calling methods on the right object type",
                "Verify the object has the method you're trying to use"
            ])
            analysis["common_causes"].extend([
                "Wrong object type",
                "Method doesn't exist",
                "Data type mismatch"
            ])
        
        elif "typeerror" in error_str or "type" in error_str:
            analysis["error_category"] = "type_error"
            analysis["suggestions"].extend([
                "Check data types of your variables",
                "Ensure numeric operations use numeric data",
                "Convert data types if needed using .astype()"
            ])
            analysis["common_causes"].extend([
                "Data type mismatch",
                "Numeric operation on non-numeric data",
                "Incompatible types for operation"
            ])
        
        elif "indexerror" in error_str or "index" in error_str:
            analysis["error_category"] = "index_error"
            analysis["suggestions"].extend([
                "Check the size of your dataframes",
                "Verify index values are within bounds",
                "Use .iloc[] or .loc[] for safe indexing"
            ])
            analysis["common_causes"].extend([
                "Index out of bounds",
                "Empty dataframe",
                "Wrong index reference"
            ])
        
        elif "keyerror" in error_str or "key" in error_str:
            analysis["error_category"] = "key_error"
            analysis["suggestions"].extend([
                "Check column names in your dataset",
                "Use .columns to see available columns",
                "Verify column names match exactly (case-sensitive)"
            ])
            analysis["common_causes"].extend([
                "Column doesn't exist",
                "Typo in column name",
                "Case sensitivity issue"
            ])
        
        elif "valueerror" in error_str or "value" in error_str:
            analysis["error_category"] = "value_error"
            analysis["suggestions"].extend([
                "Check the values in your data",
                "Handle missing or invalid data",
                "Verify data meets function requirements"
            ])
            analysis["common_causes"].extend([
                "Invalid data values",
                "Missing data",
                "Data format issues"
            ])
        
        # Add general suggestions
        analysis["suggestions"].extend([
            "Check the error message for specific details",
            "Verify your data structure matches expectations",
            "Try running the code step by step to isolate the issue"
        ])
        
        return analysis

# Factory function for creating the node
def smart_code_executor_node():
    """Create a smart code executor node instance"""
    return SmartCodeExecutorNode() 