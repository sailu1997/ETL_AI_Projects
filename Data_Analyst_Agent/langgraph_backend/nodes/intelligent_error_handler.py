"""
Intelligent Error Handler Node

This single node handles the complete ReACT error handling flow:
- Analyzes execution errors using AI
- Generates intelligent fixes using AI
- Applies fixes and retries execution
- Manages retry loop until success or max attempts
- Only activated after smart_code_executor when errors occur
"""

import re
from typing import Dict, Any, List
from .base_node import BaseNode
from state import ConversationState
import openai
import pandas as pd


class IntelligentErrorHandlerNode(BaseNode):
    """Single node for intelligent error handling and self-correction"""
    
    def __init__(self):
        super().__init__(
            name="intelligent_error_handler",
            description="AI-powered error analysis, fixing, and retry logic"
        )
    
    def get_required_fields(self) -> List[str]:
        """Return the required fields for this node."""
        return ["session_id", "user_query"]
    
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors intelligently using AI-powered ReACT approach"""
        try:
            # Use print instead of logger for now
            print(f"DEBUG: IntelligentErrorHandler - Starting execution")
            
            # Check if we have an execution result with an error
            execution_result = state.get("result", {}).get("execution_result", {})
            
            if execution_result.get("success", False):
                # No error - this shouldn't happen, but just return
                print(f"DEBUG: IntelligentErrorHandler - No error to handle")
                return state
            
            # Get error details
            error_message = execution_result.get("error", "")
            error_type = execution_result.get("error_type", "unknown")
            
            print(f"DEBUG: IntelligentErrorHandler - Handling error: {error_type}")
            print(f"DEBUG: IntelligentErrorHandler - Error message: {error_message}")
            
            # Initialize retry tracking - CRITICAL FIX: Get current attempt from state
            current_attempt = state.get("fix_attempt", 0)
            max_attempts = 3
            
            print(f"DEBUG: IntelligentErrorHandler - Attempt {current_attempt + 1} of {max_attempts}")
            
            # Check if we've exceeded max attempts
            if current_attempt >= max_attempts:
                print(f"DEBUG: IntelligentErrorHandler - Max attempts reached, giving up")
                state["result"]["execution_result"] = {
                    "success": False,
                    "error": f"Failed after {max_attempts} fix attempts",
                    "requires_human_intervention": True,
                    "fix_attempts": current_attempt,
                    "final_attempt": True
                }
                return state
            
            # Use AI to analyze the error and generate a fix
            print(f"DEBUG: IntelligentErrorHandler - Using AI to analyze and fix error")
            
            # Step 1: AI Error Analysis
            error_analysis = self._ai_analyze_error(error_message, error_type, state)
            
            # Step 2: AI Code Fixing - CRITICAL: Pass the latest working code version
            fixed_code = self._ai_fix_code(error_analysis, state)
            
            if not fixed_code:
                print(f"DEBUG: IntelligentErrorHandler - AI failed to generate fix")
                state["result"]["execution_result"] = {
                    "success": False,
                    "error": "AI failed to generate code fix",
                    "requires_human_intervention": True
                }
                return state
            
            # Step 3: Execute the fixed code
            print(f"DEBUG: IntelligentErrorHandler - Executing fixed code")
            execution_result = self._execute_fixed_code(fixed_code, state)
            
            # Store the execution result
            state["result"]["execution_result"] = execution_result
            
            # CRITICAL FIX: Increment fix_attempt BEFORE checking success
            state["fix_attempt"] = current_attempt + 1
            print(f"DEBUG: IntelligentErrorHandler - Updated fix_attempt to: {state['fix_attempt']}")
            
            # CRITICAL: Store the fixed code for future iterations
            if "fix_history" not in state:
                state["fix_history"] = []
            
            # Add this fix to the history
            fix_entry = {
                "attempt": state["fix_attempt"],
                "error_type": error_type,
                "error_message": error_message,
                "fixed_code": fixed_code,
                "timestamp": self._get_timestamp()
            }
            state["fix_history"].append(fix_entry)
            print(f"DEBUG: IntelligentErrorHandler - Added fix to history. Total fixes: {len(state['fix_history'])}")
            
            # Check if execution was successful
            if execution_result.get("success", False):
                print(f"DEBUG: IntelligentErrorHandler - Fixed code executed successfully!")
                state["result"]["reasoning"] = f"Code executed successfully after {state['fix_attempt']} fix attempts"
                state["result"]["fix_summary"] = {
                    "fixes_applied": error_analysis.get("suggested_fixes", []),
                    "attempts_required": state['fix_attempt'],
                    "final_success": True,
                    "fix_history": state["fix_history"]
                }
                return state
            else:
                print(f"DEBUG: IntelligentErrorHandler - Fixed code still failed")
                print(f"DEBUG: IntelligentErrorHandler - Error: {execution_result.get('error', 'Unknown error')}")
                
                # Check if we should retry again
                if self._should_retry_again(execution_result, state['fix_attempt'], max_attempts):
                    print(f"DEBUG: IntelligentErrorHandler - Will retry again")
                    # The workflow will loop back to this node
                    return state
                else:
                    print(f"DEBUG: IntelligentErrorHandler - Giving up, requires human intervention")
                    state["result"]["execution_result"]["requires_human_intervention"] = True
                    state["result"]["execution_result"]["final_attempt"] = True
                    return state
            
        except Exception as e:
            error_msg = f"Error in IntelligentErrorHandlerNode: {str(e)}"
            print(f"DEBUG: {error_msg}")
            return self.handle_error(state, e)
    
    def _ai_analyze_error(self, error_message: str, error_type: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to intelligently analyze the error"""
        try:
            # Build context for AI analysis
            context = self._build_ai_context(state)
            
            # Create prompt for error analysis
            prompt = f"""You are an expert Python error analyst. Analyze this error and provide intelligent insights.

ERROR DETAILS:
- Error Type: {error_type}
- Error Message: {error_message}

CONTEXT:
- User Query: {context.get('user_query', 'Unknown')}
- File Context: {context.get('file_context', 'Unknown')}

ANALYZE:
1. What caused this error?
2. What type of fix is needed?
3. Can this be automatically fixed?
4. What specific actions should be taken?

Provide your analysis in this format:
ERROR_CLASSIFICATION: [datetime_parsing_error|missing_column_error|import_error|type_mismatch_error|attribute_error|syntax_error|unknown_error]
CAN_AUTO_FIX: [true|false]
CONFIDENCE: [0.0-1.0]
ROOT_CAUSE: [brief explanation]
SUGGESTED_FIXES: [list of specific fixes]
"""
            
            # Call OpenAI for error analysis
            analysis_response = self._call_openai(prompt, "error_analysis")
            
            if analysis_response:
                # Parse the AI response
                parsed_analysis = self._parse_ai_analysis(analysis_response)
                return parsed_analysis
            else:
                # Fallback to basic analysis
                return self._fallback_error_analysis(error_message, error_type)
                
        except Exception as e:
            print(f"DEBUG: IntelligentErrorHandler - AI error analysis failed: {e}")
            return self._fallback_error_analysis(error_message, error_type)
    
    def _ai_fix_code(self, error_analysis: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Use AI to generate fixed code"""
        try:
            # Get the latest working code version from fix history, or original if no history
            fix_history = state.get("fix_history", [])
            
            if fix_history:
                # Use the most recent fixed code as the base
                latest_fix = fix_history[-1]
                base_code = latest_fix["fixed_code"]
                print(f"DEBUG: IntelligentErrorHandler - Using latest fixed code from attempt {latest_fix['attempt']} as base")
            else:
                # Use original code if no fix history
                generated_code = state.get("result", {}).get("generated_code", {})
                base_code = generated_code.get("code", "")
                print(f"DEBUG: IntelligentErrorHandler - Using original code as base (no fix history)")
            
            if not base_code:
                print(f"DEBUG: IntelligentErrorHandler - No base code available for fixing")
                return None
            
            # Build context for AI fixing
            context = self._build_ai_context(state)
            
            # Create prompt for code fixing with better error handling
            prompt = f"""You are an expert Python code fixer. Fix this code based on the error analysis.

ERROR ANALYSIS:
{error_analysis}

USER QUERY: {context.get('user_query', 'Unknown')}
FILE CONTEXT: {context.get('file_context', 'Unknown')}

BASE CODE (this is the latest working version, build upon it):
```python
{base_code}
```

CRITICAL REQUIREMENTS:
1. Fix the specific error identified in the analysis
2. Ensure the fixed code will execute successfully
3. Keep the same logic and functionality
4. Only fix what's necessary - don't rewrite the entire code
5. Add any necessary imports or data type conversions
6. **MOST IMPORTANT**: The code MUST create a variable called 'result' that contains the final output
7. **MOST IMPORTANT**: Use the correct file ID from the context: {state.get('file_ids', [])}
8. **MOST IMPORTANT**: Access the DataFrame using: df = dfs['{state.get('file_ids', [''])[0] if state.get('file_ids') else ''}']
9. **MOST IMPORTANT**: The 'result' variable should contain the filtered/processed data
10. **MOST IMPORTANT**: Build upon the base code above - don't start from scratch

EXAMPLE OF CORRECT STRUCTURE:
```python
# Load the DataFrame
df = dfs['{state.get('file_ids', [''])[0] if state.get('file_ids') else ''}']

# Your analysis code here (build upon the base code)
# ... (filtering, processing, etc.)

# CRITICAL: Create the result variable
result = filtered_data  # or whatever your final output is
```

Return ONLY the fixed Python code, no explanations.

FIXED CODE:
```python
"""
            
            # Call OpenAI for code fixing
            fixed_code = self._call_openai(prompt, "code_fix")
            
            if fixed_code:
                # Clean up the response
                if fixed_code.startswith("```python"):
                    fixed_code = fixed_code[9:]
                if fixed_code.endswith("```"):
                    fixed_code = fixed_code[:-3]
                
                fixed_code = fixed_code.strip()
                
                print(f"DEBUG: IntelligentErrorHandler - AI generated fixed code: {len(fixed_code)} chars")
                print(f"DEBUG: IntelligentErrorHandler - Fixed code preview: {fixed_code[:200]}...")
                return fixed_code
            else:
                return None
                
        except Exception as e:
            print(f"DEBUG: IntelligentErrorHandler - AI code fixing failed: {e}")
            return None
    
    def _execute_fixed_code(self, fixed_code: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the fixed code using the same logic as SmartCodeExecutor"""
        try:
            # Create execution environment
            execution_env = self._get_execution_environment(state)
            
            print(f"DEBUG: IntelligentErrorHandler - Executing fixed code with dfs keys: {list(execution_env.get('dfs', {}).keys())}")
            
            # Execute the fixed code
            exec(fixed_code, execution_env)
            
            print(f"DEBUG: IntelligentErrorHandler - Code execution completed. Available variables: {list(execution_env.keys())}")
            
            # Check if 'result' variable was created
            if 'result' in execution_env:
                result = execution_env['result']
                print(f"DEBUG: IntelligentErrorHandler - Found 'result' variable: {type(result)}")
                
                # Process the result
                processed_result = self._process_execution_result(result, execution_env)
                
                return {
                    "success": True,
                    "result": processed_result,
                    "execution_env": {
                        "variables": list(execution_env.keys()),
                        "dataframes": [k for k, v in execution_env.items() if isinstance(v, pd.DataFrame)]
                    },
                    "fix_attempt": state.get("fix_attempt", 0)
                }
            else:
                print(f"DEBUG: IntelligentErrorHandler - No 'result' variable found in execution environment")
                return {
                    "success": False,
                    "error": "No 'result' variable found in executed code",
                    "fix_attempt": state.get("fix_attempt", 0)
                }
                
        except Exception as e:
            print(f"DEBUG: IntelligentErrorHandler - EXECUTION ERROR: {str(e)}")
            print(f"DEBUG: IntelligentErrorHandler - Error type: {type(e).__name__}")
            import traceback
            print(f"DEBUG: IntelligentErrorHandler - Full traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "fix_attempt": state.get("fix_attempt", 0)
            }
    
    def _build_ai_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context information for AI analysis"""
        context = {
            "user_query": state.get("user_query", ""),
            "file_ids": state.get("file_ids", [])
        }
        
        # Add file metadata if available
        if context["file_ids"]:
            try:
                from database import db_manager
                file_metadata = []
                for file_id in context["file_ids"]:
                    file_info = db_manager.get_file_by_id(file_id, "temp_session")
                    if file_info and file_info.get("metadata"):
                        file_metadata.append({
                            "file_id": file_id,
                            "columns": file_info["metadata"].get("columns", []),
                            "data_types": file_info["metadata"].get("data_types", {})
                        })
                context["file_context"] = file_metadata
            except Exception as e:
                print(f"DEBUG: IntelligentErrorHandler - Could not load file metadata: {e}")
                context["file_context"] = "Metadata unavailable"
        
        return context
    
    def _call_openai(self, prompt: str, purpose: str) -> str:
        """Call OpenAI API for AI-powered analysis or fixing"""
        try:
            # Get OpenAI API key from environment
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print(f"DEBUG: IntelligentErrorHandler - No OpenAI API key found")
                return None
            
            # Configure OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert Python {'error analyst' if purpose == 'error_analysis' else 'code fixer'}. Provide clear, actionable {'analysis' if purpose == 'error_analysis' else 'fixed code'}."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=2000
            )
            
            # Extract the response
            ai_response = response.choices[0].message.content.strip()
            print(f"DEBUG: IntelligentErrorHandler - AI response for {purpose}: {len(ai_response)} chars")
            return ai_response
            
        except Exception as e:
            print(f"DEBUG: IntelligentErrorHandler - OpenAI API call failed: {str(e)}")
            return None
    
    def _parse_ai_analysis(self, analysis_response: str) -> Dict[str, Any]:
        """Parse the AI error analysis response"""
        try:
            # Simple parsing of the structured response
            lines = analysis_response.split('\n')
            analysis = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    
                    if key == 'can_auto_fix':
                        analysis[key] = value.lower() == 'true'
                    elif key == 'confidence':
                        analysis[key] = float(value) if value.replace('.', '').isdigit() else 0.5
                    elif key == 'suggested_fixes':
                        # Parse list format
                        fixes = [fix.strip() for fix in value.strip('[]').split(',') if fix.strip()]
                        analysis[key] = fixes
                    else:
                        analysis[key] = value
            
            # Ensure all required fields are present
            required_fields = ['error_classification', 'can_auto_fix', 'confidence', 'root_cause', 'suggested_fixes']
            for field in required_fields:
                if field not in analysis:
                    if field == 'can_auto_fix':
                        analysis[field] = False
                    elif field == 'confidence':
                        analysis[field] = 0.5
                    elif field == 'suggested_fixes':
                        analysis[field] = ['Manual review required']
                    else:
                        analysis[field] = 'Unknown'
            
            return analysis
            
        except Exception as e:
            print(f"DEBUG: IntelligentErrorHandler - Failed to parse AI analysis: {e}")
            return self._fallback_error_analysis("", "unknown")
    
    def _fallback_error_analysis(self, error_message: str, error_type: str) -> Dict[str, Any]:
        """Fallback error analysis when AI fails"""
        error_str = str(error_message).lower()
        
        # Basic classification
        if "string vs timestamp" in error_str or "str vs timestamp" in error_str:
            classification = "datetime_parsing_error"
        elif "column not found" in error_str:
            classification = "missing_column_error"
        elif "import error" in error_str:
            classification = "import_error"
        else:
            classification = "unknown_error"
        
        return {
            "error_classification": classification,
            "can_auto_fix": classification in ["datetime_parsing_error", "import_error"],
            "confidence": 0.6,
            "root_cause": f"Basic analysis: {error_type}",
            "suggested_fixes": ["Manual review recommended"]
        }
    
    def _get_execution_environment(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution environment similar to SmartCodeExecutor"""
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import math
        import json
        import os
        
        env = {}
        
        # Add standard imports
        env.update({
            'pd': pd,
            'np': np,
            'go': go,
            'math': math,
            'json': json
        })
        
        # Load DataFrames into dfs dictionary
        if state.get("file_ids"):
            file_ids = state["file_ids"]
            print(f"DEBUG: IntelligentErrorHandler - Loading DataFrames for file_ids: {file_ids}")
            
            dfs = {}
            session_id = state.get("session_id")
            
            for file_id in file_ids:
                try:
                    file_id = str(file_id).strip("'\"")
                    
                    from database import db_manager
                    file_info = db_manager.get_file_by_id(file_id, session_id)
                    if file_info:
                        file_path = file_info.get('cleaned_filename') or file_info.get('uploaded_filename')
                        if file_path and os.path.exists(file_path):
                            if file_path.lower().endswith(('.xlsx', '.xls')):
                                dfs[file_id] = pd.read_excel(file_path, engine='openpyxl')
                            else:
                                try:
                                    dfs[file_id] = pd.read_csv(
                                        file_path,
                                        low_memory=False,
                                        parse_dates=['Collected Date', 'Discharged Date', 'Initiated Date']
                                    )
                                except Exception:
                                    dfs[file_id] = pd.read_csv(file_path, low_memory=False)
                            print(f"DEBUG: IntelligentErrorHandler - Loaded DataFrame {file_id} with shape: {dfs[file_id].shape}")
                        else:
                            print(f"WARNING: File path not found for {file_id}: {file_path}")
                    else:
                        print(f"WARNING: No file info found for {file_id}")
                except Exception as e:
                    print(f"ERROR: Failed to load DataFrame for {file_id}: {e}")
            
            env["dfs"] = dfs
            print(f"DEBUG: IntelligentErrorHandler - Added dfs dictionary with keys: {list(dfs.keys())}")
        
        return env
    
    def _process_execution_result(self, result: Any, execution_env: Dict[str, Any]) -> Dict[str, Any]:
        """Process the execution result into a structured format"""
        if result is None:
            return {"type": "none", "value": None, "message": "No result returned"}
        
        result_type = type(result).__name__
        
        if hasattr(result, 'shape') and hasattr(result, 'columns'):  # DataFrame
            return {
                "type": "dataframe",
                "shape": result.shape,
                "columns": result.columns.tolist(),
                "head": result.head(5).to_dict(orient='records'),
                "dtypes": {k: str(v) for k, v in result.dtypes.items()}
            }
        elif isinstance(result, (pd.Series, pd.Index)):
            return {
                "type": "series",
                "length": len(result),
                "dtype": str(result.dtype),
                "head": result.head(10).tolist()
            }
        elif isinstance(result, (int, float, str, bool)):
            return {
                "type": "scalar",
                "value": result,
                "python_type": result_type
            }
        else:
            return {
                "type": "unknown",
                "value": str(result),
                "python_type": result_type
            }
    
    def _should_retry_again(self, execution_result: Dict[str, Any], current_attempt: int, max_attempts: int) -> bool:
        """Determine if we should retry again"""
        print(f"DEBUG: IntelligentErrorHandler - _should_retry_again called with attempt {current_attempt} of {max_attempts}")
        
        # Don't retry if we've reached max attempts
        if current_attempt >= max_attempts:
            print(f"DEBUG: IntelligentErrorHandler - Max attempts reached, no more retries")
            return False
        
        # Don't retry if the error is clearly unfixable
        error_message = str(execution_result.get("error", "")).lower()
        
        unfixable_errors = [
            "file not found",
            "permission denied",
            "out of memory",
            "syntax error",
            "invalid syntax",
            "keyerror",  # Column/attribute not found
            "attributeerror",  # Object has no attribute
            "nameerror",  # Name not defined
            "import error",  # Module not found
            "no module named"  # Import issues
        ]
        
        for unfixable in unfixable_errors:
            if unfixable in error_message:
                print(f"DEBUG: IntelligentErrorHandler - Unfixable error detected: {unfixable}")
                return False
        
        # Don't retry if we're getting the same error repeatedly (likely unfixable)
        if current_attempt >= 2:  # After 2 attempts, be more conservative about retries
            print(f"DEBUG: IntelligentErrorHandler - High attempt count, being conservative about retries")
            return False
        
        # Retry for other types of errors
        print(f"DEBUG: IntelligentErrorHandler - Will retry again")
        return True

    def _get_timestamp(self) -> str:
        """Return current UTC timestamp as ISO 8601 string"""
        try:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).isoformat()
        except Exception:
            from datetime import datetime
            return datetime.utcnow().isoformat()


# Factory function for creating the node
def intelligent_error_handler_node():
    return IntelligentErrorHandlerNode()
