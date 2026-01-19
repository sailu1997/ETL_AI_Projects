"""
Smart Code Generator Node

This node generates Python code for data analysis tasks using LLM.
Based on the SmartCodeGeneratorToolLC from LangChain.
"""

import os
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, Any, List
from .base_node import BaseNode
from state import ConversationState, state_manager
from dotenv import load_dotenv

load_dotenv()

class SmartCodeGeneratorNode(BaseNode):
    """Node for generating smart code for data analysis using LLM"""
    
    def __init__(self):
        super().__init__(
            name="smart_code_generator",
            description="Intelligently generates Python code to analyze dataframes using LLM"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "user_query", "file_ids"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Generate smart code for data analysis using LLM"""
        try:
            print("=== DEBUG: SmartCodeGenerator execute method called ===")
            
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for smart_code_generator node")
            
            self.log_execution(state)
            
            session_id = state["session_id"]
            user_query = state["user_query"]
            file_ids = state.get("file_ids", [])
            
            # Get query classification from state
            query_classification = {}
            conversation_context = {}
            if "query_classification" in state:
                query_classification = state["query_classification"]
            
            # Load conversation context from conversation memory and previous analysis
            conversation_memory = state.get("conversation_memory", {})
            previous_analysis = state.get("previous_analysis", {})
            
            print(f"DEBUG: SmartCodeGenerator - Raw conversation_memory: {conversation_memory}")
            print(f"DEBUG: SmartCodeGenerator - Raw previous_analysis: {previous_analysis}")
            
            # Build comprehensive conversation context
            if conversation_memory or previous_analysis:
                conversation_context = {
                    "is_follow_up": len(conversation_memory.get("analysis_history", [])) > 0,
                    "context_clues": self._extract_context_clues(user_query, conversation_memory, previous_analysis),
                    "previous_analysis": previous_analysis,
                    "data_relationships": self._extract_data_relationships(conversation_memory, previous_analysis),
                    "conversation_memory": conversation_memory
                }
                
                print(f"DEBUG: SmartCodeGenerator - Built conversation context: {conversation_context}")
            else:
                print(f"DEBUG: SmartCodeGenerator - No conversation memory or previous analysis found")
                conversation_context = {}
            
            # Use file IDs for on-demand loading (no DataFrame storage in state)
            print(f"DEBUG: SmartCodeGenerator - Session ID: {session_id}")
            print(f"DEBUG: SmartCodeGenerator - File IDs available: {file_ids}")
            print(f"DEBUG: SmartCodeGenerator - Number of files: {len(file_ids)}")
            print(f"DEBUG: SmartCodeGenerator - Conversation context: {conversation_context}")
            
            # Use metadata from conversation context if available, otherwise extract on-demand
            if conversation_context and conversation_context.get("file_metadata"):
                file_metadata = conversation_context["file_metadata"]
                print(f"DEBUG: SmartCodeGenerator - Using metadata from conversation context for {len(file_metadata)} files")
            else:
                file_metadata = self._extract_file_metadata_from_ids(file_ids, session_id)
                print(f"DEBUG: SmartCodeGenerator - Extracted file metadata on-demand for {len(file_metadata)} files")
            
            # Determine the type of output needed based on query classification
            print(f"DEBUG: SmartCodeGenerator - Full query_classification: {query_classification}")
            print(f"DEBUG: SmartCodeGenerator - Query classification type: {type(query_classification)}")
            print(f"DEBUG: SmartCodeGenerator - Query classification keys: {list(query_classification.keys()) if query_classification else 'None'}")
            
            response_format = query_classification.get("response_format", "mixed")
            query_type = query_classification.get("query_type", "general_analysis")
            
            print(f"DEBUG: SmartCodeGenerator - Response format: {response_format}")
            print(f"DEBUG: SmartCodeGenerator - Query type: {query_type}")
            
            should_plot = (
                response_format == "plot" or 
                query_type == "visualization"
            )
            
            should_generate_table = (
                response_format == "table" or
                query_type in ["data_filtering", "aggregation", "statistical_analysis"]
            )
            
            print(f"DEBUG: SmartCodeGenerator - Should plot: {should_plot}")
            print(f"DEBUG: SmartCodeGenerator - Should generate table: {should_generate_table}")
            
            # Generate reasoning based on query classification
            if should_plot:
                reasoning = f"Query analysis: The query '{user_query}' is classified as '{query_type}' and will be visualized with a plot (response_format: {response_format}) to provide better insights and understanding of the data."
            elif should_generate_table:
                reasoning = f"Query analysis: The query '{user_query}' is classified as '{query_type}' and will generate a structured table (response_format: {response_format}) with the requested data and analysis."
            else:
                reasoning = f"Query analysis: The query '{user_query}' is classified as '{query_type}' and will be analyzed using data operations (response_format: {response_format}) to provide insights and results."
            
            # Generate the prompt using backend logic
            print(f"DEBUG: SmartCodeGenerator - About to generate prompt...")
            try:
                prompt = self._generate_code_prompt(file_metadata, user_query, should_plot, should_generate_table, conversation_context)
                print(f"DEBUG: SmartCodeGenerator - Generated prompt length: {len(prompt)}")
                print(f"DEBUG: SmartCodeGenerator - Prompt preview: {prompt[:200]}...")
            except Exception as e:
                print(f"DEBUG: SmartCodeGenerator - Error generating prompt: {e}")
                raise e
            
            # Call OpenAI to generate code (as per backend)
            print(f"DEBUG: SmartCodeGenerator - About to import OpenAI...")
            from openai import OpenAI
            print(f"DEBUG: SmartCodeGenerator - About to create OpenAI client...")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            print(f"DEBUG: SmartCodeGenerator - Calling OpenAI API...")
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert. Generate only valid Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                print(f"DEBUG: SmartCodeGenerator - OpenAI API call successful")
            except Exception as e:
                print(f"DEBUG: SmartCodeGenerator - Error calling OpenAI API: {e}")
                raise e
            
            raw_code = response.choices[0].message.content
            print(f"DEBUG: SmartCodeGenerator - Raw OpenAI response: {raw_code}")
            
            code = self._extract_first_code_block(raw_code)
            print(f"DEBUG: SmartCodeGenerator - Extracted code block: {code}")
            
            # Preprocess the code to fix common issues (as per backend)
            code = self._preprocess_code(code)
            print(f"DEBUG: SmartCodeGenerator - Preprocessed code: {code}")
            
            # Store results in result field
            print(f"DEBUG: SmartCodeGenerator - About to store generated code")
            if state.get("result") is None:
                state["result"] = {}
            
            print(f"DEBUG: SmartCodeGenerator - Storing generated_code with code: {code}")
            state["result"]["generated_code"] = {
                "code": code,
                "query": user_query,
                "file_ids": file_ids,
                "should_plot": should_plot,
                "reasoning": reasoning
            }
            
            print(f"DEBUG: SmartCodeGenerator - Set current_step to code_generated")
            state["current_step"] = "code_generated"
            
            # Add reasoning to result
            print(f"DEBUG: SmartCodeGenerator - Adding reasoning: {reasoning}")
            state["result"]["reasoning"] = reasoning
            
            # Add AI message to conversation
            print(f"DEBUG: SmartCodeGenerator - Adding AI message to conversation")
            from langchain_core.messages import AIMessage
            ai_message = f"Code generated successfully for your {query_classification.get('query_type', 'analysis')} request."
            state["messages"].append(AIMessage(content=ai_message))
            
            print(f"DEBUG: SmartCodeGenerator - About to return state")
            self.log_execution(state, "Code generated successfully")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _generate_code_prompt(self, dfs_metadata: dict, query: str, should_plot: bool, should_generate_table: bool, conversation_context: dict = None) -> str:
        """Generate comprehensive prompt for code generation with conversation context"""
        # Build dataset information from metadata instead of actual DataFrames
        df_profiles = []
        for file_id, metadata in dfs_metadata.items():
            # Extract information from metadata
            columns = metadata.get('columns', [])
            shape = metadata.get('shape', (0, 0))
            dtypes = metadata.get('dtypes', {})
            null_counts = metadata.get('null_counts', {})
            head_sample = metadata.get('head_sample', [])
            file_path = metadata.get('file_path', '')
            
            # Create categorical analysis from head sample
            categorical_analysis = {}
            for col in columns:
                if dtypes.get(col) == 'object':  # Categorical column
                    # Count unique values from head sample
                    unique_values = set()
                    for row in head_sample:
                        if col in row and row[col] is not None:
                            unique_values.add(str(row[col]))
                    categorical_analysis[col] = list(unique_values)[:5]  # Top 5 unique values

            data_profile = {
                "row_count": shape[0] if shape else 0,
                "column_counts": {col: metadata.get('unique_counts', {}).get(col, 0) for col in columns},
                "missing_values": null_counts,
                "data_types": dtypes,
                "categorical_values": categorical_analysis,
                "file_path": file_path
            }

            df_profiles.append(f"""
            ### DataFrame: `{file_id}`
            - File Path: {file_path}
            - Columns: {', '.join(columns)}
            - Row count: {data_profile['row_count']}
            - Missing values: {data_profile['missing_values']}
            - Data types: {data_profile['data_types']}
            - Top Values in Categorical Columns: {data_profile['categorical_values']}
            """)
        all_profiles = "\n".join(df_profiles)

        # Build conversation context section
        context_section = ""
        if conversation_context:
            context_section = f"""
            ### CONVERSATION CONTEXT:
            - Is this a follow-up query: {conversation_context.get('is_follow_up', False)}
            - Context clues: {conversation_context.get('context_clues', 'None')}
            - Previous analysis: {conversation_context.get('previous_analysis', {})}
            - Data relationships: {conversation_context.get('data_relationships', {})}
            - File metadata: {conversation_context.get('file_metadata', {})}
            
            ### CONTEXT INTERPRETATION:
            """
            
            # Add intelligent context interpretation
            if conversation_context.get('is_follow_up', False):
                previous_analysis = conversation_context.get('previous_analysis', {})
                data_relationships = conversation_context.get('data_relationships', {})
                
                context_section += "### FOLLOW-UP QUERY INSTRUCTIONS:\n"
                context_section += "- This is a follow-up question to previous analysis\n"
                context_section += "- You MUST maintain the same data context and relationships\n"
                context_section += "- Do NOT switch to different datasets or analysis topics\n"
                context_section += "- Focus on the SAME data that was analyzed in previous queries\n\n"
                
                # Add specific context about previous analysis
                if previous_analysis:
                    prev_query = previous_analysis.get('user_query', 'Unknown')
                    context_section += f"- Previous query was: '{prev_query}'\n"
                    context_section += f"- You should continue analyzing the SAME data relationship\n"
                    context_section += f"- If the user asks for a different chart type, use the SAME data\n\n"
                
                # Add data relationship context
                if data_relationships.get('rent_analysis'):
                    context_section += "- Data focus: Monthly rent analysis\n"
                    context_section += "- Key columns: monthly_rent, flat_type, location\n"
                    context_section += "- Maintain focus on rent vs flat type relationships\n\n"
                
                # Add conversation history summary
                if conversation_context.get('conversation_memory', {}).get('analysis_history'):
                    context_section += "### CONVERSATION HISTORY:\n"
                    history = conversation_context['conversation_memory']['analysis_history']
                    for i, analysis in enumerate(history[-3:], 1):  # Show last 3 conversations
                        query = analysis.get('user_query', 'Unknown query')
                        context_section += f"{i}. Query: {query}\n"
                    context_section += "\n"
                    
                    context_section += "### CRITICAL INSTRUCTIONS FOR FOLLOW-UP:\n"
                    context_section += "1. Use the EXACT SAME data that was loaded in previous queries\n"
                    context_section += "2. Focus on the SAME data relationships (e.g., monthly_rent vs flat_type)\n"
                    context_section += "3. If user asks for different chart types, use the SAME data\n"
                    context_section += "4. Do NOT try to load new data or switch datasets\n"
                    context_section += "5. Maintain consistency with previous analysis context\n\n"
            
            context_section += "\n"

        if should_plot:
            return f"""
            CRITICAL INSTRUCTION: You MUST assign the final plot to a variable named `result`. 
            DO NOT use fig.show(), fig.print(), or return the figure.
            
            CORRECT: result = fig
            WRONG: fig.show()
            WRONG: return fig
            WRONG: print(fig)

            You are a senior data analyst. The user has uploaded pandas DataFrames, available in a dictionary called `dfs` (keyed by file_id):
            {all_profiles}
            {context_section}
            ### Query: "{query}"
            ### INSTRUCTIONS (READ CAREFULLY):
            1. **FIRST: Access the DataFrames from the dfs dictionary**
            2. **Example: df = dfs["file_id"] for each DataFrame you need**
            3. **Available file IDs: {list(dfs_metadata.keys())}**
            4. Only use column names exactly as listed above. Do not invent or guess column names
            5. Use pandas for data manipulation and plotly.graph_objects (go) for plotting.
            6. **CRITICAL: YOU MUST assign the final plot to a variable named `result` (e.g., result = fig).**
            7. If you do NOT assign the plot to `result`, the plot will NOT be shown to the user.
            8. **NEVER use .show(), .print(), or return the figure.**
            9. Only use column names exactly as listed above. Do not invent or guess column names.
            10. Use pandas for data manipulation and plotly.graph_objects (go) for plotting.
            11. Create a meaningful plot with proper labels and a title.
            12. **CONTEXT AWARENESS: If this is a follow-up query, use the same data relationships from previous analysis.**
            13. Output your final code inside a single code block starting with ```python and ending with ```.
            14. Return only valid Python code.

            ### FORBIDDEN CODE PATTERNS (DO NOT USE):
            - fig.show()
            - fig.print()
            - return fig
            - print(fig)
            - plt.show()
            - display(fig)

            ### REQUIRED CODE PATTERN:
            - result = fig

            ### FINAL REMINDER: 
            - ALWAYS use: result = fig
            - NEVER use: fig.show(), fig.print(), return fig, or print(fig)
            
            If you do not assign the plot to `result`, the user will see nothing.
            """
        elif should_generate_table:
            return f"""
            You are a senior data analyst. The user has uploaded CSV files with the following information:
            {all_profiles}
            {context_section}
            ### Query: "{query}"
            ### Instructions for TABLE GENERATION:
            1. **FIRST: Access the DataFrames from the dfs dictionary**
            2. **Example: df = dfs["file_id"] for each DataFrame you need**
            3. **Available file IDs: {list(dfs_metadata.keys())}**
            4. Perform the requested analysis using pandas operations.
            5. Only use column names exactly as listed above. Do not invent or guess column names.
            6. **CRITICAL: Assign your final table result to a variable named `result`.**
            7. The result should be a pandas DataFrame that can be displayed as a table:
               - Filtered data: result = df[df['column'] > value]
               - Aggregated data: result = df.groupby('column').agg({{'column2': 'mean'}})
               - Statistical summary: result = df.describe()
               - Pivot table: result = df.pivot_table(values='value', index='index_col', columns='col_col')
            8. **DO NOT create plots or visualizations.**
            9. **DO create structured, tabular data that answers the query.**
            10. **CONTEXT AWARENESS: If this is a follow-up query, use the same data relationships from previous analysis.**
            11. Ensure the result DataFrame has meaningful column names and is properly formatted.
            12. Wrap the snippet in a single ```python code fence.
            13. Return only valid Python code.

            ### Examples of good table results:
            - result = df[df['age'] > 25]  # Filtered table
            - result = df.groupby('category').agg({{'sales': 'sum', 'count': 'count'}})  # Aggregated table
            - result = df.pivot_table(values='amount', index='date', columns='category')  # Pivot table
            """
        else:
            return f"""
            You are a senior data analyst. The user has uploaded CSV files with the following information:
            {all_profiles}
            {context_section}
            ### Query: "{query}"
            ### Instructions:
            1. **FIRST: Access the DataFrames from the dfs dictionary**
            2. **Example: df = dfs["file_id"] for each DataFrame you need**
            3. **Available file IDs: {list(dfs_metadata.keys())}**
            4. Perform the requested analysis using pandas operations.
            5. Only use column names exactly as listed above. Do not invent or guess column names.
            6. **CRITICAL: Assign your final analysis result to a variable named `result`.**
            7. The result can be:
               - A string with your analysis and reasoning
               - A dictionary with statistics and insights
               - A pandas DataFrame with calculated results
               - Any other data structure that answers the query
            8. **DO NOT create plots or visualizations.**
            9. **CONTEXT AWARENESS: If this is a follow-up query, use the same data relationships from previous analysis.**
            10. Provide clear, insightful analysis in text format.
            11. Wrap the snippet in a single ```python code fence.
            12. Return only valid Python code.

            ### Examples of good results:
            - result = "The dataset contains 1000 rows with 5 columns. The average value is 25.5."
            - result = {{"summary": "Dataset analysis", "count": 1000, "insights": ["insight1", "insight2"]}}
            - result = df.describe()  # Statistical summary
            """
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess generated code to fix common issues (as per backend)"""
        # Remove fig.show() calls and replace with result = fig
        code = re.sub(r'fig\.show\(\)', 'result = fig', code)
        code = re.sub(r'fig\.print\(\)', 'result = fig', code)
        
        # Remove return fig statements and replace with result = fig
        code = re.sub(r'return\s+fig', 'result = fig', code)
        code = re.sub(r'return\s+\(fig\)', 'result = fig', code)
        
        # Remove print(fig) statements
        code = re.sub(r'print\s*\(\s*fig\s*\)', '', code)
        
        # Remove other forbidden patterns
        code = re.sub(r'plt\.show\(\)', 'result = fig', code)
        code = re.sub(r'display\s*\(\s*fig\s*\)', 'result = fig', code)
        
        # If there's a fig variable but no result assignment, add it
        if 'fig' in code and 'result = fig' not in code and 'result=' not in code:
            # Find the last line that creates a fig and add result = fig after it
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if 'fig =' in line or 'fig=' in line:
                    # Add result = fig on the next line
                    lines.insert(i + 1, 'result = fig')
                    break
            
            # If we didn't find a fig creation line, add result = fig at the end
            if 'result = fig' not in '\n'.join(lines):
                lines.append('result = fig')
            
            code = '\n'.join(lines)
        
        # Ensure there's always a result assignment for non-plotting code
        if 'result =' not in code and 'result=' not in code:
            # Check if this looks like analysis code (not plotting)
            if 'fig' not in code and 'plot' not in code and 'go.Figure' not in code:
                # Add a basic result assignment at the end
                lines = code.split('\n')
                lines.append('result = "Analysis completed successfully"')
                code = '\n'.join(lines)
        
        return code

    def _extract_context_clues(self, user_query: str, conversation_memory: dict, previous_analysis: dict) -> str:
        """Extract context clues from the current query and conversation history"""
        clues = []
        
        # Check if this is a follow-up visualization request
        if any(word in user_query.lower() for word in ['pie', 'line', 'bar', 'chart', 'plot', 'graph']):
            clues.append("visualization_request")
        
        # Check if this is about the same data
        if previous_analysis:
            prev_query = previous_analysis.get('user_query', '').lower()
            if any(word in prev_query for word in ['rent', 'flat', 'monthly']):
                clues.append("rent_analysis_followup")
        
        # Check for specific data relationships
        if 'rent' in user_query.lower() and 'flat' in user_query.lower():
            clues.append("rent_vs_flat_type_analysis")
        
        return ", ".join(clues) if clues else "general_analysis"

    def _extract_data_relationships(self, conversation_memory: dict, previous_analysis: dict) -> dict:
        """Extract data relationships from conversation history"""
        relationships = {}
        
        if conversation_memory:
            analysis_history = conversation_memory.get("analysis_history", [])
            if analysis_history:
                # Get the most recent analysis
                latest = analysis_history[-1]
                relationships["rent_analysis"] = {
                    "primary_analysis": latest.get("data_relationship", "unknown"),
                    "potential_relationships": ["monthly_rent vs flat_type", "monthly_rent vs location", "flat_type vs location"]
                }
        
        if previous_analysis:
            relationships["previous_focus"] = previous_analysis.get("data_relationship", "unknown")
        
        return relationships

    def _extract_file_metadata(self, dataframes: dict) -> dict:
        """Extract file metadata for context"""
        metadata = {}
        
        for file_id, df_info in dataframes.items():
            metadata[file_id] = {
                "columns": df_info.get("columns", []),
                "rows": df_info.get("shape", (0, 0))[0] if df_info.get("shape") else 0,
                "file_type": "csv"
            }
        
        return metadata
    
    def _extract_file_metadata_from_ids(self, file_ids: list, session_id: str) -> dict:
        """Extract rich file metadata from file IDs using database metadata"""
        metadata = {}
        
        try:
            from database import db_manager
            
            for file_id in file_ids:
                # Get file info from database
                file_info = db_manager.get_file_by_id(file_id, session_id)
                if file_info:
                    # Get the rich metadata that was stored during cleaning
                    db_metadata = file_info.get('metadata', {})
                    
                    # Extract rich metadata for code generation
                    metadata[file_id] = {
                        "file_id": file_id,
                        "filename": file_info.get('original_filename', 'Unknown'),
                        "cleaned_filename": file_info.get('cleaned_filename', ''),
                        "uploaded_filename": file_info.get('uploaded_filename', ''),
                        "file_type": "csv",
                        "is_cleaned": file_info.get('is_cleaned', False),
                        "shape": (db_metadata.get('row_count', 0), db_metadata.get('column_count', 0)),
                        "columns": db_metadata.get('columns', []),
                        "dtypes": db_metadata.get('dtypes', {}),
                        "null_counts": db_metadata.get('null_counts', {}),
                        "head_sample": db_metadata.get('sample_data', []),
                        "file_path": file_info.get('cleaned_filename') or file_info.get('uploaded_filename', ''),
                        "row_count": db_metadata.get('row_count', 0),
                        "column_count": db_metadata.get('column_count', 0),
                        "numeric_stats": db_metadata.get('numeric_stats', {}),
                        "has_numeric_data": db_metadata.get('has_numeric_data', False),
                        "has_categorical_data": db_metadata.get('has_categorical_data', False)
                    }
                    
                    print(f"DEBUG: SmartCodeGenerator - Extracted rich metadata for {file_id}: {metadata[file_id]}")
                else:
                    metadata[file_id] = {
                        "file_id": file_id,
                        "error": "File not found in database"
                    }
        except Exception as e:
            print(f"Error extracting file metadata: {e}")
            metadata = {"error": f"Failed to extract metadata: {str(e)}"}
        
        return metadata

    def _extract_first_code_block(self, text: str) -> str:
        """Extracts the first Python code block from a markdown-formatted string (as per backend)"""
        start = text.find("```python")
        if start == -1:
            return ""
        start += len("```python")
        end = text.find("```", start)
        if end == -1:
            return ""
        return text[start:end].strip()

# Factory function for creating the node
def smart_code_generator_node():
    return SmartCodeGeneratorNode() 