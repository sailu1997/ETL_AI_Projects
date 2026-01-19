"""
Query Classifier Node

This node uses LLM to intelligently classify user queries and determine the appropriate response format.
"""

import os
from typing import Dict, Any
from .base_node import BaseNode
from state import ConversationState

class QueryClassifierNode(BaseNode):
    """Node for intelligently classifying user queries using LLM"""
    
    def __init__(self):
        super().__init__(
            name="query_classifier",
            description="Uses LLM to classify user queries and determine response format"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "user_query"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Classify the user query using LLM with conversation memory"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for query_classifier node")
            
            self.log_execution(state)
            
            user_query = state["user_query"]
            
            # Get conversation context from messages if available
            conversation_context = "No previous conversation."
            if state.get("messages") and len(state["messages"]) > 1:
                # Get last few messages for context
                recent_messages = state["messages"][-3:]  # Last 3 messages
                context_parts = []
                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                        context_parts.append(f"{type(msg).__name__}: {content}")
                conversation_context = "\n".join(context_parts)
            
            # Get conversation memory from state if available
            conversation_memory = state.get("conversation_memory", {})
            previous_analysis = state.get("previous_analysis", {})
            
            # Build enhanced context with memory
            enhanced_context = conversation_context
            if conversation_memory or previous_analysis:
                memory_info = []
                
                if conversation_memory.get("last_analysis"):
                    last_analysis = conversation_memory["last_analysis"]
                    memory_info.append(f"Previous Analysis: {last_analysis.get('user_query', 'Unknown')}")
                    memory_info.append(f"Query Type: {last_analysis.get('query_type', 'Unknown')}")
                    if last_analysis.get('data_relationship'):
                        memory_info.append(f"Data Relationship: {last_analysis['data_relationship']}")
                    if last_analysis.get('columns_analyzed'):
                        memory_info.append(f"Columns Analyzed: {', '.join(last_analysis['columns_analyzed'])}")
                
                if previous_analysis:
                    memory_info.append(f"Previous Result Type: {previous_analysis.get('result_type', 'Unknown')}")
                    if previous_analysis.get('data_relationship'):
                        memory_info.append(f"Data Relationship: {previous_analysis['data_relationship']}")
                
                if memory_info:
                    enhanced_context += "\n\n" + "\n".join(memory_info)
            
            # Classify the query using LLM with enhanced context
            classification = self._classify_query_with_llm(user_query, enhanced_context)
            
            # Store classification in state
            state["query_classification"] = classification
            state["current_step"] = "query_classified"
            
            print(f"DEBUG: QueryClassifier - Stored classification: {classification}")
            print(f"DEBUG: QueryClassifier - Response format: {classification.get('response_format')}")
            print(f"DEBUG: QueryClassifier - State keys after classification: {list(state.keys())}")
            
            # Update conversation memory with follow-up detection
            if classification.get("is_follow_up", False):
                if "conversation_memory" not in state:
                    state["conversation_memory"] = {}
                state["conversation_memory"]["is_follow_up"] = True
                state["conversation_memory"]["context_clues"] = classification.get("context_clues", "")
                
                # Mark this as a follow-up for the analyze_data node
                if "previous_analysis" not in state:
                    state["previous_analysis"] = {}
                state["previous_analysis"]["is_follow_up"] = True
            
            self.log_execution(state, f"Query classified: {classification['query_type']} (follow-up: {classification.get('is_follow_up', False)})")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _classify_query_with_llm(self, query: str, conversation_context: str = "") -> Dict[str, Any]:
        """Use LLM to classify the query with conversation context"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Create the classification prompt
            classification_prompt = f"""
Classify this data analysis query:

Query: "{query}"
Context: {conversation_context}
Refer to the Context to understand the user's intent. If the user is asking for a follow-up, return is_follow_up as true. Also return the context clues from the query.
Return JSON:
{{
    "query_type": "visualization|statistical_analysis|data_filtering|aggregation|general_analysis|conversation_memory",
    "response_format": "plot|table|text|code|mixed|none",
    "needs_dataframe_info": true/false,
    "is_follow_up": true/false,
    "context_clues": "any specific context clues from the query"
}}

Rules:
- Use "conversation_memory" if asking about previous discussion
- Use "general_analysis" if query is too vague
- Focus on user intent, not keywords

**Context Awareness:**
- Analyze if this query builds on previous conversation
- Detect implicit references to previous data relationships
- Understand follow-up requests for different visualization types
"""
            
            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analysis expert who classifies user queries."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            import json
            classification_text = response.choices[0].message.content.strip()
            
            # Extract JSON from the response
            start_idx = classification_text.find('{')
            end_idx = classification_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = classification_text[start_idx:end_idx]
                classification = json.loads(json_str)
            else:
                raise Exception("No valid JSON found in LLM response")
            
            return classification
            
        except Exception as e:
            raise Exception(f"Query classification failed: {str(e)}")

# Factory function for creating the node
def query_classifier_node():
    """Create a query classifier node instance"""
    return QueryClassifierNode() 