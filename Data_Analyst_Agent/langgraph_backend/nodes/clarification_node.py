"""
Clarification Node

This node handles ambiguous queries by asking for clarification from the user.
"""

import os
from typing import Dict, Any, List
from .base_node import BaseNode
from state import ConversationState

class ClarificationNode(BaseNode):
    """Node for handling ambiguous queries and requesting clarification"""
    
    def __init__(self):
        super().__init__(
            name="clarification",
            description="Handles ambiguous queries by requesting clarification from the user"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["session_id", "user_query"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Handle ambiguous queries and request clarification"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for clarification node")
            
            self.log_execution(state)
            
            user_query = state["user_query"]
            
            # Get query classification from state
            query_classification = {}
            if "query_classification" in state:
                query_classification = state["query_classification"]
            
            # Determine if clarification is needed
            needs_clarification = self._needs_clarification(user_query, query_classification)
            
            if needs_clarification:
                # Generate clarification questions
                clarification_questions = self._generate_clarification_questions(user_query, query_classification)
                
                # Store clarification in result field
                if state.get("result") is None:
                    state["result"] = {}
                
                state["result"]["clarification_needed"] = True
                state["result"]["clarification_questions"] = clarification_questions
                state["result"]["reasoning"] = "Query is ambiguous and requires clarification"
                state["current_step"] = "clarification_requested"
                
                # Add AI message to conversation
                from langchain_core.messages import AIMessage
                ai_message = f"I need some clarification to better understand your request. {clarification_questions[0]}"
                state["messages"].append(AIMessage(content=ai_message))
                
                self.log_execution(state, "Clarification requested")
            else:
                # No clarification needed
                if state.get("result") is None:
                    state["result"] = {}
                
                state["result"]["clarification_needed"] = False
                state["result"]["reasoning"] = "Query is clear and can proceed with analysis"
                state["current_step"] = "clarification_not_needed"
                
                self.log_execution(state, "No clarification needed")
            
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _needs_clarification(self, query: str, classification: Dict[str, Any]) -> bool:
        """Determine if the query needs clarification"""
        query_lower = query.lower()
        
        # Check for vague terms that typically need clarification
        vague_terms = [
            "trend", "pattern", "relationship", "correlation", "distribution",
            "analysis", "insights", "overview", "summary", "what", "how"
        ]
        
        # Check if query contains vague terms without specific context
        has_vague_terms = any(term in query_lower for term in vague_terms)
        
        # Check if query is too short or generic
        is_too_generic = len(query.split()) <= 3 and has_vague_terms
        
        # Check classification for general analysis
        is_general_analysis = classification.get("query_type") == "general_analysis"
        
        # Check if query lacks specific column references
        lacks_specifics = not any(word in query_lower for word in ["column", "field", "data", "file"])
        
        return is_too_generic or is_general_analysis or lacks_specifics
    
    def _generate_clarification_questions(self, query: str, classification: Dict[str, Any]) -> List[str]:
        """Generate specific clarification questions based on the query"""
        questions = []
        
        query_lower = query.lower()
        
        # Check what type of clarification is needed
        if "trend" in query_lower:
            questions.append("What specific trend are you looking for? (e.g., over time, by category, by value range)")
            questions.append("Which columns should I use to analyze the trend?")
        
        elif "pattern" in query_lower or "relationship" in query_lower:
            questions.append("What type of pattern or relationship are you interested in?")
            questions.append("Which columns should I compare or analyze together?")
        
        elif "correlation" in query_lower:
            questions.append("Which specific columns would you like me to check for correlations?")
            questions.append("Are you looking for positive, negative, or any type of correlation?")
        
        elif "distribution" in query_lower:
            questions.append("Which column's distribution would you like to see?")
            questions.append("Would you prefer a histogram, box plot, or other visualization?")
        
        elif "analysis" in query_lower or "insights" in query_lower:
            questions.append("What specific aspect of the data would you like me to analyze?")
            questions.append("Are you looking for statistical summaries, visualizations, or both?")
        
        elif "overview" in query_lower or "summary" in query_lower:
            questions.append("What level of detail would you like in the overview?")
            questions.append("Are you interested in specific columns or the entire dataset?")
        
        else:
            # Generic clarification
            questions.append("Could you please be more specific about what you'd like to analyze?")
            questions.append("Which columns or aspects of the data are most important to you?")
            questions.append("What type of result would be most helpful? (e.g., visualization, statistics, text summary)")
        
        # Add data-specific questions if we have file information
        questions.append("Would you like me to show you what columns are available in your dataset first?")
        
        return questions

# Factory function for creating the node
def clarification_node():
    """Create a clarification node instance"""
    return ClarificationNode() 