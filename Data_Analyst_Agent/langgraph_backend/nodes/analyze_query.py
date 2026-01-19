"""
Analyze Query Node

This node analyzes user queries to determine the appropriate workflow path.
It checks conversation state and routes accordingly.
"""

from typing import Dict, Any, List
from .base_node import BaseNode
from state import ConversationState, state_manager

class AnalyzeQueryNode(BaseNode):
    """Node for analyzing user queries and determining workflow path"""
    
    def __init__(self):
        super().__init__(
            name="analyze_query",
            description="Analyzes user queries and determines workflow routing"
        )
    
    def get_required_fields(self) -> list:
        """Get required fields for this node"""
        return ["user_query", "session_id"]
    
    def execute(self, state: ConversationState) -> ConversationState:
        """Analyze the query and determine the appropriate workflow path"""
        try:
            # Validate state
            if not self.validate_state(state):
                raise ValueError("Invalid state for analyze_query node")
            
            self.log_execution(state)
            
            user_query = state["user_query"]
            session_id = state["session_id"]
            
            # Get conversation memory for this session
            conversation = state_manager.get_or_create_conversation(session_id)
            
            # Analyze if this query is about previous topics
            topic_analysis = conversation.is_query_about_previous_topic(user_query)
            
            # Check if we have existing files for this session
            existing_files = state_manager.get_session_files(session_id)
            has_existing_files = len(existing_files) > 0
            
            # Determine the query type and required actions
            query_analysis = self._analyze_query_type(user_query, existing_files)
            
            # Update state with analysis results
            state["query_classification"] = query_analysis
            state["is_previous_topic"] = topic_analysis.get("is_previous_topic", False)
            state["conversation_context"] = conversation.get_full_context()
            state["conversation_summary"] = conversation.conversation_summary
            
            # Set routing flags
            if topic_analysis.get("is_previous_topic", False) and conversation.conversation_summary:
                # User is asking about previous topics - answer from memory
                state["current_step"] = "query_analyzed_memory"
                state["reasoning"] = f"Query is about previous topic: {topic_analysis.get('related_topic', 'unknown')}. Will answer from conversation memory."
            elif query_analysis["requires_file_upload"]:
                # User wants to upload a new file
                state["current_step"] = "query_analyzed_upload"
                state["reasoning"] = "Query requires new file upload. Proceeding to file upload workflow."
            elif has_existing_files and query_analysis["can_use_existing_files"]:
                # User can use existing files
                state["current_step"] = "query_analyzed_existing"
                state["file_ids"] = [f['file_id'] for f in existing_files]
                state["reasoning"] = f"Query can use existing files. Found {len(existing_files)} files."
            else:
                # Default case - need to upload file
                state["current_step"] = "query_analyzed_default"
                state["reasoning"] = "Query analysis complete. Proceeding with default workflow."
            
            self.log_execution(state, f"Query analyzed: {state['reasoning']}")
            return state
            
        except Exception as e:
            return self.handle_error(state, e)
    
    def _analyze_query_type(self, query: str, existing_files: List[Dict]) -> Dict[str, Any]:
        """Analyze the type of query and determine required actions"""
        query_lower = query.lower()
        
        # Check for file upload requests
        upload_keywords = ["upload", "new file", "add file", "import", "load data"]
        requires_file_upload = any(keyword in query_lower for keyword in upload_keywords)
        
        # Check if query can use existing files
        analysis_keywords = ["analyze", "show", "plot", "chart", "calculate", "find", "what is", "median", "mean", "average"]
        can_use_existing_files = any(keyword in query_lower for keyword in analysis_keywords)
        
        # Check for file management requests
        file_management_keywords = ["list files", "show files", "what files", "available files"]
        is_file_management = any(keyword in query_lower for keyword in file_management_keywords)
        
        # Check for conversation memory requests
        memory_keywords = ["previous", "before", "discussed", "asked", "what was", "what did", "earlier"]
        is_memory_request = any(keyword in query_lower for keyword in memory_keywords)
        
        return {
            "requires_file_upload": requires_file_upload,
            "can_use_existing_files": can_use_existing_files,
            "is_file_management": is_file_management,
            "is_memory_request": is_memory_request,
            "query_intent": self._determine_intent(query_lower),
            "has_existing_files": len(existing_files) > 0
        }
    
    def _determine_intent(self, query: str) -> str:
        """Determine the primary intent of the query"""
        if any(word in query for word in ["upload", "import", "load"]):
            return "file_upload"
        elif any(word in query for word in ["analyze", "show", "plot", "chart"]):
            return "data_analysis"
        elif any(word in query for word in ["list", "show files", "what files"]):
            return "file_management"
        elif any(word in query for word in ["previous", "before", "discussed"]):
            return "conversation_memory"
        elif any(word in query for word in ["clean", "process", "prepare"]):
            return "data_cleaning"
        else:
            return "general_query"

# Factory function for creating the node
def analyze_query_node():
    """Create an analyze query node instance"""
    return AnalyzeQueryNode()
