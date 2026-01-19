"""
Base node class for LangGraph workflow nodes

Provides common functionality and interfaces for all workflow nodes.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from state import ConversationState

class BaseNode(ABC):
    """Base class for all workflow nodes"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, state: ConversationState) -> ConversationState:
        """
        Execute the node's logic
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        pass
    
    def validate_state(self, state: ConversationState) -> bool:
        """
        Validate that the state has required fields for this node
        
        Args:
            state: Conversation state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        required_fields = self.get_required_fields()
        for field in required_fields:
            if field not in state:
                print(f"❌ Missing required field: {field}")
                return False
            if state[field] is None:
                print(f"❌ Required field is None: {field}")
                return False
            if isinstance(state[field], (str, bytes)) and len(state[field]) == 0:
                print(f"❌ Required field is empty: {field}")
                return False
        return True
    
    def get_required_fields(self) -> list:
        """
        Get list of required fields for this node
        
        Returns:
            List of required field names
        """
        return []
    
    def log_execution(self, state: ConversationState, result: Any = None):
        """
        Log node execution for debugging
        
        Args:
            state: Current state
            result: Optional result from execution
        """
        print(f"\n{'='*60}")
        print(f"🔧 NODE: {self.name.upper()}")
        print(f"📝 Description: {self.description}")
        print(f"📊 Input State Keys: {list(state.keys())}")
        
        # Show key state information
        if 'session_id' in state:
            print(f"🆔 Session ID: {state['session_id']}")
        if 'user_query' in state:
            print(f"❓ User Query: {state['user_query']}")
        if 'file_ids' in state:
            print(f"📁 File IDs: {state['file_ids']}")
        if 'query_classification' in state:
            print(f"🏷️  Query Classification: {state['query_classification']}")
        
        print(f"{'='*60}")
        
        if result:
            print(f"✅ RESULT: {result}")
            print(f"{'='*60}\n")
    
    def handle_error(self, state: ConversationState, error: Exception) -> ConversationState:
        """
        Handle errors during node execution with intelligent recovery
        
        Args:
            state: Current state
            error: Exception that occurred
            
        Returns:
            Updated state with error information and recovery attempts
        """
        error_msg = f"Error in {self.name}: {str(error)}"
        state["error"] = error_msg
        state["current_step"] = "error"
        
        # Try to recover from the error intelligently
        recovered_state = self._attempt_error_recovery(state, error)
        if recovered_state:
            state = recovered_state
            state["error"] = None  # Clear error if recovery successful
            state["current_step"] = "recovered"
            print(f"✅ Error recovered in {self.name}")
            return state
        
        # If recovery failed, provide user-friendly error message
        user_friendly_error = self._generate_user_friendly_error(error)
        state["error"] = user_friendly_error
        state["result"] = {
            "error_type": "unrecoverable",
            "user_message": user_friendly_error,
            "suggested_actions": self._suggest_recovery_actions(error),
            "query_type": "error"
        }
        
        print(f"❌ Error in {self.name}: {str(error)}")
        return state
    
    def _attempt_error_recovery(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """
        Attempt to automatically recover from the error
        
        Args:
            state: Current state with error
            error: The exception that occurred
            
        Returns:
            Recovered state if successful, None if recovery failed
        """
        try:
            # Common error recovery strategies
            if "database" in str(error).lower() or "connection" in str(error).lower():
                return self._recover_database_error(state, error)
            elif "file" in str(error).lower() or "not found" in str(error).lower():
                return self._recover_file_error(state, error)
            elif "permission" in str(error).lower() or "access" in str(error).lower():
                return self._recover_permission_error(state, error)
            elif "timeout" in str(error).lower():
                return self._recover_timeout_error(state, error)
            else:
                return self._recover_generic_error(state, error)
                
        except Exception as recovery_error:
            print(f"Recovery attempt failed: {str(recovery_error)}")
            return None
    
    def _recover_database_error(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """Attempt to recover from database-related errors"""
        try:
            # Try to reconnect or use cached data
            if hasattr(self, 'db_manager'):
                # Try to refresh connection
                if hasattr(self.db_manager, 'refresh_connection'):
                    self.db_manager.refresh_connection()
                    return state
            return None
        except:
            return None
    
    def _recover_file_error(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """Attempt to recover from file-related errors"""
        try:
            # Check if file exists in different location or try alternative files
            if 'file_ids' in state and state['file_ids']:
                # Try to find alternative files
                for file_id in state['file_ids']:
                    if hasattr(self, 'db_manager'):
                        file_info = self.db_manager.get_file_info(file_id)
                        if file_info and file_info.get('file_path'):
                            # File exists, try to use it
                            return state
            return None
        except:
            return None
    
    def _recover_permission_error(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """Attempt to recover from permission-related errors"""
        # Permission errors usually can't be auto-recovered
        return None
    
    def _recover_timeout_error(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """Attempt to recover from timeout errors"""
        try:
            # Could implement retry logic here
            return None
        except:
            return None
    
    def _recover_generic_error(self, state: ConversationState, error: Exception) -> Optional[ConversationState]:
        """Attempt generic error recovery"""
        try:
            # Try to continue with partial data or cached results
            if 'result' in state and state['result']:
                # If we have some results, try to continue
                return state
            return None
        except:
            return None
    
    def _generate_user_friendly_error(self, error: Exception) -> str:
        """Generate user-friendly error messages"""
        error_str = str(error).lower()
        
        if "database" in error_str or "connection" in error_str:
            return "We're having trouble connecting to our database. Please try again in a moment."
        elif "file" in error_str or "not found" in error_str:
            return "The requested file couldn't be found. Please check if the file was uploaded correctly."
        elif "permission" in error_str or "access" in error_str:
            return "We don't have permission to access this resource. Please contact support if this persists."
        elif "timeout" in error_str:
            return "The operation is taking longer than expected. Please try again."
        elif "memory" in error_str or "out of memory" in error_str:
            return "The file is too large to process. Please try with a smaller file or contact support."
        else:
            return "Something unexpected happened. Please try again or contact support if the problem persists."
    
    def _suggest_recovery_actions(self, error: Exception) -> list:
        """Suggest actions the user can take to recover from the error"""
        error_str = str(error).lower()
        
        if "database" in error_str or "connection" in error_str:
            return [
                "Wait a few moments and try again",
                "Check your internet connection",
                "Contact support if the problem persists"
            ]
        elif "file" in error_str or "not found" in error_str:
            return [
                "Re-upload the file",
                "Check if the file format is supported",
                "Ensure the file isn't corrupted"
            ]
        elif "permission" in error_str or "access" in error_str:
            return [
                "Log out and log back in",
                "Check if your session has expired",
                "Contact support for access issues"
            ]
        elif "timeout" in error_str:
            return [
                "Try again with a smaller file",
                "Check your internet connection",
                "Try during off-peak hours"
            ]
        else:
            return [
                "Try the operation again",
                "Check if all required fields are filled",
                "Contact support if the problem continues"
            ] 