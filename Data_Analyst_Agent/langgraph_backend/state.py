from typing import Dict, List, Optional, Any, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel
from datetime import datetime
from database import db_manager

class ConversationState(TypedDict):
    """Minimal state for LangGraph workflow - optimized for checkpointing"""
    session_id: str
    user_query: str
    current_step: str
    messages: List[BaseMessage]
    file_ids: List[str]
    # Only essential fields that change during workflow
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    # Query classification for intelligent processing
    query_classification: Optional[Dict[str, Any]]
    # Conversation memory (consolidated from both fields)
    conversation_memory: Optional[Dict[str, Any]]
    # Previous analysis context
    previous_analysis: Optional[Dict[str, Any]]

class SessionData(BaseModel):
    """Session data model"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    is_active: bool

class SimpleStateManager:
    """Simplified state management"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str):
        """Create a new session"""
        try:
            # Create in database first
            db_manager.create_user_session(session_id)
            print(f"DEBUG: Created session {session_id} in database")
        except Exception as e:
            print(f"DEBUG: Warning - could not create session in database: {e}")
            # Continue anyway as the memory session is sufficient for basic functionality
        
        # Create in memory
        self.active_sessions[session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        print(f"DEBUG: Created session {session_id} in memory")
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info"""
        if session_id not in self.active_sessions:
            # Check database
            session_data = db_manager.get_user_session(session_id)
            if session_data:
                self.active_sessions[session_id] = {
                    "created_at": datetime.fromisoformat(session_data['created_at']),
                    "last_activity": datetime.fromisoformat(session_data['last_activity'])
                }
                
                # Also load conversation history
                try:
                    history = db_manager.get_conversation_history(session_id)
                    if history:
                        self.active_sessions[session_id]['conversation_history'] = history
                except Exception as e:
                    print(f"Warning: Could not load conversation history: {e}")
        
        if session_id in self.active_sessions:
            # Update activity
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            db_manager.update_user_activity(session_id)
            return self.active_sessions[session_id]
        
        return None
    
    def update_session_data(self, session_id: str, **updates) -> bool:
        """Update session data with new information"""
        if session_id not in self.active_sessions:
            # Try to load the session first
            self.get_session(session_id)
            if session_id not in self.active_sessions:
                return False
        
        session = self.active_sessions[session_id]
        session.update(updates)
        
        # Update activity
        session["last_activity"] = datetime.now()
        db_manager.update_user_activity(session_id)
        
        print(f"DEBUG: Updated session {session_id} with keys: {list(updates.keys())}")
        return True
    
    def get_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get files for a session"""
        return db_manager.get_user_files(session_id)
    
    def add_file(self, session_id: str, file_data: Dict[str, Any]) -> str:
        """Add file to session"""
        file_id = db_manager.add_file_record(
            session_id=session_id,
            original_filename=file_data['original_filename'],
            uploaded_filename=file_data['uploaded_filename'],
            file_type=file_data['file_type'],
            file_size=file_data['file_size'],
            metadata=file_data.get('metadata')
        )
        return file_id
    
    def update_file(self, file_id: str, cleaned_data: Dict[str, Any]):
        """Update file with cleaning results"""
        db_manager.update_file_cleaned_status(
            file_id=file_id,
            cleaned_filename=cleaned_data.get('cleaned_filename', ''),
            cleaning_log=cleaned_data.get('cleaning_log'),
            impact_metrics=cleaned_data.get('impact_metrics')
        )
    
    def get_workflow_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow state for a session (for paused workflows)"""
        # For now, return None as we don't have paused workflow functionality yet
        return None
    
    def get_or_create_conversation(self, session_id: str) -> Dict[str, Any]:
        """Get or create conversation for a session"""
        # For now, return a simple conversation object
        return {
            "session_id": session_id,
            "topics": [],
            "messages": []
        }
    
    def update_conversation_memory(self, session_id: str, analysis_result: Dict[str, Any]):
        """Update conversation memory for a session (consolidated with history)"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        if 'conversation_memory' not in session:
            session['conversation_memory'] = {}
        
        # Update memory with analysis results
        session['conversation_memory'].update(analysis_result)
        
        # Also store conversation history in memory for easy access
        if 'conversation_history' not in session['conversation_memory']:
            session['conversation_memory']['conversation_history'] = []
    
    def update_conversation_history(self, session_id: str, human_query: str, ai_response: str):
        """Update conversation history for a session (consolidated with memory)"""
        if session_id not in self.active_sessions:
            # Try to load from database first
            self.get_session(session_id)
            if session_id not in self.active_sessions:
                return
        
        session = self.active_sessions[session_id]
        
        # Ensure conversation memory exists
        if 'conversation_memory' not in session:
            session['conversation_memory'] = {}
        
        # Initialize conversation history in memory
        if 'conversation_history' not in session['conversation_memory']:
            session['conversation_memory']['conversation_history'] = []
        
        # Add new message pair
        session['conversation_memory']['conversation_history'].append({
            'human': human_query,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 3 messages
        if len(session['conversation_memory']['conversation_history']) > 3:
            session['conversation_memory']['conversation_history'] = session['conversation_memory']['conversation_history'][-3:]
        
        # Persist to database
        try:
            db_manager.update_conversation_history(session_id, session['conversation_memory']['conversation_history'])
        except Exception as e:
            print(f"Warning: Could not persist conversation history: {e}")
    
    def get_conversation_memory(self, session_id: str) -> Dict[str, Any]:
        """Get conversation memory for a session"""
        # First check active_sessions
        if session_id in self.active_sessions:
            memory = self.active_sessions[session_id].get('conversation_memory', {})
            if memory:
                return memory
        
        # If not in active_sessions, try to get from session data
        try:
            session_data = self.get_session(session_id)
            if session_data and isinstance(session_data, dict):
                memory = session_data.get('conversation_memory', {})
                if memory:
                    # Load into active_sessions for future use
                    if session_id not in self.active_sessions:
                        self.active_sessions[session_id] = {}
                    self.active_sessions[session_id]['conversation_memory'] = memory
                    return memory
        except Exception as e:
            print(f"Warning: Could not get conversation memory from session data: {e}")
        
        return {}
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session (from consolidated memory)"""
        # First check active_sessions conversation memory
        if session_id in self.active_sessions:
            memory = self.active_sessions[session_id].get('conversation_memory', {})
            if 'conversation_history' in memory:
                return memory['conversation_history']
        
        # If not in active_sessions, try to load from database
        try:
            history = db_manager.get_conversation_history(session_id)
            if history:
                # Load into active_sessions conversation memory for future use
                if session_id not in self.active_sessions:
                    self.get_session(session_id)
                if session_id in self.active_sessions:
                    if 'conversation_memory' not in self.active_sessions[session_id]:
                        self.active_sessions[session_id]['conversation_memory'] = {}
                    self.active_sessions[session_id]['conversation_memory']['conversation_history'] = history
            return history
        except Exception as e:
            print(f"Warning: Could not get conversation history from database: {e}")
            return []
    
    def get_last_conversation_context(self, session_id: str, max_messages: int = 3) -> List[Dict[str, Any]]:
        """Get last N messages for context (from consolidated memory)"""
        history = self.get_conversation_history(session_id)
        return history[-max_messages:] if history else []

# Global state manager
state_manager = SimpleStateManager()

def create_initial_state(session_id: str, user_query: str, file_ids: List[str] = None) -> ConversationState:
    """Create minimal initial state"""
    # Ensure session exists
    if not state_manager.get_session(session_id):
        state_manager.create_session(session_id)
    
    # Get file IDs from session if not provided
    if file_ids is None:
        try:
            session_files = state_manager.get_session_files(session_id)
            file_ids = [f['file_id'] for f in session_files if f.get('file_id')]
            print(f"DEBUG: create_initial_state - Found {len(file_ids)} files in session: {file_ids}")
        except Exception as e:
            print(f"DEBUG: create_initial_state - Error getting session files: {e}")
            file_ids = []
    
    # Get existing conversation memory if available
    existing_memory = {}
    try:
        existing_memory = state_manager.get_conversation_memory(session_id)
        print(f"DEBUG: create_initial_state - Loaded existing conversation memory: {existing_memory}")
    except Exception as e:
        print(f"DEBUG: create_initial_state - Error getting conversation memory: {e}")
        existing_memory = {}
    
    # Conversation history is now consolidated in conversation_memory
    print(f"DEBUG: create_initial_state - Final conversation_memory: {existing_memory}")

    return ConversationState(
        session_id=session_id,
        user_query=user_query,
        current_step="started",
        messages=[HumanMessage(content=user_query)],
        file_ids=file_ids,
        result={},
        error=None,
        query_classification=None,
        conversation_memory=existing_memory,  # Use existing memory if available, empty dict if not
        previous_analysis=None
    )

def update_state(state: ConversationState, **updates) -> ConversationState:
    """Update state with new values"""
    state.update(updates)
    return state