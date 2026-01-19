from typing import Dict, List, Optional, Any, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel
import json
from datetime import datetime
from database import db_manager
import os

class ConversationState(TypedDict):
    """State for the conversation flow"""
    messages: List[BaseMessage]
    session_id: str
    user_query: str
    file_ids: List[str]
    current_step: str
    agent_scratchpad: str
    intermediate_steps: List[tuple]
    final_answer: Optional[str]
    error: Optional[str]
    plot_data: Optional[Dict]
    reasoning: Optional[str]
    conversation_summary: Optional[str]
    metadata: Dict[str, Any]
    # File processing fields
    file_content: Optional[bytes]
    filename: Optional[str]
    should_clean: Optional[bool]
    uploaded_file: Optional[Dict[str, Any]]
    cleaned_file: Optional[Dict[str, Any]]
    file_insights: Optional[Dict[str, Any]]
    # Query analysis fields
    query_classification: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    dataframe_info: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    # Conversation memory fields
    conversation_context: Optional[str]
    is_previous_topic: Optional[bool]
    previous_topic_summary: Optional[str]

class SessionData(BaseModel):
    """Session data model"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    is_active: bool
    files: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]

class ConversationMemory:
    """Manages conversation memory and context with intelligent topic understanding"""
    
    def __init__(self, session_id: str, max_messages: int = 20):
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: List[BaseMessage] = []
        self.conversation_summary: Optional[str] = None
        self.topic_history: List[Dict[str, Any]] = []
        self.previous_analyses: Dict[str, Any] = {}
    
    def add_message(self, message: BaseMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
        
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            # Keep the first message (system context) and last max_messages-1
            self.messages = [self.messages[0]] + self.messages[-(self.max_messages-1):]
    
    def get_context_messages(self) -> List[BaseMessage]:
        """Get messages for context (excluding the current query)"""
        return self.messages[:-1] if self.messages else []
    
    def get_full_context(self) -> str:
        """Get a text summary of the conversation context"""
        if not self.messages:
            return "No previous conversation."
        
        context_parts = []
        for i, msg in enumerate(self.messages[-5:], 1):  # Last 5 messages
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content[:100]}...")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content[:100]}...")
        
        return "\n".join(context_parts)
    
    def update_summary(self, summary: str):
        """Update the conversation summary"""
        self.conversation_summary = summary
    
    def add_topic(self, topic: str, analysis_result: Dict[str, Any] = None):
        """Add a new topic to the conversation history"""
        self.topic_history.append({
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "analysis_result": analysis_result
        })
        
        # Keep only last 10 topics
        if len(self.topic_history) > 10:
            self.topic_history = self.topic_history[-10:]
    
    def get_topic_summary(self) -> str:
        """Get a summary of previous topics"""
        if not self.topic_history:
            return "No previous topics discussed."
        
        topics = [topic["topic"] for topic in self.topic_history[-5:]]  # Last 5 topics
        return f"Previous topics: {', '.join(topics)}"
    
    def is_query_about_previous_topic(self, query: str) -> Dict[str, Any]:
        """Intelligently determine if query is about previous topics"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Create context from previous topics and analyses
            context = self.get_topic_summary()
            if self.previous_analyses:
                context += f"\nPrevious analyses: {json.dumps(self.previous_analyses, indent=2)}"
            
            analysis_prompt = f"""
You are an intelligent conversation analyzer. Determine if the user's query is asking about:
1. Previous topics/analyses that were already discussed
2. A new analysis request
3. General conversation memory questions

Previous conversation context:
{context}

Current query: "{query}"

Respond in JSON format:
{{
    "is_previous_topic": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your decision",
    "related_topic": "specific previous topic if applicable, or null",
    "response_type": "one of: previous_topic, new_analysis, conversation_memory, unclear"
}}

Examples:
- "what was my previous question on?" → conversation_memory
- "show me the same plot but for salary" → previous_topic
- "what is the median of monthly rent?" → new_analysis
- "can you show me that heatmap again?" → previous_topic
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a conversation analysis expert."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse the response
            import json
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = analysis_text[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
                else:
                    # Fallback analysis
                    return self._fallback_topic_analysis(query)
            except json.JSONDecodeError:
                return self._fallback_topic_analysis(query)
                
        except Exception as e:
            # Fallback to keyword-based analysis
            return self._fallback_topic_analysis(query)
    
    def _fallback_topic_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback topic analysis using keywords and patterns"""
        query_lower = query.lower()
        
        # Check for conversation memory patterns
        memory_keywords = ["previous", "before", "discussed", "asked", "question", "what was", "what did", "earlier", "last time"]
        if any(keyword in query_lower for keyword in memory_keywords):
            return {
                "is_previous_topic": True,
                "confidence": 0.8,
                "reasoning": "Query contains conversation memory keywords",
                "related_topic": None,
                "response_type": "conversation_memory"
            }
        
        # Check for reference to previous analyses
        reference_keywords = ["same", "again", "that", "previous", "last", "before", "earlier"]
        if any(keyword in query_lower for keyword in reference_keywords):
            return {
                "is_previous_topic": True,
                "confidence": 0.7,
                "reasoning": "Query references previous content",
                "related_topic": "previous analysis",
                "response_type": "previous_topic"
            }
        
        # Check for new analysis patterns
        analysis_keywords = ["what is", "show me", "calculate", "find", "get", "analyze", "plot", "chart", "heatmap", "median", "mean", "average"]
        if any(keyword in query_lower for keyword in analysis_keywords):
            return {
                "is_previous_topic": False,
                "confidence": 0.6,
                "reasoning": "Query appears to request new analysis",
                "related_topic": None,
                "response_type": "new_analysis"
            }
        
        # Default to unclear
        return {
            "is_previous_topic": False,
            "confidence": 0.3,
            "reasoning": "Unable to determine if query is about previous topics",
            "related_topic": None,
            "response_type": "unclear"
        }
    
    def store_analysis_result(self, query: str, result: Dict[str, Any]):
        """Store analysis result for future reference"""
        self.previous_analyses[query] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        # Keep only last 10 analyses
        if len(self.previous_analyses) > 10:
            # Remove oldest entries
            keys_to_remove = list(self.previous_analyses.keys())[:-10]
            for key in keys_to_remove:
                del self.previous_analyses[key]
    
    def get_previous_analysis(self, query: str) -> Optional[Dict[str, Any]]:
        """Get previous analysis result if available"""
        return self.previous_analyses.get(query)
    
    def clear(self):
        """Clear the conversation memory"""
        self.messages = []
        self.conversation_summary = None
        self.topic_history = []
        self.previous_analyses = {}

class StateManager:
    """Manages state across the LangGraph flow using database storage"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationMemory] = {}
    
    def create_user_session(self, session_id: str):
        """Create a new user session"""
        db_manager.create_user_session(session_id)
    
    def get_or_create_session(self, session_id: str) -> SessionData:
        """Get or create a session using database"""
        # Check if session exists in database
        session_data = db_manager.get_user_session(session_id)
        
        if not session_data:
            # Create new session in database with the provided session_id
            db_manager.create_user_session(session_id)
            # Try to get the session again
            session_data = db_manager.get_user_session(session_id)
        
        # Update activity
        db_manager.update_user_activity(session_id)
        
        # Convert to SessionData model
        return SessionData(
            session_id=session_id,
            created_at=datetime.fromisoformat(session_data['created_at']) if session_data else datetime.now(),
            last_activity=datetime.fromisoformat(session_data['last_activity']) if session_data else datetime.now(),
            is_active=session_data['is_active'] if session_data else True,
            files=db_manager.get_user_files(session_id),
            conversation_history=[]
        )
    
    def get_or_create_conversation(self, session_id: str) -> ConversationMemory:
        """Get or create conversation memory for a session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationMemory(session_id)
        return self.conversations[session_id]
    
    def add_file_to_session(self, session_id: str, file_data: Dict[str, Any]):
        """Add a file to a session using database"""
        # Add file to database
        file_id = db_manager.add_file_record(
            session_id=session_id,
            original_filename=file_data['original_filename'],
            uploaded_filename=file_data['uploaded_filename'],
            file_type=file_data['file_type'],
            file_size=file_data['file_size'],
            metadata=file_data.get('metadata')
        )
        
        # Update file_data with the generated file_id
        file_data['file_id'] = file_id
    
    def get_session_files(self, session_id: str) -> List[Dict[str, Any]]:
        """Get files for a session from database"""
        return db_manager.get_user_files(session_id)
    
    def update_file_in_session(self, session_id: str, file_id: str, updated_file_data: Dict[str, Any]):
        """Update a file in a session using database"""
        # Update cleaned status in database
        db_manager.update_file_cleaned_status(
            file_id=file_id,
            cleaned_filename=updated_file_data.get('cleaned_filename', ''),
            cleaning_log=updated_file_data.get('cleaning_log'),
            impact_metrics=updated_file_data.get('impact_metrics')
        )
    
    def log_conversation(self, session_id: str, user_query: str, ai_response: str, metadata: Dict[str, Any] = None):
        """Log a conversation turn to database"""
        db_manager.log_user_action(
            session_id=session_id,
            action_type="conversation",
            action_details={
                "user_query": user_query,
                "ai_response": ai_response,
                "metadata": metadata or {}
            }
        )
    
    def clear_session(self, session_id: str):
        """Clear a session and its conversation"""
        db_manager.deactivate_user_session(session_id)
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def store_workflow_state(self, session_id: str, state: ConversationState):
        """Store a paused workflow state"""
        # Store in memory for quick access
        if not hasattr(self, 'workflow_states'):
            self.workflow_states = {}
        self.workflow_states[session_id] = state
    
    def get_workflow_state(self, session_id: str) -> Optional[ConversationState]:
        """Retrieve a paused workflow state"""
        if hasattr(self, 'workflow_states'):
            return self.workflow_states.get(session_id)
        return None
    
    def clear_workflow_state(self, session_id: str):
        """Clear a stored workflow state"""
        if hasattr(self, 'workflow_states') and session_id in self.workflow_states:
            del self.workflow_states[session_id]

# Global state manager
state_manager = StateManager()

def create_initial_state(session_id: str, user_query: str, file_ids: List[str] = None) -> ConversationState:
    """Create initial state for a conversation with intelligent memory"""
    conversation = state_manager.get_or_create_conversation(session_id)
    
    # Analyze if this query is about previous topics
    topic_analysis = conversation.is_query_about_previous_topic(user_query)
    
    # Add the user query to conversation history
    conversation.add_message(HumanMessage(content=user_query))
    
    return ConversationState(
        messages=conversation.messages,
        session_id=session_id,
        user_query=user_query,
        file_ids=file_ids or [],
        current_step="start",
        agent_scratchpad="",
        intermediate_steps=[],
        final_answer=None,
        error=None,
        plot_data=None,
        reasoning=None,
        conversation_summary=conversation.conversation_summary,
        conversation_context=conversation.get_full_context(),
        is_previous_topic=topic_analysis.get("is_previous_topic", False),
        previous_topic_summary=conversation.get_topic_summary(),
        metadata={
            "session_files": state_manager.get_session_files(session_id),
            "conversation_context": conversation.get_full_context(),
            "topic_analysis": topic_analysis
        }
    )

def update_state_with_response(state: ConversationState, response: str, plot_data: Dict = None, reasoning: str = None):
    """Update state with AI response and store in conversation memory"""
    conversation = state_manager.get_or_create_conversation(state["session_id"])
    
    # Add AI response to conversation
    ai_message = AIMessage(content=response)
    conversation.add_message(ai_message)
    
    # Store the analysis result for future reference
    analysis_result = {
        "response": response,
        "plot_data": plot_data,
        "reasoning": reasoning,
        "file_ids": state["file_ids"],
        "timestamp": datetime.now().isoformat()
    }
    conversation.store_analysis_result(state["user_query"], analysis_result)
    
    # Add topic to conversation history if it's a new analysis
    if not state.get("is_previous_topic", False):
        conversation.add_topic(state["user_query"], analysis_result)
    
    # Update state
    state["messages"] = conversation.messages
    state["final_answer"] = response
    state["plot_data"] = plot_data
    state["reasoning"] = reasoning
    state["current_step"] = "complete"
    
    # Log the conversation
    state_manager.log_conversation(
        state["session_id"],
        state["user_query"],
        response,
        {
            "plot_data": plot_data is not None,
            "reasoning": reasoning,
            "file_ids": state["file_ids"],
            "is_previous_topic": state.get("is_previous_topic", False)
        }
    )
    
    return state 