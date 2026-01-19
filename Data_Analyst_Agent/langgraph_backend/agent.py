from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, ToolExecutor
from langchain_core.tools import BaseTool
import asyncio
from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage


class ConversationState(TypedDict):
    user_query: str
    messages: List[BaseMessage]
    session_id: str
    file_ids: List[str]
    dataframe_info: List[Dict[str, Any]]
    current_step: str
    error_info: str
    plot_data: Dict[str, Any]
    conversation_summary: str
    metadata: Dict[str, Any]

workflow = StateGraph(ConversationState)





