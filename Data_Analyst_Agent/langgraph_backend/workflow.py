"""
Main LangGraph Workflow

This orchestrates the complete file processing and analysis flow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import ConversationState
from nodes import (
    upload_file_node, clean_file_node, list_files_node,
    query_classifier_node, file_selector_node, analyze_data_node, smart_code_generator_node, smart_code_executor_node,
    intelligent_error_handler_node
)


def create_chat_analysis_workflow():
    """Create the complete chat analysis workflow with conversation memory"""
    
    # Disable LangChain tracing to avoid TracerException
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_ENDPOINT"] = ""
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""
    
    # Create the workflow
    workflow = StateGraph(ConversationState)
    
    # Add all the analysis nodes
    workflow.add_node("query_classifier", query_classifier_node().execute)
    workflow.add_node("file_selector", file_selector_node().execute)
    workflow.add_node("analyze_data", analyze_data_node().execute)
    workflow.add_node("smart_code_generator", smart_code_generator_node().execute)
    workflow.add_node("smart_code_executor", smart_code_executor_node().execute)

    # Add the single intelligent error handler node
    workflow.add_node("intelligent_error_handler", intelligent_error_handler_node().execute)
    
    # Set the entry point
    workflow.set_entry_point("query_classifier")
    
    # Define the main flow: query_classifier -> file_selector -> analyze_data -> smart_code_generator -> smart_code_executor
    workflow.add_edge("query_classifier", "file_selector")
    workflow.add_edge("file_selector", "analyze_data")
    workflow.add_edge("analyze_data", "smart_code_generator")
    workflow.add_edge("smart_code_generator", "smart_code_executor")
    
    # Define conditional routing from smart_code_executor based on success/failure
    def has_execution_error(state):
        execution_result = state.get("result", {}).get("execution_result", {})
        return not execution_result.get("success", False)

    workflow.add_conditional_edges(
        "smart_code_executor",
        has_execution_error,
        {
            True: "intelligent_error_handler",  # Go to error handler if there's an error
            False: END                          # End workflow if execution was successful
        }
    )
    
    # Define conditional routing for retry loop from intelligent_error_handler
    def should_retry(state):
        execution_result = state.get("result", {}).get("execution_result", {})
        return not execution_result.get("success", False) and state.get("fix_attempt", 0) < 3

    workflow.add_conditional_edges(
        "intelligent_error_handler",
        should_retry,
        {
            True: "intelligent_error_handler",  # Loop back to try again
            False: END                          # End workflow
        }
    )
    
    # Compile the workflow without checkpointing to avoid DataFrame serialization issues
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app

# Create the workflow instances
complete_app = create_chat_analysis_workflow()  # Use the chat analysis workflow for the complete app

