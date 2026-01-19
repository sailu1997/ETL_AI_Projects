"""
Workflow Logger for LangGraph

Provides detailed logging of workflow execution and state transitions.
"""

import time
from typing import Dict, Any

class WorkflowLogger:
    """Logs workflow execution with detailed flow information"""
    
    def __init__(self):
        self.execution_start = None
        self.node_count = 0
    
    def start_workflow(self, workflow_name: str, initial_state: Dict[str, Any]):
        """Log the start of a workflow"""
        self.execution_start = time.time()
        self.node_count = 0
        
        print(f"\n{'🚀'*20} WORKFLOW START {'🚀'*20}")
        print(f"📋 Workflow: {workflow_name}")
        print(f"⏰ Start Time: {time.strftime('%H:%M:%S')}")
        print(f"📊 Initial State Keys: {list(initial_state.keys())}")
        if 'session_id' in initial_state:
            print(f"🆔 Session: {initial_state['session_id']}")
        if 'user_query' in initial_state:
            print(f"❓ Query: {initial_state['user_query']}")
        print(f"{'🚀'*20} WORKFLOW START {'🚀'*20}\n")
    
    def log_node_transition(self, from_node: str, to_node: str, state: Dict[str, Any]):
        """Log transition between nodes"""
        self.node_count += 1
        elapsed = time.time() - self.execution_start if self.execution_start else 0
        
        print(f"\n{'➡️'*15} NODE TRANSITION {'➡️'*15}")
        print(f"🔄 Step {self.node_count}: {from_node} → {to_node}")
        print(f"⏱️  Elapsed Time: {elapsed:.2f}s")
        print(f"📊 State Keys: {list(state.keys())}")
        print(f"{'➡️'*15} NODE TRANSITION {'➡️'*15}\n")
    
    def end_workflow(self, final_state: Dict[str, Any]):
        """Log the end of a workflow"""
        total_time = time.time() - self.execution_start if self.execution_start else 0
        
        print(f"\n{'🏁'*20} WORKFLOW END {'🏁'*20}")
        print(f"📊 Total Nodes Executed: {self.node_count}")
        print(f"⏱️  Total Execution Time: {total_time:.2f}s")
        print(f"📊 Final State Keys: {list(final_state.keys())}")
        if 'current_step' in final_state:
            print(f"📍 Final Step: {final_state['current_step']}")
        if 'reasoning' in final_state:
            print(f"💭 Final Reasoning: {final_state['reasoning']}")
        print(f"{'🏁'*20} WORKFLOW END {'🏁'*20}\n")

# Global workflow logger instance
workflow_logger = WorkflowLogger() 