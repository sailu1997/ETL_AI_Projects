"""
Generate Workflow PNG

This script generates a PNG diagram of the LangGraph workflow using the built-in draw_mermaid_png() method.
"""

import os
from workflow import complete_app

def generate_workflow_png():
    """Generate PNG diagram of the workflow"""
    try:
        # Create directory if it doesn't exist
        os.makedirs("generated_plots", exist_ok=True)
        
        # Generate a visual representation of the state graph and save as PNG
        print("🔧 Generating workflow diagram...")
        
        # In LangGraph 0.2.0, we need to access the graph differently
        # The complete_app is already compiled, so we try to access its graph
        if hasattr(complete_app, 'get_graph'):
            image = complete_app.get_graph().draw_mermaid_png()
        elif hasattr(complete_app, 'graph'):
            image = complete_app.graph.draw_mermaid_png()
        else:
            # Try to access the underlying graph
            image = complete_app._graph.draw_mermaid_png()
        
        # Save the image to a file
        diagram_path = "generated_plots/langgraph_workflow.png"
        with open(diagram_path, "wb") as file:
            file.write(image)
        
        print(f"✅ Workflow diagram saved to: {diagram_path}")
        return diagram_path
        
    except Exception as e:
        print(f"❌ Error generating workflow PNG: {str(e)}")
        print("This might be due to LangGraph version compatibility issues.")
        print("💡 Trying alternative approach...")
        
        # Try using the built-in diagram generation from workflow.py
        try:
            from workflow import generate_workflow_diagram
            result = generate_workflow_diagram()
            if result:
                print(f"✅ Workflow diagram generated using alternative method: {result}")
                return result
        except Exception as e2:
            print(f"❌ Alternative method also failed: {str(e2)}")
        
        return None

if __name__ == "__main__":
    print("🎯 Generating LangGraph Workflow PNG Diagram...")
    result = generate_workflow_png()
    if result:
        print(f"📊 Workflow diagram available at: {result}")
    else:
        print("❌ Failed to generate workflow PNG")
        print("💡 You can still view the workflow structure using the show_workflow_structure.py script") 