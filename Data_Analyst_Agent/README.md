# Data Analyst Agent - Multi-Agent AI System

An intelligent data analysis system that converts natural language queries into executable Python code for data analysis and visualization. This repository showcases two implementations: a **Proof of Concept (PoC)** and an **advanced LangGraph-based backend**.

## Project Overview

This project demonstrates the evolution of a multi-agent AI system designed to democratize data analysis by allowing users to query CSV data using natural language instead of writing code.

### Problem Statement

Data analysis requires programming skills (Python, pandas, SQL), which creates barriers for non-technical users. Traditional BI tools lack flexibility, and writing analysis code is time-consuming.

### Solution

An AI-powered system that:
- Accepts natural language queries about uploaded CSV files
- Automatically generates Python code (pandas + plotly) to answer queries
- Executes code safely and returns results (tables, charts, insights)
- Maintains conversation context for follow-up questions
- Handles errors intelligently with automatic retry logic

## Repository Structure

```
DataAnalystAgent/
│
├── PoC/                          # Proof of Concept Implementation
│   ├── backend/                  # FastAPI backend with multi-agent system
│   │   ├── main.py              # API entry point
│   │   ├── routes/              # Agent endpoints
│   │   │   ├── reasoningagent.py       # Query understanding
│   │   │   ├── codegenerationagent.py  # Code generation
│   │   │   ├── executionagent.py       # Code execution
│   │   │   └── insights.py             # Data profiling
│   │   ├── models/              # Agent logic and OpenAI client
│   │   └── tools/               # Analysis utilities
│   ├── frontend/                # Streamlit web interface
│   │   └── main.py             # Interactive chat UI
│   ├── requirements.txt
│   ├── .gitignore
│   ├── env_template.txt
│   └── README.md               # PoC documentation
│
├── langgraph_backend/           # Advanced LangGraph Implementation
│   ├── main.py                  # FastAPI app with LangGraph integration
│   ├── workflow.py              # LangGraph workflow orchestration
│   ├── state.py                 # State management
│   ├── database.py              # SQLite session management
│   ├── nodes/                   # Specialized workflow nodes
│   │   ├── query_classifier.py     # Query type classification
│   │   ├── file_selector.py        # Intelligent file selection
│   │   ├── analyze_data.py         # Data structure analysis
│   │   ├── smart_code_generator.py # Context-aware code gen
│   │   ├── smart_code_executor.py  # Safe code execution
│   │   └── intelligent_error_handler.py  # Error recovery
│   ├── requirements.txt
│   ├── .gitignore
│   ├── env_template.txt
│   └── README.md               # LangGraph documentation
│
└── README.md                    # This file - Project overview
```

## Implementations Comparison

| Feature | PoC | LangGraph Backend |
|---------|-----|-------------------|
| **Architecture** | Simple multi-agent REST API | Stateful workflow orchestration |
| **Agents** | 4 separate endpoints | 6+ specialized nodes in workflow |
| **State Management** | Stateless (request/response) | Stateful with checkpointing |
| **Conversation Memory** | None | Persistent conversation context |
| **Error Handling** | Basic error messages | Intelligent retry with auto-fix (3 attempts) |
| **Query Classification** | Manual routing | Automatic query type detection |
| **File Management** | In-memory | SQLite with metadata tracking |
| **Streaming** | No | Server-Sent Events (SSE) |
| **Session Management** | None | Full session lifecycle |
| **Code Execution** | Simple exec() | Isolated execution with metrics |
| **Data Cleaning** | Manual | LLM-powered automatic cleaning |
| **Frontend** | Streamlit (included) | Separate (not included) |
| **Complexity** | Low | High |
| **Use Case** | Demo & learning | Production-ready prototype |

## Quick Start

### PoC (Simple Version)

1. **Navigate to PoC directory**:
```bash
cd PoC/
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp env_template.txt .env
# Add your OPENAI_API_KEY to .env
```

4. **Run backend**:
```bash
cd backend
python main.py
```

5. **Run frontend** (in new terminal):
```bash
cd frontend
streamlit run main.py
```

6. **Access UI**: Open browser to `http://localhost:8501`

### LangGraph Backend (Advanced Version)

1. **Navigate to langgraph_backend directory**:
```bash
cd langgraph_backend/
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp env_template.txt .env
# Add your OPENAI_API_KEY to .env
```

4. **Run backend**:
```bash
python main.py
```

5. **Test API**: Backend runs on `http://localhost:8000`
   - Health check: `curl http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`

## Use Cases & Examples

### Example Queries

Both implementations support natural language queries like:

**Statistical Analysis:**
- "What is the average monthly rent by flat type?"
- "Show me the top 5 most expensive properties"
- "Calculate the correlation between rent and area"

**Visualizations:**
- "Create a bar chart of monthly rent by flat type"
- "Show me a pie chart of property distribution by location"
- "Plot a scatter chart of area vs rent"

**Data Filtering:**
- "Show me all properties with rent above $3000"
- "Filter 3-room flats in the central region"
- "Display properties with area greater than 1000 sqft"

**Follow-up Questions** (LangGraph only):
- "Make it a line chart instead"
- "What about 4-room flats?"
- "Show the same data as a pie chart"

## Technology Stack

### Core Technologies
- **Python 3.9+**: Primary language
- **FastAPI**: REST API framework
- **OpenAI GPT-4o-mini**: LLM for code generation and query understanding
- **Pandas**: Data manipulation
- **Plotly**: Interactive visualizations

### PoC Specific
- **Streamlit**: Frontend web interface
- **Pydantic**: Data validation

### LangGraph Specific
- **LangGraph**: Workflow orchestration
- **LangChain**: LLM abstractions
- **SQLite**: Session and file management
- **aiosqlite**: Async database operations

## Multi-Agent Architecture

### PoC Architecture

```
User Query → Reasoning Agent → Code Generation Agent → Execution Agent → Results
                    ↓
              Insights Agent (for data profiling)
```

**Agents:**
1. **Reasoning Agent**: Understands query intent and context
2. **Code Generation Agent**: Generates pandas/plotly code
3. **Execution Agent**: Safely executes generated code
4. **Insights Agent**: Provides data profiling and summary statistics

### LangGraph Architecture

```
Query → Classifier → File Selector → Analyze Data → Code Gen → Execute
            ↓                                                      ↓
       Conversation Memory                                    Error Handler
                                                                    ↓
                                                             Retry (max 3x)
```

**Nodes:**
1. **Query Classifier**: Determines query type and response format
2. **File Selector**: Intelligently selects relevant files
3. **Analyze Data**: Extracts metadata and structure
4. **Code Generator**: Context-aware code generation
5. **Code Executor**: Safe execution with metrics
6. **Error Handler**: Automatic error recovery

## Key Features

### PoC Features
✅ Natural language to Python code generation
✅ Multi-agent system with specialized roles
✅ Interactive Streamlit interface
✅ Data profiling with YData Profiling
✅ Support for pandas and plotly operations
✅ Basic error handling
✅ CSV file upload

### LangGraph Features
✅ All PoC features, plus:
✅ **Stateful conversation management**
✅ **Automatic query classification**
✅ **Intelligent file selection**
✅ **LLM-powered data cleaning**
✅ **Streaming responses (SSE)**
✅ **Session lifecycle management**
✅ **Persistent conversation memory**
✅ **Automatic error recovery with retry**
✅ **Database-backed file tracking**
✅ **Workflow visualization support**

## Project Evolution

### Phase 1: PoC (Proof of Concept)
- **Goal**: Validate the concept of natural language data analysis
- **Approach**: Simple multi-agent REST API with Streamlit frontend
- **Result**: Successfully demonstrated feasibility with basic functionality

### Phase 2: LangGraph Backend (Advanced Implementation)
- **Goal**: Build production-ready system with state management
- **Approach**: LangGraph workflow orchestration with intelligent agents
- **Result**: Robust system with conversation memory, error recovery, and session management

### Future Phases (Potential)
- Phase 3: Add code sandboxing for security
- Phase 4: Support for SQL databases and multiple file formats
- Phase 5: Multi-user support with authentication
- Phase 6: Real-time collaboration features

## Development Journey

### Challenges Solved

1. **Code Generation Accuracy**: 
   - PoC: Basic prompt engineering
   - LangGraph: Context-aware prompts with file metadata

2. **Error Handling**:
   - PoC: Simple error messages
   - LangGraph: Intelligent retry with LLM-powered fixes

3. **Conversation Context**:
   - PoC: No context between queries
   - LangGraph: Full conversation memory with follow-up support

4. **File Management**:
   - PoC: In-memory only
   - LangGraph: Persistent SQLite storage with metadata

5. **State Management**:
   - PoC: Stateless
   - LangGraph: Stateful with checkpointing

## Lessons Learned

1. **Start Simple**: PoC validated the concept before building complex architecture
2. **Iterative Development**: Each implementation built on lessons from the previous
3. **State is Critical**: Conversation memory dramatically improves UX
4. **Error Recovery**: Automatic retry logic significantly improves reliability
5. **Separation of Concerns**: Node-based architecture enables easier testing and maintenance

## Performance Considerations

### PoC
- **Response Time**: 2-5 seconds per query
- **Scalability**: Single-user, no session management
- **Memory**: Entire DataFrame loaded per request

### LangGraph
- **Response Time**: 3-7 seconds per query (includes classification + generation)
- **Scalability**: Multi-user with session isolation
- **Memory**: On-demand loading, metadata-based optimization

## Security Considerations

⚠️ **Both implementations execute generated code - use with caution!**

**Current Limitations:**
- Code runs in same process (no sandboxing)
- No input sanitization for generated code
- API keys stored in environment variables

**For Production:**
- Implement code sandboxing (Docker containers, restricted environments)
- Add rate limiting and authentication
- Validate and sanitize all inputs
- Use secrets management service for API keys

## Future Enhancements

### Short Term
- [ ] Add unit tests and integration tests
- [ ] Implement code sandboxing
- [ ] Add support for Excel and JSON files
- [ ] Create frontend for LangGraph backend

### Long Term
- [ ] Support for SQL databases
- [ ] Multi-file joins and aggregations
- [ ] User authentication and authorization
- [ ] Real-time collaborative analysis
- [ ] Workflow visualization dashboard
- [ ] Export results to multiple formats

## Environment Variables

Both implementations require:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (LangGraph only)
LANGCHAIN_TRACING_V2=false
HOST=0.0.0.0
PORT=8000
```

## Testing

### PoC Testing
1. Upload a CSV file via Streamlit UI
2. Ask natural language questions
3. View generated code, execution results, and visualizations

### LangGraph Testing
1. Use API endpoints (Postman, curl, or Python requests)
2. Test with `/docs` interactive API documentation
3. Monitor logs for workflow execution details

## Documentation

- **PoC README**: `PoC/README.md` - Detailed PoC documentation
- **LangGraph README**: `langgraph_backend/README.md` - Comprehensive LangGraph guide
- **API Docs**: `http://localhost:8000/docs` (when LangGraph backend is running)

## Contributing

This is a demonstration project. If you find issues or have suggestions:
1. Review the architecture and implementation
2. Consider the trade-offs between simplicity (PoC) and features (LangGraph)
3. Test changes thoroughly before deployment

## License

This project is for demonstration and educational purposes.

## Acknowledgments

- **OpenAI**: GPT-4o-mini for code generation
- **LangChain/LangGraph**: Workflow orchestration framework
- **Streamlit**: Rapid frontend prototyping
- **FastAPI**: Modern Python web framework

## Contact

For questions about this project, please review the detailed READMEs in each implementation folder.

---

**Note**: This project demonstrates AI-powered code generation. Always review generated code before production use and implement proper security measures including code sandboxing, input validation, and access controls.
