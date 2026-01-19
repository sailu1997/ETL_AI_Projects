# Data Analyst Agent - LangGraph Backend

An advanced multi-agent system for intelligent data analysis using **LangGraph** workflow orchestration. This implementation provides stateful conversation management, intelligent query classification, and automatic code generation/execution for data analysis tasks.

## Overview

This backend uses LangGraph to orchestrate a complex workflow of specialized nodes that work together to understand queries, select appropriate data, generate Python code, and execute analysis tasks.

### Key Features

- **LangGraph Workflow Orchestration**: Stateful workflow with checkpointing and error recovery
- **Multi-Node Architecture**: Specialized nodes for each stage of analysis
- **Intelligent Query Classification**: Automatically determines query type and response format
- **File Management**: Upload, clean, and process CSV files with metadata tracking
- **Smart Code Generation**: Context-aware Python code generation using GPT-4o-mini
- **Automatic Execution**: Safe code execution with intelligent error handling
- **Conversation Memory**: Maintains context across multiple queries
- **Session Management**: SQLite-based session and file tracking
- **Streaming Responses**: Real-time streaming of analysis results

## Architecture

### Workflow Nodes

```
User Query
    ↓
query_classifier → Classifies query type and determines response format
    ↓
file_selector → Selects relevant files for the query
    ↓
analyze_data → Analyzes data structure and relationships
    ↓
smart_code_generator → Generates Python code using LLM
    ↓
smart_code_executor → Executes generated code safely
    ↓ (if error)
intelligent_error_handler → Attempts to fix errors (max 3 retries)
    ↓
Result/END
```

### Node Descriptions

1. **Query Classifier**: Analyzes natural language query to determine:
   - Query type (visualization, filtering, aggregation, etc.)
   - Response format (plot, table, or mixed)
   - Required operations

2. **File Selector**: Intelligently selects relevant files based on:
   - Query content analysis
   - File metadata and column information
   - Conversation history

3. **Analyze Data**: Extracts and provides:
   - DataFrame structure and statistics
   - Column types and null counts
   - Sample data for context

4. **Smart Code Generator**: Generates Python code with:
   - Context-aware prompts based on query type
   - Conversation memory for follow-up queries
   - Plotly for visualizations, pandas for data manipulation

5. **Smart Code Executor**: Safely executes code in isolated environment with:
   - Proper error capturing
   - Result extraction (plots, tables, or text)
   - Execution metrics

6. **Intelligent Error Handler**: Attempts to fix execution errors by:
   - Analyzing error messages
   - Regenerating corrected code
   - Retry logic (up to 3 attempts)

## Technology Stack

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: Core abstractions for LLM interactions
- **FastAPI**: RESTful API endpoints
- **OpenAI GPT-4o-mini**: Code generation and query understanding
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **SQLite**: Session and file metadata storage
- **Python 3.9+**: Core runtime

## API Endpoints

### Core Endpoints

```
POST /chat_stream
    - Stream analysis results in real-time
    - Request: { session_id, query }
    - Response: Server-Sent Events (SSE) stream

POST /session/create
    - Create new user session
    - Returns: session_id

POST /session/validate
    - Validate session exists and is active
    - Request: { session_id }

GET /session/info/{session_id}
    - Get session information and file list

POST /session/deactivate
    - Deactivate user session
    - Request: { session_id }
```

### File Management

```
POST /upload_file
    - Upload CSV file
    - Request: multipart/form-data
    - Returns: file_id

POST /clean_data/
    - Clean uploaded data using LLM-powered cleaning
    - Request: { session_id, file_id }
    - Returns: cleaned_file_id, cleaning_log

GET /files/list
    - List all files for session
    - Query: session_id
```

### Insights & Analysis

```
POST /insights/
    - Get data insights (summary statistics, profiling)
    - Request: { session_id, file_ids, insight_type }
    - Response: Pandas profiling or summary statistics
```

### Health Check

```
GET /health
    - Health check endpoint
    - Returns: { status, service, version }
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- OpenAI API key

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**:
```bash
cp env_template.txt .env
# Edit .env and add your OpenAI API key
```

3. **Run the server**:
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

## Environment Variables

Create a `.env` file based on `env_template.txt`:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LangChain Configuration
LANGCHAIN_TRACING_V2=false

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true

# Database Configuration  
DATABASE_PATH=user_sessions.db
```

## Usage Examples

### 1. Create Session
```bash
curl -X POST "http://localhost:8000/session/create"
```

### 2. Upload File
```bash
curl -X POST "http://localhost:8000/upload_file" \
  -F "file=@data.csv" \
  -F "session_id=your_session_id"
```

### 3. Clean Data
```bash
curl -X POST "http://localhost:8000/clean_data/" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your_session_id", "file_id": "your_file_id"}'
```

### 4. Query Data (Streaming)
```bash
curl -X POST "http://localhost:8000/chat_stream" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your_session_id", "query": "Show me a bar chart of monthly rent by flat type"}'
```

## State Management

The system uses a sophisticated state management approach:

### ConversationState Schema

```python
{
    "session_id": str,              # Unique session identifier
    "user_query": str,              # Current query
    "current_step": str,            # Current workflow step
    "messages": List[BaseMessage],  # LangChain message history
    "file_ids": List[str],          # Associated file IDs
    "result": Dict,                 # Analysis results
    "error": Optional[str],         # Error information
    "query_classification": Dict,   # Query type and format
    "conversation_memory": Dict,    # Cross-query context
    "previous_analysis": Dict       # Last analysis results
}
```

### Database Schema

- **user_sessions**: Tracks active user sessions
- **user_files**: Stores file metadata and cleaning logs
- **conversation_history**: Maintains query/response history

## Conversation Memory

The system maintains conversation context to handle follow-up queries:

- **Analysis History**: Recent queries and results
- **Data Relationships**: Identified column relationships
- **Context Clues**: Query intent and continuation markers
- **File Metadata**: Cached DataFrame information

This enables natural follow-up queries like:
- "Show me a pie chart" (after "What is the average rent by flat type?")
- "Make it a line chart instead" (switching visualization types)
- "What about 3-room flats?" (filtering based on previous context)

## Error Handling

The intelligent error handler provides:

1. **Automatic Retry**: Up to 3 attempts to fix errors
2. **Error Analysis**: LLM analyzes error messages
3. **Code Correction**: Generates fixed code based on error context
4. **Graceful Degradation**: Returns partial results when possible

## Development

### Project Structure

```
langgraph_backend/
├── main.py                    # FastAPI app entry point
├── workflow.py                # LangGraph workflow definition
├── state.py                   # State management and schemas
├── agent.py                   # Legacy agent code
├── database.py                # SQLite database manager
├── api_integration.py         # API endpoints
├── workflow_logger.py         # Logging utilities
├── nodes/                     # Workflow nodes
│   ├── __init__.py
│   ├── base_node.py          # Base node class
│   ├── query_classifier.py   # Query classification
│   ├── file_selector.py      # File selection logic
│   ├── analyze_data.py       # Data analysis
│   ├── smart_code_generator.py    # Code generation
│   ├── smart_code_executor.py     # Code execution
│   ├── intelligent_error_handler.py   # Error handling
│   ├── upload_file.py        # File upload
│   ├── clean_file.py         # Data cleaning
│   ├── list_files.py         # File listing
│   ├── get_dataframe_info.py # DataFrame inspection
│   └── clarification_node.py # Clarification requests
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

### Adding New Nodes

1. Create a new node class inheriting from `BaseNode`
2. Implement `execute(state: ConversationState) -> ConversationState`
3. Add node to workflow in `workflow.py`
4. Define edges connecting to/from your node

### Testing

Run the backend and test with curl or a REST client:

```bash
# Terminal 1: Run backend
python main.py

# Terminal 2: Test endpoints
curl http://localhost:8000/health
```

## Limitations

- Supports CSV files only
- Code execution is not sandboxed (use with caution in production)
- Limited to pandas and plotly for data manipulation/visualization
- Session data stored in local SQLite (not suitable for distributed systems)

## Security Considerations

1. **API Key Security**: Never commit `.env` files
2. **Code Execution**: Generated code runs in the same process (potential security risk)
3. **Input Validation**: Validate all file uploads and user inputs
4. **Rate Limiting**: Consider adding rate limiting for production use
5. **Database**: Use proper authentication for production databases

## Future Enhancements

- [ ] Add code sandboxing for secure execution
- [ ] Support more file formats (Excel, JSON, Parquet)
- [ ] Implement user authentication and authorization
- [ ] Add support for SQL databases
- [ ] Create workflow visualization dashboard
- [ ] Add unit tests and integration tests
- [ ] Implement caching for frequently run queries
- [ ] Add support for multi-file joins and merges

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Errors**: Check your API key in `.env`

3. **Database Errors**: Delete `user_sessions.db` to reset

4. **Port Already in Use**: Change `PORT` in `.env` or kill the process using port 8000

## License

This project is for demonstration and educational purposes.

## Contact

For questions or issues, please review the code and documentation carefully.
