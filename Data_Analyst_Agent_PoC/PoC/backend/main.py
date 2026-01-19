from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.codegenerationagent import router as CodeGenerationAgent
from routes.executionagent import router as ExecutionAgent
from routes.insights import router as data_insight
from routes.reasoningagent import router as ReasoningAgent
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

app = FastAPI(title="Data analyst agent")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_insight)
app.include_router(CodeGenerationAgent)
app.include_router(ExecutionAgent)
app.include_router(ReasoningAgent)