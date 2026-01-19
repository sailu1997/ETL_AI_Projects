from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any
from openai import OpenAI
import os
import re
from dotenv import load_dotenv
from tools.reasoning_curator import ReasoningCurator

load_dotenv()

router = APIRouter()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ReasoningRequest(BaseModel):
    query: str
    result: Any

@router.post("/reasoningaagent/")
async def ReasoningAgent(payload: ReasoningRequest):
    """Generates reasoning about the result using LLM and extracts thinking and final explanation."""
    prompt = ReasoningCurator(payload.query, payload.result)

    # Non-streaming LLM call (FastAPI backend shouldn't use Streamlit)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an insightful data analyst. Provide clear explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )

    full_response = response.choices[0].message.content
    
    # Extract thinking content between <think>...</think> tags
    thinking_match = re.search(r"<think>(.*?)</think>", full_response, flags=re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    
    # Remove thinking tags from final output
    cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
    
    return {"thinking_content": thinking_content, "cleaned": cleaned}